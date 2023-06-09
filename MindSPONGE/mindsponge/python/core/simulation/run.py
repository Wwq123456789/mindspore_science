# Copyright 2021-2023 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of MindSPONGE:
# MindSpore Simulation Package tOwards Next Generation molecular modelling.
#
# MindSPONGE is open-source software based on the AI-framework:
# MindSpore (https://www.mindspore.cn/)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
RunOneStepCell
"""

from typing import Tuple
from mindspore import ops
from mindspore.ops import functional as F
from mindspore import Tensor
from mindspore import jit
from mindspore.nn import Cell

from mindspore.parallel._utils import (_get_device_num, _get_gradients_mean,
                                       _get_parallel_mode)
from mindspore.context import ParallelMode
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.nn.optim import Optimizer

from .energy import WithEnergyCell
from .force import WithForceCell
from ...function.functions import get_integer, all_none
from ...optimizer import Updater


class RunOneStepCell(Cell):
    r"""Cell to run one step simulation.

        This Cell wraps the `energy` and `force` with the `optimizer`. The backward graph will be created
        in the construct function to update the atomic coordinates of the simulation system.

    Args:

        energy (WithEnergyCell):    Cell that wraps the simulation system with
                                    the potential energy function.
                                    Defatul: None

        force (WithForceCell):      Cell that wraps the simulation system with
                                    the atomic force function.
                                    Defatul: None

        optimizer (Optimizer):      Optimizer for simulation. Defatul: None

        steps (int):                Steps for JIT. Default: 1

        sens (float):               The scaling number to be filled as the input of backpropagation.
                                    Default: 1.0

    Supported Platforms:

        ``Ascend`` ``GPU``

    Symbols:

        B:  Batchsize, i.e. number of walkers of the simulation.

        A:  Number of the atoms in the simulation system.

        D:  Spatial dimension of the simulation system. Usually is 3.

    """
    def __init__(self,
                 energy: WithEnergyCell = None,
                 force: WithForceCell = None,
                 optimizer: Optimizer = None,
                 steps: int = 1,
                 sens: float = 1.0,
                 ):

        super().__init__(auto_prefix=False)

        if all_none([energy, force]):
            raise ValueError('energy and force cannot be both None!')

        self._neighbour_list_pace = None

        self.system_with_energy = energy
        if self.system_with_energy is not None:
            self.system = self.system_with_energy.system
            self.units = self.system_with_energy.units
            self.system_with_energy.set_grad()
            self._neighbour_list_pace = self.system_with_energy.neighbour_list_pace

        self.system_with_force = force
        if self.system_with_force is not None:
            self.system_with_force.set_grad()

            force_pace = self.system_with_force.neighbour_list_pace

            if self.system_with_energy is None or self._neighbour_list_pace == 0:
                self._neighbour_list_pace = force_pace
            else:
                if force_pace not in (self._neighbour_list_pace, 0):
                    raise ValueError(
                        f'The neighbour_list_pace in WithForceCell ({force_pace}) cannot match '
                        f'the neighbour_list_pace in WithEnergyCell ({self._neighbour_list_pace}).')

        if self.system_with_energy is None:
            self.system = self.system_with_force.system

        self.optimizer = optimizer
        if self.optimizer is None:
            print('[WARNING] No optimizer! The simulation system will not be updated!')

        self.use_updater = isinstance(self.optimizer, Updater)
        self.weights = self.optimizer.parameters

        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = F.identity
        self.parallel_mode = _get_parallel_mode()
        self.reducer_flag = self.parallel_mode in (
            ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL)
        if self.reducer_flag:
            self.mean = _get_gradients_mean()
            self.degree = _get_device_num()
            self.grad_reducer = DistributedGradReducer(
                self.weights, self.mean, self.degree)

        self.steps = get_integer(steps)

    @property
    def neighbour_list_pace(self) -> int:
        r"""update step for neighbour list

        Return:
            int, steps

        """
        return self._neighbour_list_pace

    @property
    def energy_cutoff(self) -> Tensor:
        r"""cutoff distance for neighbour list in WithEnergyCell

        Return:
            Tensor, cutoff

        """
        if self.system_with_energy is None:
            return None
        return self.system_with_energy.cutoff

    @property
    def force_cutoff(self) -> Tensor:
        r"""cutoff distance for neighbour list in WithForceCell

        Return:
            Tensor, cutoff

        """
        if self.system_with_force is None:
            return None
        return self.system_with_force.cutoff

    @property
    def length_unit(self) -> str:
        r"""length unit

        Return:
            str, length unit

        """
        return self.units.length_unit

    @property
    def energy_unit(self) -> str:
        r"""energy unit

        Return:
            str, energy unit

        """
        return self.units.energy_unit

    @property
    def num_energies(self) -> int:
        r"""number of energy terms :math:`U`

        Return:
            int, number of energy terms

        """
        if self.system_with_energy is None:
            return 0
        return self.system_with_energy.num_energies

    @property
    def energy_names(self) -> list:
        r"""names of energy terms

        Return:
            list of str, names of energy terms

        """
        if self.system_with_energy is None:
            return []
        return self.system_with_energy.energy_names

    @property
    def bias_names(self) -> list:
        r"""name of bias potential energies

        Return:
            list of str, the bias potential energies

        """
        if self.system_with_energy is None:
            return []
        return self.system_with_energy.bias_names

    @property
    def num_biases(self) -> int:
        r"""number of bias potential energies :math:`V`

        Return:
            int, number of bias potential energies

        """
        if self.system_with_energy is None:
            return 0
        return self.system_with_energy.num_biases

    @property
    def energies(self) -> Tensor:
        r"""Tensor of potential energy components.

        Return:
            energies(Tensor):   Tensor of shape `(B, U)`. Data type is float.

        """
        if self.system_with_energy is None:
            return None
        return self.system_with_energy.energies

    @property
    def biases(self) -> Tensor:
        r"""Tensor of bias potential components.

        Return:
            biases(Tensor): Tensor of shape `(B, V)`. Data type is float.

        """
        if self.system_with_energy is None:
            return None
        return self.system_with_energy.biases

    @property
    def bias(self) -> Tensor:
        r"""Tensor of the total bias potential.

        Return:
            bias(Tensor): Tensor of shape `(B, 1)`. Data type is float.

        """
        if self.system_with_energy is None:
            return None
        return self.system_with_energy.bias

    @property
    def bias_function(self) -> Cell:
        r"""Cell of bias potential function"""
        if self.system_with_energy is None:
            return None
        return self.system_with_energy.bias_function

    def update_neighbour_list(self):
        r"""update neighbour list"""
        if self.system_with_energy is not None:
            self.system_with_energy.update_neighbour_list()
        if self.system_with_force is not None:
            self.system_with_force.update_neighbour_list()
        return self

    def update_bias(self, step: int):
        r"""update bias potential

        Args:
            step (int): Simulatio step.

        """
        if self.system_with_energy is not None:
            self.system_with_energy.update_bias(step)
        return self

    def update_wrapper(self, step: int):
        r"""update energy wrapper

        Args:
            step (int): Simulatio step.

        """
        if self.system_with_energy is not None:
            self.system_with_energy.update_wrapper(step)
        return self

    def update_modifier(self, step: int):
        r"""update force modifier

        Args:
            step (int): Simulatio step.

        """
        if self.system_with_force is not None:
            self.system_with_force.update_modifier(step)
        return self

    def set_pbc_grad(self, value: bool):
        r"""set whether to calculate the gradient of PBC box"""
        if self.system_with_energy is not None:
            self.system_with_energy.set_pbc_grad(value)
        if self.system_with_force is not None:
            self.system_with_force.set_pbc_grad(value)
        return self

    def set_steps(self, steps: int):
        r"""set steps for JIT

        Args:
            step (int): Simulatio step.

        """
        self.steps = get_integer(steps)
        return self

    @jit
    def run_one_step(self, *inputs):
        r"""run one step simulation

        Returns:
            energy (Tensor):    Tensor of shape `(B, 1)`. Data type is float.
                                Total potential energy.
            force (Tensor):     Tensor of shape `(B, A, D)`. Data type is float.
                                Atomic force.
        Symbols:
            B:  Batchsize, i.e. number of walkers of the simulation.
            A:  Number of the atoms in the simulation system.
            D:  Spatial dimension of the simulation system. Usually is 3.

        """
        energy = 0
        force = 0
        virial = None
        if self.system_with_energy is not None:
            energy = self.system_with_energy(*inputs)

            sens = F.fill(energy.dtype, energy.shape, self.sens)
            grads = self.grad(self.system_with_energy, self.weights)(*inputs, sens)

            force = -grads[0]
            if len(grads) > 1:
                virial = 0.5 * grads[1] * self.system.pbc_box

        if self.system_with_force is not None:
            energy, force, virial = self.system_with_force(energy, force, virial)

        if self.optimizer is not None:
            if self.use_updater:
                energy = F.depend(energy, self.optimizer(energy, force, virial))
            else:
                grads = (-force,)
                energy = F.depend(energy, self.optimizer(grads))

        return energy, force

    def construct(self, *inputs) -> Tuple[Tensor, Tensor]:
        r"""run simulation

        Returns:
            energy (Tensor):    Tensor of shape `(B, 1)`. Data type is float.
                                Total potential energy.
            force (Tensor):     Tensor of shape `(B, A, D)`. Data type is float.
                                Atomic force.

        Symbols:
            B:  Batchsize, i.e. number of walkers of the simulation.
            A:  Number of the atoms in the simulation system.
            D:  Spatial dimension of the simulation system. Usually is 3.

        """
        if self.steps == 1:
            return self.run_one_step(*inputs)

        energy = None
        force = None
        for _ in range(self.steps):
            energy, force = self.run_one_step(*inputs)

        return energy, force
