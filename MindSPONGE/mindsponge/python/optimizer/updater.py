# Copyright 2021 The AIMM Group at Shenzhen Bay Laboratory & Peking University
#
# Developer: Yi Isaac Yang, Dechin Chen, Jun Zhang, Yijie Xia
#
# Email: yangyi@szbl.ac.cn
#
# This code is a part of MindSPONGE.
#
# The Cybertron-Code is open-source software based on the AI-framework:
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
"""Space updater"""

import mindspore as ms
import mindspore.numpy as msnp
from mindspore import Tensor, Parameter
from mindspore import ops
from mindspore.ops import functional as F
from mindspore.nn import CellList
from mindspore.nn import Optimizer
from mindspore.nn.optim.optimizer import opt_init_args_register
from mindspore.common.initializer import initializer

from ..system import Molecule
from ..control import Controller
from ..function import functions as func

class Updater(Optimizer):
    r"""Optimizer to update parameters of space (coordinates and PBC box)

    Args:

        system (Molecule):          Simulation system.

        controller (Controller):    Controller.

        time_step (float):          Time step. Defulat: 1e-3

        velocity (Tensor):          Tensor of shape (B, A, D). Data type is float.
                                    Default: None

        weight_decay (float):       An value for the weight decay. Default: 0

        loss_scale (float):         A value for the loss scale. Default: 1

    Symbols:

        B:  Batchsize, i.e. number of walkers in simulation

        A:  Number of atoms.

        D:  Dimension of the simulation system. Usually is 3.

    """
    @opt_init_args_register
    def __init__(self,
                 system: Molecule,
                 controller: Controller = None,
                 time_step: float = 1e-3,
                 velocity: Tensor = None,
                 weight_decay: float = 0.0,
                 loss_scale: float = 1.0,
                 ):

        super().__init__(
            learning_rate=time_step,
            parameters=system.space_parameters(),
            weight_decay=weight_decay,
            loss_scale=loss_scale,
        )

        self.time_step = Tensor(time_step, ms.float32)

        self.system = system
        self.coordinate = self.system.coordinate
        self.pbc_box = self.system.pbc_box

        # (B,A)
        self.atom_mass = self.system.atom_mass
        self.inv_mass = self.system.inv_mass
        # (B,A,1)
        self._atom_mass = F.expand_dims(self.atom_mass, -1)
        self._inv_mass = F.expand_dims(self.inv_mass, -1)

        self.num_walker = system.num_walker
        self.num_atoms = system.num_atoms
        self.dimension = system.dimension

        self.units = self.system.units
        self.boltzmann = self.units.boltzmann
        self.kinetic_unit_scale = self.units.kinetic_ref
        self.press_unit_scale = self.units.pressure_ref

        if velocity is None:
            self.velocity = Parameter(msnp.zeros_like(self.coordinate), name='velocity')
        else:
            velocity = Tensor(velocity)
            if velocity.ndim == 2:
                velocity = F.expand_dims(velocity, 0)
            if velocity.shape != self.coordinate.shape:
                raise ValueError('The shape of velocity '+str(velocity.shape) +
                                 'must be equal to the shape of coordinate '+str(self.coordinate.shape)+'!')
            self.velocity = Parameter(velocity, name='velocity')

        self.num_constraints = 0
        self.num_controller = 0
        if controller is None:
            self.controller = None
        else:
            if isinstance(controller, Controller):
                self.num_controller = 1
                controller = [controller]
            elif isinstance(controller, list):
                self.num_controller = len(controller)
            else:
                raise TypeError('The type of "controller" must be Controller or list but got: '
                                +str(type(controller)))

            self.controller = CellList(controller)
            for i in range(self.num_controller):
                self.num_constraints += self.controller[i].num_constraints

        self.degrees_of_freedom = system.degrees_of_freedom - self.num_constraints

        self.identity = ops.Identity()

        self.kinetics = None
        self.temperature = None
        if self.velocity is not None:
            kinetics = self.get_kinetics(self.velocity)
            temperature = self.get_temperature(kinetics)
            # (B,D)
            self.kinetics = Parameter(kinetics, name="kinetics")
            # (B)
            self.temperature = Parameter(temperature, name="temperature")

        self.virial = None
        self.pressure = None
        if self.pbc_box is not None:
            # (B,D)
            self.virial = Parameter(initializer(
                'zeros', (self.num_walker, self.dimension), ms.float32), name="virial")
            self.pressure = Parameter(initializer(
                'zeros', (self.num_walker, self.dimension), ms.float32), name="pressure")

        self.step = Parameter(Tensor(0, ms.int32), name='updater_step')

        if controller is not None:
            for i in range(self.num_controller):
                self.controller[i].set_time_step(self.time_step)
                self.controller[i].set_degrees_of_freedom(self.degrees_of_freedom)

    def set_step(self, step: int = 0):
        """set time step"""
        step = Tensor(step, ms.int32)
        F.depend(True, F.assign(self.step, step))
        return self

    def update_coordinate(self, coordinate: Tensor, success: bool = True) -> bool:
        """update the parameters of coordinate"""
        success = F.depend(success, F.assign(self.coordinate, coordinate))
        return success

    def update_pbc_box(self, pbc_box: Tensor, success: bool = True) -> bool:
        """update the parameters of PBC box"""
        if self.pbc_box is None:
            return success
        return F.depend(success, F.assign(self.pbc_box, pbc_box))

    def update_velocity(self, velocity: Tensor, success: bool = True) -> bool:
        """update the parameters of velocity"""
        return F.depend(success, F.assign(self.velocity, velocity))

    def update_kinetics(self, kinetics: Tensor, success: bool = True) -> bool:
        """update the parameters of kinects"""
        if self.kinetics is None:
            return success
        return F.depend(success, F.assign(self.kinetics, kinetics))

    def update_temperature(self, temperature: Tensor, success: bool = True) -> bool:
        """update the parameters of temperature"""
        if self.temperature is None:
            return success
        return F.depend(success, F.assign(self.temperature, temperature))

    def update_virial(self, virial: Tensor, success: bool = True) -> bool:
        """update the parameters of virial"""
        if self.pbc_box is None:
            return success
        return F.depend(success, F.assign(self.virial, virial))

    def update_pressure(self, pressure: Tensor, success: bool = True) -> bool:
        """update the parameters of pressure"""
        if self.pbc_box is None:
            return success
        return F.depend(success, F.assign(self.pressure, pressure))

    def get_velocity(self) -> Tensor:
        """get velocity"""
        if self.velocity is None:
            return None
        return self.identity(self.velocity)

    def get_kinetics(self, velocity: Tensor) -> Tensor:
        """get kinectics"""
        # (B,A,D)
        kinetics = 0.5 * self._atom_mass * velocity**2
        # (B,D) <- (B,A,D)
        kinetics = F.reduce_sum(kinetics, -2)
        return kinetics * self.kinetic_unit_scale

    def get_temperature(self, kinetics: Tensor = None) -> Tensor:
        """get temperature"""
        # (B) <- (B,D)
        kinetics = F.reduce_sum(kinetics, -1)
        return 2 * kinetics / self.degrees_of_freedom / self.boltzmann

    def get_pressure(self, kinetics: Tensor, virial: Tensor, pbc_box: Tensor) -> Tensor:
        """get pressure"""
        if self.pbc_box is None:
            return None
        # (B,D) = ((B,D) - (B, D)) / (B,1)
        volume = func.keepdim_prod(pbc_box, -1)
        pressure = 2 * (kinetics - virial) / volume
        return pressure * self.press_unit_scale

    def get_virial(self, pbc_grad, pbc_box):
        """get virial"""
        # (B,D)
        return 0.5 * pbc_grad * pbc_box

    def get_dt(self):
        """get time step"""
        return self.get_lr()

    def next_step(self, success: bool = True) -> bool:
        """finish the current optimization step and move to next step"""
        return F.depend(success, F.assign(self.step, self.step+1))

    def construct(self, gradients: tuple, loss: Tensor = None):
        """update the parameters of system"""

        gradients = self.decay_weight(gradients)
        gradients = self.scale_grad(gradients)

        coordinate = self.coordinate
        velocity = self.velocity
        force = -gradients[0]
        energy = loss
        kinetics = self.kinetics
        pbc_box = self.pbc_box
        virial = None
        if self.pbc_box is not None:
            virial = self.get_virial(gradients[1], pbc_box)

        step = self.identity(self.step)
        if self.controller is not None:
            for i in range(self.num_controller):
                coordinate, velocity, force, energy, kinetics, virial, pbc_box = \
                    self.controller[i](coordinate, velocity, force, energy, kinetics, virial, pbc_box, step)

        temperature = self.get_temperature(kinetics)
        pressure = self.get_pressure(kinetics, virial, pbc_box)

        success = True
        success = self.update_coordinate(coordinate, success)
        success = self.update_velocity(velocity, success)
        success = self.update_pbc_box(pbc_box, success)
        success = self.update_kinetics(kinetics, success)
        success = self.update_temperature(temperature, success)
        success = self.update_virial(virial, success)
        success = self.update_pressure(pressure, success)

        return self.next_step(success)
