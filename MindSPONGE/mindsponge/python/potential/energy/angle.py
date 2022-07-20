# Copyright 2021-2022 The AIMM Group at Shenzhen Bay Laboratory & Peking University
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
"""Angle energy"""

import mindspore as ms
from mindspore import Tensor
from mindspore import Parameter
from mindspore.ops import functional as F

from .energy import EnergyCell
from ...colvar import AtomAngles
from ...function import functions as func
from ...function.units import Units


class AngleEnergy(EnergyCell):
    r"""Energy term of bond angles

    Args:

        index (Tensor):         Tensor of shape (B, a, 3). Data type is int.
                                Atom index of bond angles.

        tk_init (Tensor):       Tensor of shape (1, a). Data type is float.
                                The harmonic force constants for angles.

        teq_init (Tensor):      Tensor of shape (1, a). Data type is float.
                                The equilibrium value of bond angle.

        scale (float):          A constant value to scale the output. Default: 0.5

        use_pbc (bool):         Whether to use periodic boundary condition.

        energy_unit (str):      Energy unit. Default: None

        units (Units):          Units of length and energy. Default: None

    Symbols:

        B:  Batchsize, i.e. number of walkers in simulation

        a:  Number of angles.

        D:  Dimension of the simulation system. Usually is 3.

    """

    def __init__(self,
                 index: Tensor,
                 tk_init: Tensor,
                 teq_init: Tensor,
                 scale: float = 0.5,
                 use_pbc: bool = None,
                 energy_unit: str = 'kj/mol',
                 units: Units = None,
                 ):

        super().__init__(
            label='angle',
            output_dim=1,
            use_pbc=use_pbc,
            energy_unit=energy_unit,
            units=units,
        )

        # (1,a,3)
        index = Tensor(index, ms.int32)
        if index.ndim == 2:
            index = F.expand_dims(index, 0)
        self.index = Parameter(index, name='angle_index', requires_grad=False)

        self.angles = AtomAngles(index, use_pbc=use_pbc)

        self.num_angles = index.shape[-2]

        # (1,a)
        tk_init = Tensor(tk_init, ms.float32)
        if tk_init.shape[-1] != self.num_angles:
            raise ValueError('The last shape of tk_init ('+str(tk_init.shape[-1]) +
                             ') must be equal to num_angles ('+str(self.num_angles)+')!')
        if tk_init.ndim == 1:
            tk_init = F.expand_dims(tk_init, 0)
        if tk_init.ndim > 2:
            raise ValueError('The rank of tk_init cannot be larger than 2!')
        self.angle_force_constant = Parameter(tk_init, name='angle_force_constant')

        teq_init = Tensor(teq_init, ms.float32)
        if teq_init.shape[-1] != self.num_angles:
            raise ValueError('The last shape of teq_init ('+str(teq_init.shape[-1]) +
                             ') must be equal to num_angles ('+str(self.num_angles)+')!')
        if teq_init.ndim == 1:
            teq_init = F.expand_dims(teq_init, 0)
        if teq_init.ndim > 2:
            raise ValueError('The rank of teq_init cannot be larger than 2!')
        self.angle_equil_value = Parameter(teq_init, name='angle_equil_value')

        self.scale = Tensor(scale, ms.float32)

    def set_pbc(self, use_pbc=None):
        self.use_pbc = use_pbc
        self.angles.set_pbc(use_pbc)
        return self

    def construct(self,
                  coordinate: Tensor,
                  neighbour_index: Tensor = None,
                  neighbour_mask: Tensor = None,
                  neighbour_coord: Tensor = None,
                  neighbour_distance: Tensor = None,
                  inv_neigh_dis: Tensor = None,
                  pbc_box: Tensor = None,
                  ):
        r"""Calculate energy term.

        Args:
            coordinate (Tensor):            Tensor of shape (B, A, D). Data type is float.
                                            Position coordinate of atoms in system
            neighbour_index (Tensor):       Tensor of shape (B, A, N). Data type is int.
                                            Index of neighbour atoms.
            neighbour_mask (Tensor):        Tensor of shape (B, A, N). Data type is bool.
                                            Mask for neighbour index.
            neighbour_coord (Tensor):       Tensor of shape (B, A, N). Data type is bool.
                                            Position coorindates of neighbour atoms.
            neighbour_distance (Tensor):    Tensor of shape (B, A, N). Data type is float.
                                            Distance between neighbours atoms.
            inv_neigh_dis (Tensor):         Tensor of shape (B, A, N). Data type is float.
                                            Reciprocal of distances.
            pbc_box (Tensor):               Tensor of shape (B, D). Data type is float.
                                            Tensor of PBC box. Default: None

        Returns:
            energy (Tensor):    Tensor of shape (B, 1). Data type is float.

        Symbols:
            B:  Batchsize, i.e. number of walkers in simulation
            A:  Number of atoms.
            D:  Dimension of the simulation system. Usually is 3.

        """
        # (B,M)
        theta = self.angles(coordinate, pbc_box)
        # (B,M) = (B,M) - (1,M)
        dtheta = theta - self.angle_equil_value
        dtheta2 = dtheta * dtheta

        # E_angle = 1/2 * k_\theta * (\theta-\theta_0)^2
        # (B,M) = (1,M) * (B,M) * k
        energy = self.angle_force_constant * dtheta2 * self.scale

        # (B,1) <- (B,M)
        return func.keepdim_sum(energy, -1)
