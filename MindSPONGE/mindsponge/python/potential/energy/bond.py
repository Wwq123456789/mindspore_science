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
"""Bond energy"""

import mindspore as ms
from mindspore import Tensor
from mindspore import Parameter
from mindspore.ops import functional as F

from .energy import EnergyCell
from ...colvar import AtomDistances
from ...function import functions as func
from ...function.units import Units


class BondEnergy(EnergyCell):
    r"""Energy term of bond length

    Args:

        index (Tensor):         Tensor of shape (B, b, 2). Data type is int.
                                Atom index of bond.

        rk_init (Tensor):       Tensor of shape (1, b). Data type is float.
                                The harmonic force constants for bonds.

        req_init (Tensor):      Tensor of shape (1, b). Data type is float.
                                The equilibrium value of bonds.

        scale (float):          A constant value to scale the output. Default: 0.5

        use_pbc (bool):         Whether to use periodic boundary condition.

        length_unit (str):      Length unit for position coordinates. Default: None

        energy_unit (str):      Energy unit. Default: None

        units (Units):          Units of length and energy. Default: None

    Symbols:

        B:  Batchsize, i.e. number of walkers in simulation

        b:  Number of bonds.

        D:  Dimension of the simulation system. Usually is 3.

    """

    def __init__(self,
                 index: Tensor,
                 rk_init: Tensor,
                 req_init: Tensor,
                 scale: float = 0.5,
                 use_pbc: bool = None,
                 length_unit: str = 'nm',
                 energy_unit: str = 'kj/mol',
                 units: Units = None,
                 ):

        super().__init__(
            label='bond',
            output_dim=1,
            use_pbc=use_pbc,
            length_unit=length_unit,
            energy_unit=energy_unit,
            units=units,
        )

        # (1,b,2)
        index = Tensor(index, ms.int32)
        if index.ndim == 2:
            index = F.expand_dims(index, 0)
        self.index = Parameter(index, name='bond_index', requires_grad=False)

        # (B,b)
        self.distances = AtomDistances(
            self.index, use_pbc=use_pbc, length_unit=self.units)

        # b
        self.num_bonds = index.shape[-2]

        # (B,b)
        rk_init = Tensor(rk_init, ms.float32)
        if rk_init.shape[-1] != self.num_bonds:
            raise ValueError('The last shape of rk_init ('+str(rk_init.shape[-1]) +
                             ') must be equal to num_bonds ('+str(self.num_bonds)+')!')
        if rk_init.ndim == 1:
            rk_init = F.expand_dims(rk_init, 0)
        if rk_init.ndim > 2:
            raise ValueError('The rank of rk_init cannot be larger than 2!')
        self.bond_force_constant = Parameter(rk_init, name='bond_force_constant')

        req_init = Tensor(req_init, ms.float32)
        if req_init.shape[-1] != self.num_bonds:
            raise ValueError('The last shape of req_init ('+str(req_init.shape[-1]) +
                             ') must be equal to num_bonds ('+str(self.num_bonds)+')!')
        if req_init.ndim == 1:
            req_init = F.expand_dims(req_init, 0)
        if req_init.ndim > 2:
            raise ValueError('The rank of req_init cannot be larger than 2!')
        self.bond_equil_value = Parameter(req_init, name='bond_equil_value')

        self.scale = Tensor(scale, ms.float32)

    def set_pbc(self, use_pbc=None):
        self.use_pbc = use_pbc
        self.distances.set_pbc(use_pbc)
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
        length = self.distances(coordinate, pbc_box) * self.input_unit_scale

        # (B,M) = (B,M) - (1,M)
        dl = length - self.bond_equil_value
        # (B,M)
        dl2 = dl * dl

        # E_bond = 1/2 * k_l * (l-l_0)^2
        # (B,M) = (1,M) * (B,M) * k
        energy = self.bond_force_constant * dl2 * self.scale

        # (B,1) <- (B,M)
        return func.keepdim_sum(energy, -1)
