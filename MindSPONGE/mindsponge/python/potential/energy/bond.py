# Copyright 2021 The AIMM Group at Shenzhen Bay Laboratory & Peking University & Huawei Technologies Co., Ltd
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
"""bond energy"""
import mindspore as ms
from mindspore import Tensor
from mindspore import Parameter
from mindspore.ops import functional as F
from mindspore.common.initializer import initializer
from ...common.functions import keepdim_sum
from .energy import EnergyCell
from ...common.colvar import AtomDistances, BondedDistances


class BondEnergy(EnergyCell):
    """bond energy"""
    def __init__(
            self,
            index,
            rk_init,
            req_init,
            scale=0.5,
            pbc=None,
            unit_length=None,
            unit_energy=None,
    ):
        super().__init__(
            pbc=pbc,
            unit_length=unit_length,
            unit_energy=unit_energy,
        )

        # (1,b,2)
        index = Tensor(index, ms.int32)
        if index.ndim == 2:
            index = F.expand_dims(index, 0)
        self.index = Parameter(index, name='bond_index', requires_grad=False)

        # (B,b)
        self.distances = AtomDistances(self.index, use_pbc=pbc, unit_length=self.units)

        # b
        self.num_bonds = index.shape[-2]

        # (1,b)
        if rk_init is None:
            rk_init = initializer('one', [1, self.num_bonds], ms.float32)
        else:
            rk_init = Tensor(rk_init, ms.float32)
            if rk_init.shape[-1] != self.num_bonds:
                raise ValueError('The last shape of rk_init (' + str(rk_init.shape[-1]) +
                                 ') must be equal to num_bonds (' + str(self.num_bonds) + ')!')
            if rk_init.ndim == 1:
                rk_init = F.expand_dims(rk_init, 0)
            if rk_init.ndim > 2:
                raise ValueError('The rank of rk_init cannot be larger than 2!')
        self.bond_force_constant = Parameter(rk_init, name='bond_force_constant')

        if req_init is None:
            req_init = initializer(self.units.length(0.1, 'nm'), [1, self.num_bonds], ms.float32)
        else:
            req_init = Tensor(req_init, ms.float32)
            if req_init.shape[-1] != self.num_bonds:
                raise ValueError('The last shape of req_init (' + str(req_init.shape[-1]) +
                                 ') must be equal to num_bonds (' + str(self.num_bonds) + ')!')
            if req_init.ndim == 1:
                req_init = F.expand_dims(req_init, 0)
            if req_init.ndim > 2:
                raise ValueError('The rank of req_init cannot be larger than 2!')
        self.bond_equil_value = Parameter(req_init, name='bond_equil_value')

        self.scale = Parameter(scale, name='scale', requires_grad=False)

    def set_pbc(self, pbc=None):
        """set pbc"""
        self.pbc = pbc
        self.distances.set_pbc(pbc)
        return self

    def calculate(self, coordinates, pbc_box=None):
        r"""Compute bond energy.

        Args:
            coordinates (ms.Tensor[float]): (B,A,D) coordinates of system
            pbc_box (ms.Tensor[float]): (B,D) box of periodic boundary condition

        Returns:
            E_bonds (ms.Tensor[float]): (B,1)

        """

        # (B,M)
        length = self.distances(coordinates, pbc_box)

        # (B,M) = (B,M) - (1,M)
        dl = length - self.bond_equil_value
        # (B,M)
        dl2 = dl * dl

        # E_bond = 1/2 * k_l * (l-l_0)^2
        # (B,M) = (1,M) * (B,M) * k
        e_bond = self.bond_force_constant * dl2 * self.scale

        # (B,1) <- (B,M)
        return keepdim_sum(e_bond, -1)


class BondEnergyFromBonds(EnergyCell):
    """BondEnergyFromBonds"""
    def __init__(
            self,
            num_bonds,
            rk_init,
            req_init,
            index=None,
            scale=0.5,
            pbc=None,
            unit_length=None,
            unit_energy=None,
    ):
        super().__init__(
            pbc=pbc,
            unit_length=unit_length,
            unit_energy=unit_energy,
        )

        self.index = index
        if index is not None:
            # (1,b)
            index = Tensor(index, ms.int32)
            if index.ndim == 1:
                index = F.expand_dims(index, 0)
            self.index = Parameter(index, name='bond_index', requires_grad=False)

        # (B,b)
        self.distances = BondedDistances(self.index, unit_length=self.units)

        # b
        self.num_bonds = num_bonds

        # (1,b)
        if rk_init is None:
            rk_init = initializer('one', [1, self.num_bonds], ms.float32)
        else:
            rk_init = Tensor(rk_init, ms.float32)
            if rk_init.shape[-1] != self.num_bonds:
                raise ValueError('The last shape of rk_init (' + str(rk_init.shape[-1]) +
                                 ') must be equal to num_bonds (' + str(self.num_bonds) + ')!')
            if rk_init.ndim == 1:
                rk_init = F.expand_dims(rk_init, 0)
            if rk_init.ndim > 2:
                raise ValueError('The rank of rk_init cannot be larger than 2!')
        self.bond_force_constant = Parameter(rk_init, name='bond_force_constant')

        if req_init is None:
            req_init = initializer(self.units.length(0.1, 'nm'), [1, self.num_bonds], ms.float32)
        else:
            req_init = Tensor(req_init, ms.float32)
            if req_init.shape[-1] != self.num_bonds:
                raise ValueError('The last shape of req_init (' + str(req_init.shape[-1]) +
                                 ') must be equal to num_bonds (' + str(self.num_bonds) + ')!')
            if req_init.ndim == 1:
                req_init = F.expand_dims(req_init, 0)
            if req_init.ndim > 2:
                raise ValueError('The rank of req_init cannot be larger than 2!')
        self.bond_equil_value = Parameter(req_init, name='bond_equil_value')

        self.scale = Parameter(scale, name='scale', requires_grad=False)

    def construct(self, bond_vectors, bond_distances):
        r"""Compute bond energy.

        Args:
            coordinates (ms.Tensor[float]): (B,A,D) coordinates of system
            pbc_box (ms.Tensor[float]): (B,D) box of periodic boundary condition

        Returns:
            E_bonds (ms.Tensor[float]): (B,1)

        """

        # (B,M)
        length = self.distances(bond_vectors, bond_distances)
        # (B,M) = (B,M) - (1,M)
        dl = length - self.bond_equil_value
        # (B,M)
        dl2 = dl * dl

        # E_bond = 1/2 * k_l * (l-l_0)^2
        # (B,M) = (1,M) * (B,M) * k
        e_bond = self.bond_force_constant * dl2 * self.scale

        # (B,1) <- (B,M)
        return keepdim_sum(e_bond, -1)
