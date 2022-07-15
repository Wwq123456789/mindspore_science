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
"""dihedral energy"""
import mindspore as ms
from mindspore import Tensor
from mindspore.ops import functional as F
from mindspore import Parameter
from mindspore.common.initializer import initializer

from ...common.functions import keepdim_sum
from .energy import EnergyCell
from ...common.colvar import AtomTorsions, BondedTorsions


class DihedralEnergy(EnergyCell):
    """dihedral energy"""
    def __init__(
            self,
            index,
            pk_init,
            pn_init,
            phase_init,
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

        # (1,d,4)
        index = Tensor(index, ms.int32)
        if index.ndim == 2:
            index = F.expand_dims(index, 0)
        self.index = Parameter(index, name='dihedral_index', requires_grad=False)

        # (1,d)
        self.torsions = AtomTorsions(index, use_pbc=pbc, unit_length=self.units)

        # d
        self.num_torsions = index.shape[-2]

        # (1,d)
        if pk_init is None:
            pk_init = initializer('one', [1, self.num_torsions], ms.float32)
        else:
            pk_init = Tensor(pk_init, ms.float32)
            if pk_init.shape[-1] != self.num_torsions:
                raise ValueError('The last shape of pk_init (' + str(pk_init.shape[-1]) +
                                 ') must be equal to num_torsions (' + str(self.num_torsions) + ')!')
            if pk_init.ndim == 1:
                pk_init = F.expand_dims(pk_init, 0)
            if pk_init.ndim > 2:
                raise ValueError('The rank of pk_init cannot be larger than 2!')
        self.dihedral_force_constant = Parameter(pk_init, name='dihedral_force_constant')

        if pn_init is None:
            pn_init = initializer('one', [1, self.num_torsions], ms.int32)
        else:
            pn_init = Tensor(pn_init, ms.int32)
            if pn_init.shape[-1] != self.num_torsions:
                raise ValueError('The last shape of pn_init (' + str(pn_init.shape[-1]) +
                                 ') must be equal to num_torsions (' + str(self.num_torsions) + ')!')
            if pn_init.ndim == 1:
                pn_init = F.expand_dims(pn_init, 0)
            if pn_init.ndim > 2:
                raise ValueError('The rank of pn_init cannot be larger than 2!')
        self.dihedral_periodicity = Parameter(pn_init, name='dihedral_periodicity')

        if phase_init is None:
            phase_init = initializer('zero', [1, self.num_torsions], ms.float32)
        else:
            phase_init = Tensor(phase_init, ms.float32)
            if phase_init.shape[-1] != self.num_torsions:
                raise ValueError('The last shape of phase_init (' + str(phase_init.shape[-1]) +
                                 ') must be equal to num_torsions (' + str(self.num_torsions) + ')!')
            if phase_init.ndim == 1:
                phase_init = F.expand_dims(phase_init, 0)
            if phase_init.ndim > 2:
                raise ValueError('The rank of phase_init cannot be larger than 2!')
        self.dihedral_phase = Parameter(phase_init, name='dihedral_phase')

        self.scale = Parameter(scale, name='scale', requires_grad=False)

    def set_pbc(self, pbc=None):
        self.pbc = pbc
        self.torsions.set_pbc(pbc)
        return self

    def calculate(self, coordinates, pbc_box=None):
        # (B,M)
        phi = self.torsions(coordinates, pbc_box)

        # (B,M) = (1,M) * (B,M)
        nphi = self.dihedral_periodicity * phi
        # (B,M)
        cosphi = F.cos(nphi - self.dihedral_phase) + 1

        # (B,M) = (1,M) + (B,M)
        e_dihedral = self.dihedral_force_constant * cosphi * self.scale

        # (B,1) <- (B,M)
        e_dihedral = keepdim_sum(e_dihedral, -1)

        return e_dihedral


class DihedralEnergyFromBonds(EnergyCell):
    """DihedralEnergyFromBonds"""
    def __init__(
            self,
            num_dihedral,
            pk_init,
            pn_init,
            phase_init,
            bond_index,
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

        # (1,d,3)
        bond_index = Tensor(bond_index, ms.int32)
        if bond_index.ndim == 2:
            bond_index = F.expand_dims(bond_index, 0)
        self.bond_index = Parameter(bond_index, name='bond_index', requires_grad=False)

        # (1,d)
        self.torsions = BondedTorsions(self.bond_index, unit_length=self.units)

        # d
        self.num_dihedral = num_dihedral

        # (1,d)
        if pk_init is None:
            pk_init = initializer('one', [1, self.num_dihedral], ms.float32)
        else:
            pk_init = Tensor(pk_init, ms.float32)
            if pk_init.shape[-1] != self.num_dihedral:
                raise ValueError('The last shape of pk_init (' + str(pk_init.shape[-1]) +
                                 ') must be equal to num_dihedral (' + str(self.num_dihedral) + ')!')
            if pk_init.ndim == 1:
                pk_init = F.expand_dims(pk_init, 0)
            if pk_init.ndim > 2:
                raise ValueError('The rank of pk_init cannot be larger than 2!')
        self.dihedral_force_constant = Parameter(pk_init, name='dihedral_force_constant')

        if pn_init is None:
            pn_init = initializer('one', [1, self.num_dihedral], ms.int32)
        else:
            pn_init = Tensor(pn_init, ms.int32)
            if pn_init.shape[-1] != self.num_dihedral:
                raise ValueError('The last shape of pn_init (' + str(pn_init.shape[-1]) +
                                 ') must be equal to num_dihedral (' + str(self.num_dihedral) + ')!')
            if pn_init.ndim == 1:
                pn_init = F.expand_dims(pn_init, 0)
            if pn_init.ndim > 2:
                raise ValueError('The rank of pn_init cannot be larger than 2!')
        self.dihedral_periodicity = Parameter(pn_init, name='dihedral_periodicity')

        if phase_init is None:
            phase_init = initializer('zero', [1, self.num_dihedral], ms.float32)
        else:
            phase_init = Tensor(phase_init, ms.float32)
            if phase_init.shape[-1] != self.num_dihedral:
                raise ValueError('The last shape of phase_init (' + str(phase_init.shape[-1]) +
                                 ') must be equal to num_dihedral (' + str(self.num_dihedral) + ')!')
            if phase_init.ndim == 1:
                phase_init = F.expand_dims(phase_init, 0)
            if phase_init.ndim > 2:
                raise ValueError('The rank of phase_init cannot be larger than 2!')
        self.dihedral_phase = Parameter(phase_init, name='dihedral_phase')

        self.scale = Parameter(scale, name='scale', requires_grad=False)

    def construct(self, bond_vectors, bond_distances):
        # (B,M)
        phi = self.torsions(bond_vectors, bond_distances)

        # (B,M) = (1,M) * (B,M)
        nphi = self.dihedral_periodicity * phi

        # (B,M)
        cosphi = F.cos(nphi - self.dihedral_phase) + 1

        # (B,M) = (1,M) + (B,M)
        e_dihedral = self.dihedral_force_constant * cosphi * self.scale

        # (B,1) <- (B,M)
        e_dihedral = keepdim_sum(e_dihedral, -1)

        return e_dihedral
