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
"""non bond"""
import mindspore as ms
from mindspore import Tensor
from mindspore import Parameter
from mindspore.ops import functional as F
from mindspore.common.initializer import initializer

from .energy import EnergyCell
from ..energy import CoulombFunction, ParticleMeshEwald, VandDerWaalsFunction


class NonBondEnergy(EnergyCell):
    """non bond energy"""
    def __init__(
            self,
            num_atoms,
            charge=None,
            atomic_radius=None,
            well_depth=None,
            cutoff=None,
            unit_length=None,
            unit_energy=None,
    ):
        super().__init__(
            pbc=None,
            unit_length=unit_length,
            unit_energy=unit_energy,
        )

        self.num_atoms = num_atoms

        self.cutoff = cutoff

        # (1,A)
        self.charge = charge

        if atomic_radius is None:
            self.atomic_radius = Parameter(initializer('one', (1, self.num_atoms), dtype=ms.float32),
                                           name='atomic_radius')
        else:
            atomic_radius = Tensor(atomic_radius, ms.float32)
            if atomic_radius.shape[-1] != self.num_atoms:
                raise ValueError('the last dimension of atomic_radius' + str(atomic_radius.shape[-1]) +
                                 ' must equal to num_atoms (' + str(self.num_atoms) + ')!')
            if atomic_radius.ndim == 1:
                atomic_radius = F.expand_dims(atomic_radius, 0)
            if atomic_radius.ndim > 2:
                raise ValueError('The rank of atomic_raidus cannot be larger than 2!')
            self.atomic_radius = Parameter(atomic_radius, name='atomic_radius')

        if well_depth is None:
            self.well_depth = Parameter(initializer('one', (1, self.num_atoms), dtype=ms.float32), name='well_depth')
        else:
            well_depth = Tensor(well_depth, ms.float32)
            if well_depth.shape[-1] != self.num_atoms:
                raise ValueError('the last dimension of well_depth' + str(well_depth.shape[-1]) +
                                 ' must equal to num_atoms (' + str(self.num_atoms) + ')!')
            if well_depth.ndim == 1:
                well_depth = F.expand_dims(well_depth, 0)
            if well_depth.ndim > 2:
                raise ValueError('The rank of well_depth cannot be larger than 2!')
            self.well_depth = Parameter(well_depth, name='well_depth')


        self.coulomb_function = CoulombFunction(self.charge)
        self.pme_function = ParticleMeshEwald(self.charge)
        self.vdw_function = VandDerWaalsFunction(self.atomic_radius, self.well_depth)

        self.ele_function = self.electronic_default
        if self.pbc is not None:
            if self.pbc:
                self.ele_function = self.pme_function
            else:
                self.ele_function = self.coulomb_function

    def set_cutoff(self, cutoff):
        self.cutoff = cutoff
        return self

    def electronic_default(self, inverse_distances, neighbour_index, pbc_box=None):
        if pbc_box is None:
            return self.coulomb_function(inverse_distances, neighbour_index, pbc_box)
        else:
            return self.pme_function(inverse_distances, neighbour_index, pbc_box)

    def construct(self, neighbour_distances, neighbour_index, neighbour_mask, pbc_box=None):
        neighbour_distances *= self.input_unit_scale
        if pbc_box is not None:
            pbc_box *= self.input_unit_scale

        # (B,A,N)
        inv_dis = 1 / neighbour_distances
        if neighbour_mask is not None:
            inv_dis = F.select(neighbour_mask, inv_dis, F.zeros_like(inv_dis))

        # (B,A,N)
        e_ele = self.ele_function(inv_dis, neighbour_index, pbc_box)
        # (B)
        e_ele = F.reduce_sum(e_ele, (-1, -2)) / 2
        # (B,1)
        e_ele = F.expand_dims(e_ele, 1)
        e_ele *= self.output_unit_scale

        # (B,A,N)
        e_vdw = self.vdw_function(inv_dis, neighbour_index)
        # (B)
        e_vdw = F.reduce_sum(e_vdw, (-1, -2)) / 2
        # (B,1)
        e_vdw = F.expand_dims(e_vdw, 1)
        e_vdw *= self.output_unit_scale

        return e_ele, e_vdw
