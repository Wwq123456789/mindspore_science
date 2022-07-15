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
"""non bond 14"""
import mindspore as ms
from mindspore import Tensor
from mindspore import Parameter
from mindspore.common.initializer import initializer
from mindspore.ops import functional as F

from .energy import EnergyCell
from ...common.colvar import AtomDistances
from ...common.functions import keepdim_sum


class NB14Energy(EnergyCell):
    """nb14 energy"""
    def __init__(
            self,
            index,
            nonbond_energy,
            one_scee=None,
            one_scnb=None,
            pbc=None,
            unit_length=None,
            unit_energy=None,
    ):
        super().__init__(
            pbc=pbc,
            unit_length=unit_length,
            unit_energy=unit_energy,
        )

        self.nonbond_energy = nonbond_energy

        # (1,n,2)
        index = Tensor(index, ms.int32)
        if index.ndim == 2:
            index = F.expand_dims(index, 0)
        self.index = index

        self.num_nb14 = index.shape[-2]

        self.distances = AtomDistances(index, use_pbc=pbc, unit_length=self.units)

        if one_scee is None:
            self.scee_scale_factor = Parameter(initializer('one', [1, self.num_nb14], ms.float32),
                                               name='scee_scale_factor', requires_grad=False)
        else:
            one_scee = Tensor(one_scee, ms.float32)
            if one_scee.shape[-1] != self.num_nb14:
                raise ValueError('The last shape of one_scee (' + str(one_scee.shape[-1]) +
                                 ') must be equal to num_nb14 (' + str(self.num_nb14) + ')!')
            if one_scee.ndim == 1:
                one_scee = F.expand_dims(one_scee, 0)
            if one_scee.ndim > 2:
                raise ValueError('The rank of one_scee cannot be larger than 2!')
            self.scee_scale_factor = Parameter(one_scee, name='scee_scale_factor', requires_grad=False)

        if one_scnb is None:
            self.scnb_scale_factor = Parameter(initializer('one', [1, self.num_nb14], ms.float32),
                                               name='scnb_scale_factor', requires_grad=False)
        else:
            one_scnb = Tensor(one_scnb, ms.float32)
            if one_scnb.shape[-1] != self.num_nb14:
                raise ValueError('The last shape of one_scnb (' + str(one_scnb.shape[-1]) +
                                 ') must be equal to num_nb14 (' + str(self.num_nb14) + ')!')
            if one_scnb.ndim == 1:
                one_scnb = F.expand_dims(one_scnb, 0)
            if one_scnb.ndim > 2:
                raise ValueError('The rank of one_scnb cannot be larger than 2!')
            self.scnb_scale_factor = Parameter(one_scnb, name='scnb_scale_factor', requires_grad=False)

        self.charge = Tensor(self.nonbond_energy.charge)
        self.atomic_radius = Tensor(self.nonbond_energy.atomic_radius)
        self.well_depth = Tensor(self.nonbond_energy.well_depth)

        self.qiqj = self.get_qiqj()
        self.rij = self.get_rij()
        self.eps = self.get_eps()

    def set_pbc(self, pbc=None):
        self.pbc = pbc
        self.distances.set_pbc(pbc)
        return self

    def get_qiqj(self):
        q = self.gather_values(self.charge, self.index)
        # (B,a) = (B,a,2)
        return F.reduce_prod(q, -1)

    def get_rij(self):
        # (1,a,2)
        rij = self.gather_values(self.atomic_radius, self.index)
        # r_ij = r_i + r_j
        # (1,a) = (B,a,2)
        return F.reduce_sum(rij, -1)

    def get_eps(self):
        # (1,a,2)
        eps = self.gather_values(self.well_depth, self.index)
        # \eps_ij = sqrt(\eps_i * \eps_j)
        # (1,a) = (1,a,2)
        return F.sqrt(F.reduce_prod(eps, -1))

    def set_cutoff(self, cutoff):
        self.cutoff = cutoff
        return self

    def calculate(self, coordinates, pbc_box=None):

        distances = self.distances(coordinates, pbc_box)
        # (B,a)
        inv_dis = 1 / distances

        # (B,a) = (1,a) * (B,a) * (1,a)
        e14_ele = self.qiqj * inv_dis * self.scee_scale_factor
        # (B,1) <- (B,a)
        e14_ele = keepdim_sum(e14_ele, -1)

        # (B,a) = (1,a)
        r0 = self.rij * inv_dis

        # (B,a)
        r6 = F.pows(r0, 6)
        r12 = r6 * r6

        # (B,a) = (1,a) * (B,a)
        e_acoeff = self.eps * r12
        e_bcoeff = self.eps * r6 * 2

        # (B,a)
        e14_vdw = e_acoeff - e_bcoeff
        e14_vdw *= self.scnb_scale_factor

        # (B,1) <- (B,a)
        e14_vdw = keepdim_sum(e14_vdw, -1)

        return e14_ele, e14_vdw
