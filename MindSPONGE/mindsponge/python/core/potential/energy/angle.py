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
"""angle energy"""
import mindspore as ms
from mindspore import Tensor
from mindspore import Parameter
from mindspore.ops import functional as F
from mindspore.common.initializer import initializer
from ....common.functions import keepdim_sum
from .energy import EnergyCell
from ....common.colvar import AtomAngles, BondedAngles


class AngleEnergy(EnergyCell):
    """angle energy"""
    def __init__(
            self,
            index,
            tk_init,
            teq_init,
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

        # (1,a,3)
        index = Tensor(index, ms.int32)
        if index.ndim == 2:
            index = F.expand_dims(index, 0)
        self.index = Parameter(index, name='angle_index', requires_grad=False)

        self.angles = AtomAngles(index, use_pbc=pbc, unit_length=self.units)

        self.num_angles = index.shape[-2]

        # (1,a)
        if tk_init is None:
            tk_init = initializer('one', [1, self.num_angles], ms.float32)
        else:
            tk_init = Tensor(tk_init, ms.float32)
            if tk_init.shape[-1] != self.num_angles:
                raise ValueError('The last shape of tk_init (' + str(tk_init.shape[-1]) +
                                 ') must be equal to num_angles (' + str(self.num_angles) + ')!')
            if tk_init.ndim == 1:
                tk_init = F.expand_dims(tk_init, 0)
            if tk_init.ndim > 2:
                raise ValueError('The rank of tk_init cannot be larger than 2!')
        self.angle_force_constant = Parameter(tk_init, name='angle_force_constant')

        if teq_init is None:
            teq_init = initializer('zeros', [1, self.num_angles], ms.float32)
        else:
            teq_init = Tensor(teq_init, ms.float32)
            if teq_init.shape[-1] != self.num_angles:
                raise ValueError('The last shape of teq_init (' + str(teq_init.shape[-1]) +
                                 ') must be equal to num_angles (' + str(self.num_angles) + ')!')
            if teq_init.ndim == 1:
                teq_init = F.expand_dims(teq_init, 0)
            if teq_init.ndim > 2:
                raise ValueError('The rank of teq_init cannot be larger than 2!')
        self.angle_equil_value = Parameter(teq_init, name='angle_equil_value')

        self.scale = Parameter(scale, name='scale', requires_grad=False)

    def set_pbc(self, pbc=None):
        """set pbc"""
        self.pbc = pbc
        self.angles.set_pbc(pbc)
        return self

    def calculate(self, coordinates, pbc_box=None):
        """
        calculate
        E_angle = 1/2 * k_\theta * (\theta-\theta_0)^2
        """
        # (B,M)
        theta = self.angles(coordinates, pbc_box)
        # (B,M) = (B,M) - (1,M)
        dtheta = theta - self.angle_equil_value
        dtheta2 = dtheta * dtheta
        # (B,M) = (1,M) * (B,M) * k
        e_angle = self.angle_force_constant * dtheta2 * self.scale
        # (B,1) <- (B,M)
        return keepdim_sum(e_angle, -1)


class AngleEnergyFromBonds(EnergyCell):
    """AngleEnergyFromBonds"""
    def __init__(
            self,
            num_angles,
            tk_init,
            teq_init,
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

        # (1,a,2)
        bond_index = Tensor(bond_index, ms.int32)
        if bond_index.ndim == 2:
            bond_index = F.expand_dims(bond_index, 0)
        self.bond_index = Parameter(bond_index, name='bond_index', requires_grad=False)

        self.angles = BondedAngles(bond_index, unit_length=self.units)

        self.num_angles = num_angles

        # (1,a)
        if tk_init is None:
            tk_init = initializer('one', [1, self.num_angles], ms.float32)
        else:
            tk_init = Tensor(tk_init, ms.float32)
            if tk_init.shape[-1] != self.num_angles:
                raise ValueError('The last shape of tk_init (' + str(tk_init.shape[-1]) +
                                 ') must be equal to num_angles (' + str(self.num_angles) + ')!')
            if tk_init.ndim == 1:
                tk_init = F.expand_dims(tk_init, 0)
            if tk_init.ndim > 2:
                raise ValueError('The rank of tk_init cannot be larger than 2!')
        self.angle_force_constant = Parameter(tk_init, name='angle_force_constant')

        if teq_init is None:
            teq_init = initializer('zeros', [1, self.num_angles], ms.float32)
        else:
            teq_init = Tensor(teq_init, ms.float32)
            if teq_init.shape[-1] != self.num_angles:
                raise ValueError('The last shape of teq_init (' + str(teq_init.shape[-1]) +
                                 ') must be equal to num_angles (' + str(self.num_angles) + ')!')
            if teq_init.ndim == 1:
                teq_init = F.expand_dims(teq_init, 0)
            if teq_init.ndim > 2:
                raise ValueError('The rank of teq_init cannot be larger than 2!')
        self.angle_equil_value = Parameter(teq_init, name='angle_equil_value')

        self.scale = Parameter(scale, name='scale', requires_grad=False)

    def construct(self, bond_vectors, bond_distances):

        # (B,M)
        theta = self.angles(bond_vectors, bond_distances)
        # (B,M) = (B,M) - (1,M)
        dtheta = theta - self.angle_equil_value
        dtheta2 = dtheta * dtheta

        # E_angle = 1/2 * k_\theta * (\theta-\theta_0)^2
        # (B,M) = (1,M) * (B,M) * k
        e_angle = self.angle_force_constant * dtheta2 * self.scale

        # (B,1) <- (B,M)
        return keepdim_sum(e_angle, -1)
