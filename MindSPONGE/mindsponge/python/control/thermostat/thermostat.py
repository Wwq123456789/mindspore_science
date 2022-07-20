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
"""thermostat"""
import mindspore as ms
from mindspore import Tensor
from mindspore import Parameter
from mindspore.nn import Cell
from mindspore.ops import functional as F
from mindspore import ms_function
from ...common.functions import keepdim_sum, concat_last_dim, concat_penulti
from ...space.system import SystemCell


class Thermostat(Cell):
    """Thermostat"""
    def __init__(
            self,
            system: SystemCell,
            temperature: Tensor = 300,
            time_step: float = 1e-3,
            coupling_time: float = 4,
    ):
        super().__init__()

        self.system = system
        self.num_walkers = self.system.num_walkers
        self.num_atoms = self.system.num_atoms
        self.dimension = self.system.dimension
        self.degrees_of_freedom = self.system.degrees_of_freedom

        self.coordinates = self.system.coordinates

        self.mass = concat_last_dim(self.system.mass)
        self.inv_mass = concat_penulti(self.system.inv_mass)
        self.inv_sqrt_mass = concat_penulti(self.system.inv_sqrt_mass)

        tot_mass = keepdim_sum(self.mass, -1)
        self.tot_mass = F.expand_dims(tot_mass, -1)

        natoms = F.select(self.mass > 0, F.ones_like(self.mass), F.zeros_like(self.mass))
        natoms = keepdim_sum(natoms, -1)
        self.natoms = F.expand_dims(natoms, -1)

        self.units = self.system.units
        self.kb = self.units.boltzmann()

        self.temperature = Tensor(temperature, ms.float32).reshape(-1, 1)
        self.target_kinetic = 0.5 * self.degrees_of_freedom * self.kb * self.temperature

        self.time_step = time_step

        coupling_time = Tensor(coupling_time, ms.float32).reshape(-1, 1)
        if coupling_time.shape[0] != self.num_walkers and coupling_time.shape[0] != 1:
            raise ValueError('The first shape of coupling_time must equal to 1 or num_walkers')
        self.coupling_time = Parameter(coupling_time, name='coupling_time', requires_grad=False)

    def get_kinetic_energy(self, m, v):
        """get kinetic energy"""
        return self.system.get_kinetic_energy(m, v)

    @ms_function
    def velocity_scale(self, T0, T, ratio=1):
        """velocity scale"""
        scale = F.sqrt(1 + ratio * (T0 / T - 1))
        return scale

    def get_system_com(self):
        """get system com"""
        mr = self.coordinates * F.expand_dims(self.mass, -1)
        r_com = keepdim_sum(mr, -2) / self.tot_mass
        return r_com

    def get_system_com_vector(self):
        """get system com vector"""
        return self.coordinates - self.get_system_com()

    def remove_com_translation(self, v):
        """remove com translation"""
        # (B,A,D) * (1,A,1)
        p = F.expand_dims(self.mass, -1) * v
        # (B,1,D) <- (B,A,D) * (1,A,1) / (B,1,1)
        dp = keepdim_sum(p, -2) / self.natoms
        p -= dp
        return p * self.inv_mass

    def construct(self, v, kinetic):
        scale = self.velocity_scale(self.target_kinetic, kinetic)
        return v * scale
