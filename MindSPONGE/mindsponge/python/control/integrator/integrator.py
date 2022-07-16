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
"""integrator"""
import mindspore as ms
from mindspore import ops
from mindspore.ops import functional as F
from mindspore.nn import Optimizer
from mindspore.common import Tensor
from mindspore import Parameter
from mindspore import ms_function
from mindspore.nn.optim.optimizer import opt_init_args_register
from ...common.functions import keepdim_sum, concat_penulti
from ...space.system import SystemCell
from ..thermostat import Thermostat

_temperature_coupling = ops.MultitypeFuncGraph("temperature_coupling")


@_temperature_coupling.register("Function", "Tensor", "Tensor")
def _velocity_scling(thermostat, temperature, velocities):
    success = True
    v_new = thermostat(velocities, temperature)
    success = F.depend(success, F.assign(velocities, v_new))
    return success


class Integrator(Optimizer):
    """integrator"""
    @opt_init_args_register
    def __init__(
            self,
            system: SystemCell,
            time_step: float = 1e-3,
            thermostat: Thermostat = None,
            weight_decay: float = 0.0,
            loss_scale: float = 1.0,
    ):
        super().__init__(
            learning_rate=time_step,
            parameters=system.trainable_params(),
            weight_decay=weight_decay,
            loss_scale=loss_scale,
        )

        self.system = system
        self.mass = system.mass
        self.inv_mass = system.inv_mass
        self.units = system.units
        self.system_mass = self.system.system_mass
        self.degrees_of_freedom = self.system.degrees_of_freedom

        self.kinetic_unit_scale = Tensor(self.units.kinetic_ref(), ms.float32)
        self.acc_unit_scale = Tensor(self.units.acceleration_ref(), ms.float32)
        self.vel_unit_scale = Tensor(self.units.velocity_ref(), ms.float32)

        self.boltzmann = self.units.boltzmann()

        self.tot_mass = keepdim_sum(self.mass[0], -1)

        self.num_walkers = self.system.num_walkers

        self.time_step = time_step

        self.coordinates = self.parameters

        self.thermostat = thermostat

        self.step = Parameter(Tensor(0, ms.int32), name='step')

        self.identity = ops.Identity()

    def get_dt(self):
        return self.get_lr()

    def temperature_coupling(self, velocities, temperature):
        success = True
        if self.thermostat is not None:
            success = self.map_(F.partial(_temperature_coupling, self.thermostat, temperature), velocities)
        return success

    @ms_function
    def get_kinetic_energy(self, m, v):
        # (B,A) <- (B,A,D)
        v2 = F.reduce_sum(v * v, -1)
        # (B,A) <- (1,A) * (B,A)
        k = 0.5 * m * v2
        return self.system.get_kinetic_energy(m, v)

    @ms_function
    def get_system_kinetic(self, velocities):
        v = concat_penulti(velocities)
        v = self.identity(v)
        k = self.get_kinetic_energy(self.system_mass, v)
        return k

    @ms_function
    def get_temperature(self, kinetic=None):
        return 2 * kinetic / self.degrees_of_freedom / self.boltzmann

    def construct(self, gradients):
        raise NotImplementedError
