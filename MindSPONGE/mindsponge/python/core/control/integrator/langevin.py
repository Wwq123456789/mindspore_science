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
"""langevin"""
import mindspore as ms
from mindspore import Tensor
from mindspore import ops
from mindspore.ops import functional as F
from mindspore import Parameter
from mindspore.nn.optim.optimizer import opt_init_args_register

from .integrator import Integrator

_langevin_integrator = ops.MultitypeFuncGraph("langevin_integrator")


@_langevin_integrator.register("Function", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor",
                               "Tensor", "Tensor")
def _step_integrate(normal, acc_scale, friction, random_scale, dt, gradients, inv_sqrt_mass, r_cur, v_cur, v_half):
    """Apply sgd optimizer to the weight parameter using Tensor."""
    success = True
    a_cur = -acc_scale * gradients * inv_sqrt_mass * inv_sqrt_mass

    v_new = v_half + a_cur * dt
    r_new_half = r_cur + 0.5 * v_new * dt
    v_random = -friction * v_new + random_scale * inv_sqrt_mass * normal(v_cur.shape)
    v_new_half = v_new + v_random
    r_new = r_new_half + 0.5 * v_new_half * dt

    success = F.depend(success, F.assign(r_cur, r_new))
    success = F.depend(success, F.assign(v_cur, v_new))
    success = F.depend(success, F.assign(v_half, v_new_half))

    return success


class Langevin(Integrator):
    """langevin"""
    @opt_init_args_register
    def __init__(
            self,
            system,
            time_step=1e-3,
            target_temp=300,
            coupling_time=2,
            weight_decay=0.0,
            loss_scale=1.0,
            seed=0,
            seed2=0,
    ):
        super().__init__(
            system=system,
            time_step=time_step,
            thermostat=None,
            weight_decay=weight_decay,
            loss_scale=loss_scale,
        )

        self.inv_sqrt_mass = self.system.inv_sqrt_mass

        self.velocities = self.system.velocities
        self.velocities_before_half_step = self.system.velocities_before_half_step
        self.num_walkers = self.system.num_walkers

        target_temp = Tensor(target_temp, ms.float32).reshape(-1, 1, 1)
        if target_temp.shape[0] != self.num_walkers and target_temp.shape[0] != 1:
            raise ValueError('The size of temperature must be equal to 1 or num_walkers!')
        self.target_temp = target_temp

        coupling_time = Tensor(coupling_time, ms.float32).reshape(-1, 1)
        if coupling_time.shape[0] != self.num_walkers and coupling_time.shape[0] != 1:
            raise ValueError('The first shape of coupling_time must equal to 1 or num_walkers')
        self.coupling_time = Parameter(coupling_time, name='coupling_time', requires_grad=False)

        # \gamma = 1.0 / \tau_t
        effective_friction_rate = 1.0 / self.coupling_time
        self.effective_friction_rate = Parameter(effective_friction_rate, name='effective_friction_rate',
                                                 requires_grad=False)

        # \f = 1 - exp(-\gamma * dt)
        friction = 1.0 - F.exp(-self.effective_friction_rate * self.time_step)
        self.friction = Parameter(friction.reshape(-1, 1), name='friction_fraction', requires_grad=False)

        # k = \sqrt(f * (2 - f) * k_B * T)
        scale = self.friction * (2 - self.friction)
        random_scale = F.sqrt(scale * self.boltzmann * self.target_temp) * self.vel_unit_scale
        self.random_scale = Parameter(random_scale.reshape(-1, 1), name='random_scale', requires_grad=False)

        self.normal = ops.StandardNormal(seed, seed2)

        self.concat_last_dim = ops.Concat(axis=-1)
        self.concat_penulti = ops.Concat(axis=-2)
        self.keep_mean = ops.ReduceMean(keep_dims=True)

    def construct(self, gradients):
        r_cur = self.coordinates
        v_cur = self.velocities
        v_half = self.velocities_before_half_step
        gradients = self.decay_weight(gradients)
        gradients = self.scale_grad(gradients)
        inv_sqrt_mass = self.inv_sqrt_mass

        kinetic = self.get_system_kinetic(v_cur)
        temperature = self.get_temperature(kinetic)
        success = self.system.update_thermo(kinetic, temperature)

        dt = self.get_dt()
        if self.is_group_lr:
            success = self.map_(
                F.partial(_langevin_integrator, self.normal, self.acc_unit_scale, self.friction, self.random_scale),
                dt, gradients, inv_sqrt_mass, r_cur, v_cur, v_half)
        else:
            success = self.map_(
                F.partial(_langevin_integrator, self.normal, self.acc_unit_scale, self.friction, self.random_scale, dt),
                gradients, inv_sqrt_mass, r_cur, v_cur, v_half)

        success = F.depend(success, F.assign(self.step, self.step + 1))

        return success
