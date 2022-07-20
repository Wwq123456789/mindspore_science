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
"""brownian"""
from mindspore import ops
from mindspore.ops import functional as F
from mindspore import Parameter
from mindspore.nn.optim.optimizer import opt_init_args_register

from .integrator import Integrator

_brownian_integrator = ops.MultitypeFuncGraph("brownian_integrator")


@_brownian_integrator.register("Function", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor")
def _step_integrate(normal, force_scale, random_scale, step, gradients, inv_sqrt_mass, r_cur):
    """Apply sgd optimizer to the weight parameter using Tensor."""
    success = True

    dr_force = -gradients * force_scale * inv_sqrt_mass * inv_sqrt_mass
    dr_random = normal(r_cur.shape) * random_scale * inv_sqrt_mass

    r_new = r_cur + dr_force + dr_random

    success = F.depend(success, F.assign(r_cur, r_new))
    success = F.depend(success, F.assign(step, step + 1))
    return success


class Brownian(Integrator):
    """brownian"""
    @opt_init_args_register
    def __init__(
            self,
            system,
            time_step=1e-3,
            temp_target=300,
            coupling_time=2,
            weight_decay=0.0,
            loss_scale=1.0,
    ):
        super().__init__(
            system=system,
            time_step=time_step,
            coupling_time=coupling_time,
            weight_decay=weight_decay,
            loss_scale=loss_scale,
        )

        self.temp_sampler.set_temperature(temp_target)
        self.temp_target = self.temp_sampler.temperature.reshape(-1, 1)

        self.inv_sqrt_mass = self.system.inv_sqrt_mass

        # \gamma = 1.0 / \tau_t
        friction_coefficient = 1.0 / self.coupling_time
        self.friction_coefficient = Parameter(friction_coefficient, name='friction_coefficient', requires_grad=False)

        # dt / \gamma
        force_scale = self.time_step / self.friction_coefficient * self.acc_unit_scale
        self.force_scale = Parameter(force_scale, name='force_scale', requires_grad=False)

        # k = \sqrt(2 * k_B * T * dt / \gamma)
        random_scale = F.sqrt(2 * self.boltzmann * self.temp_target * force_scale)
        self.random_scale = Parameter(random_scale.reshape(-1, 1), name='random_scale', requires_grad=False)

        self.normal = ops.StandardNormal()

        self.concat_last_dim = ops.Concat(axis=-1)
        self.concat_penulti = ops.Concat(axis=-2)
        self.keep_mean = ops.ReduceMean(keep_dims=True)

    def construct(self, gradients):
        r_cur = self.coordinates
        gradients = self.decay_weight(gradients)
        gradients = self.scale_grad(gradients)
        inv_sqrt_mass = self.inv_sqrt_mass

        success = self.map_(
            F.partial(_brownian_integrator, self.normal, self.force_scale, self.random_scale, self.step),
            gradients, inv_sqrt_mass, r_cur)

        return success
