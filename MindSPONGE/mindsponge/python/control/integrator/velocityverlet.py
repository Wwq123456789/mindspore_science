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
"""velocity verlet"""
from mindspore import ops
from mindspore.common.api import ms_function
from mindspore.ops import functional as F
from mindspore.nn.optim.optimizer import opt_init_args_register

from .integrator import Integrator

_vv_integrator = ops.MultitypeFuncGraph("velocity_verlet_integrator")


@_vv_integrator.register("Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor")
def _step_begin(acc_scale, dt, gradients, inv_mass, r_cur, v_cur, v_half):
    """Apply sgd optimizer to the weight parameter using Tensor."""
    success = True
    a_cur = -acc_scale * gradients * inv_mass

    v_new_half = v_cur + 0.5 * a_cur * dt
    r_new = r_cur + v_new_half * dt

    success = F.depend(success, F.assign(r_cur, r_new))
    success = F.depend(success, F.assign(v_half, v_new_half))
    return success


@_vv_integrator.register("Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor")
def _step_update(acc_scale, dt, gradients, inv_mass, r_cur, r_last, v_half, v_last):
    """Apply sgd optimizer to the weight parameter using Tensor."""
    success = True
    a_cur = -acc_scale * gradients * inv_mass

    v_cur = v_half + 0.5 * a_cur * dt
    v_new_half = v_cur + a_cur * dt
    r_new = r_cur + v_new_half * dt

    success = F.depend(success, F.assign(r_last, r_cur))
    success = F.depend(success, F.assign(r_cur, r_new))
    success = F.depend(success, F.assign(v_half, v_new_half))
    success = F.depend(success, F.assign(v_last, v_cur))
    return success


@_vv_integrator.register("Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor")
def _step_end(acc_scale, dt, gradients, inv_mass, v_cur, v_half):
    """Apply sgd optimizer to the weight parameter using Tensor."""
    success = True
    a_cur = -acc_scale * gradients * inv_mass

    v_new = v_half + 0.5 * a_cur * dt
    v_new_half = v_cur + a_cur * dt

    success = F.depend(success, F.assign(v_cur, v_new))
    success = F.depend(success, F.assign(v_half, v_new_half))

    return success


class VelocityVerlet(Integrator):
    """velocity verlet"""
    @opt_init_args_register
    def __init__(
            self,
            system,
            time_step=1e-3,
            thermostat=None,
            weight_decay=0.0,
            loss_scale=1.0,
    ):
        super().__init__(
            system=system,
            time_step=time_step,
            thermostat=thermostat,
            weight_decay=weight_decay,
            loss_scale=loss_scale,
        )

        self.thermostat = thermostat

        self.coordiantes_before_one_step = self.coordinates.clone(prefix="before_one_step")

        self.velocities = self.system.velocities
        self.velocities_before_one_step = self.system.velocities_before_one_step
        self.velocities_before_half_step = self.system.velocities_before_half_step

        self.concat_last_dim = ops.Concat(axis=-1)
        self.concat_penulti = ops.Concat(axis=-2)
        self.keep_mean = ops.ReduceMean(keep_dims=True)

    @ms_function
    def first_step(self, gradients):
        """first step"""
        r_update = self.update_coordiantes
        v_cur = self.velocities
        v_half = self.velocities_before_half_step
        gradients = self.decay_weight(gradients)
        gradients = self.scale_grad(gradients)
        inv_mass = self.inv_mass
        dt = self.get_dt()

        kinetic = self.get_system_kinetic(v_cur)
        temperature = self.get_temperature(kinetic)
        success = self.system.update_thermo(kinetic, temperature)

        if self.is_group_lr:
            success = self.map_(F.partial(_vv_integrator, self.acc_unit_scale),
                                dt, gradients, inv_mass, r_update, v_cur, v_half)
        else:
            success = self.map_(F.partial(_vv_integrator, self.acc_unit_scale, dt),
                                gradients, inv_mass, r_update, v_cur, v_half)

        success = F.depend(success, F.assign(self.step, self.step + 1))

        return success

    def construct(self, gradients):
        r_update = self.coordinates
        v_cur = self.velocities
        v_half = self.velocities_before_half_step
        r_cur = self.coordiantes_before_one_step
        gradients = self.decay_weight(gradients)
        gradients = self.scale_grad(gradients)
        inv_mass = self.inv_mass
        dt = self.get_dt()

        kinetic = self.get_system_kinetic(v_cur)
        temperature = self.get_temperature(kinetic)
        success = self.system.update_thermo(kinetic, temperature)

        if self.is_group_lr:
            success = self.map_(F.partial(_vv_integrator, self.acc_unit_scale),
                                dt, gradients, inv_mass, r_update, r_cur, v_half, v_cur)
        else:
            success = self.map_(F.partial(_vv_integrator, self.acc_unit_scale, dt),
                                gradients, inv_mass, r_update, r_cur, v_half, v_cur)

        success = F.depend(success, F.assign(self.step, self.step + 1))

        return success
