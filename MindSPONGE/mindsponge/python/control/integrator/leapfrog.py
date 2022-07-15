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
"""leapforg"""
from mindspore import ops, Tensor
from mindspore.common.api import ms_function
from mindspore.ops import functional as F
from mindspore.nn.optim.optimizer import opt_init_args_register
from .integrator import Integrator
from .constraint import Lincs

_leapfrog_integrator = ops.MultitypeFuncGraph("leapfrog_integrator")


@_leapfrog_integrator.register("Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor")
def _step_update(acc_scale, dt, gradients, inv_mass, r_cur, v_half, v_cur):
    """Apply sgd optimizer to the weight parameter using Tensor."""
    success = True
    a_cur = -acc_scale * gradients * inv_mass
    v_new_half = v_half + a_cur * dt
    r_new = r_cur + v_new_half * dt
    v_new = (v_half + v_new_half) / 2
    success = F.depend(success, F.assign(r_cur, r_new))
    success = F.depend(success, F.assign(v_half, v_new_half))
    success = F.depend(success, F.assign(v_cur, v_new))
    return success


@_leapfrog_integrator.register("Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Function")
def _step_update_with_constraint(acc_scale, dt, gradients, inv_mass, r_cur, v_half, v_cur, lincs):
    """Apply sgd optimizer to the weight parameter using Tensor."""
    success = True
    a_cur = -acc_scale * gradients * inv_mass
    v_new_half = v_half + a_cur * dt
    r_new = r_cur + v_new_half * dt
    v_new = (v_half + v_new_half) / 2
    success = F.depend(success, F.assign(r_cur, r_new))
    success = F.depend(success, F.assign(v_half, v_new_half))
    success = F.depend(success, F.assign(v_cur, v_new))
    return success


class LeapFrog(Integrator):
    """leapfrog"""
    @opt_init_args_register
    def __init__(
            self,
            system,
            time_step=1e-3,
            thermostat=None,
            weight_decay=0.0,
            loss_scale=1.0,
            constraint=False,
    ):
        super().__init__(
            system=system,
            time_step=time_step,
            thermostat=thermostat,
            weight_decay=weight_decay,
            loss_scale=loss_scale,
        )

        self.thermostat = thermostat
        self.velocities = self.system.velocities
        self.bond_index = self.system.bond_index
        self.velocities_before_half_step = self.system.velocities_before_half_step
        self.concat_last_dim = ops.Concat(axis=-1)
        self.concat_penulti = ops.Concat(axis=-2)
        self.keep_mean = ops.ReduceMean(keep_dims=True)
        self.constraint = constraint
        self.lincs = (Lincs(self.bond_index,
                            self.system.inv_mass[0].reshape((self.system.num_walkers,
                                                             self.inv_mass[0].shape[-3],
                                                             self.inv_mass[0].shape[-2])),
                            self.system.coordinates),)

    @ms_function
    def first_step(self, gradients):
        return self.construct(gradients)

    def construct(self, gradients):
        r_cur = self.coordinates
        v_cur = self.velocities
        v_half = self.velocities_before_half_step
        gradients = self.decay_weight(gradients)
        gradients = self.scale_grad(gradients)
        inv_mass = self.inv_mass
        bond_index = self.bond_index

        kinetic = self.get_system_kinetic(v_half)
        success = self.temperature_coupling(v_half, kinetic)
        temperature = self.get_temperature(kinetic)
        success = self.system.update_thermo(kinetic, temperature)

        dt = self.get_dt()

        if self.is_group_lr and self.constraint:
            success = self.map_(F.partial(_leapfrog_integrator, self.acc_unit_scale),
                                dt, gradients, inv_mass, r_cur, v_half, v_cur, self.lincs)
        elif self.is_group_lr:
            success = self.map_(F.partial(_leapfrog_integrator, self.acc_unit_scale),
                                dt, gradients, inv_mass, r_cur, v_half, v_cur)
        elif self.constraint:
            success = self.map_(F.partial(_leapfrog_integrator, self.acc_unit_scale, dt),
                                gradients, inv_mass, r_cur, v_half, v_cur, self.lincs)
        else:
            success = self.map_(F.partial(_leapfrog_integrator, self.acc_unit_scale, dt),
                                gradients, inv_mass, r_cur, v_half, v_cur)

        success = F.depend(success, F.assign(self.step, self.step + 1))
        return success
