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
"""temperature"""
import mindspore as ms
from mindspore import ops
from mindspore.ops import functional as F
from mindspore.nn import Cell
from mindspore.common import Tensor
from ....common.functions import keepdim_sum, keepdim_mean
from ....common.units import Units, global_units


class VelocitiesGenerator(Cell):
    """VelocitiesGenerator"""
    def __init__(
            self,
            temperature=300,
            remove_translation=True,
            seed=0,
            seed2=0,
            unit_length=None,
            unit_energy=None,
    ):
        super().__init__()

        if unit_length is None and unit_energy is None:
            self.units = global_units
        else:
            self.units = Units(unit_length, unit_energy)

        self.temperature = Tensor(temperature, ms.float32).reshape(-1, 1, 1)

        self.standard_normal = ops.StandardNormal(seed, seed2)

        self.kb = Tensor(self.units.boltzmann(), ms.float32)
        self.kbT = self.kb * self.temperature
        self.sigma = F.sqrt(self.kbT)

        self.velocity_unit_scale = Tensor(self.units.velocity_ref(), ms.float32)

        self.remove_translation = remove_translation

    def set_temperature(self, temperature):
        """set temperature"""
        self.temperature = Tensor(temperature, ms.float32).reshape(-1, 1, 1)
        self.multi_temp = False
        if self.temperature is not None and self.temperature.size > 1:
            self.multi_temp = False
        return self

    def construct(self, shape, mass=1, mask=None):
        v = self.standard_normal(shape) * self.sigma
        # (1,A) or (B,A) -> (1,A,1) or (B,A,1)
        m = F.expand_dims(mass, -1)
        inv_mass = 1.0 / m
        if mask is not None:
            mask = F.expand_dims(mask, -1)
            inv_mass = F.select(mask, inv_mass, F.zeros_like(mass))
        inv_sqrt_mass = F.sqrt(inv_mass)
        v *= inv_sqrt_mass * self.velocity_unit_scale

        if self.remove_translation:
            # (B,A,D) * (1,A,1)
            p = m * v
            # (1,1,1) or (B,1,1) <- (1,A,1) or (B,A,1)

            dp = keepdim_mean(p, -2)
            if mask is not None:
                sp = keepdim_sum(p, -2)
                n = keepdim_sum(F.cast(mask, ms.int32), -2)
                dp = sp / n
            # (B,A,D) - (B,1,D) = (B,A,D)
            p -= dp
            v = p * inv_mass

        return v
