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
"""berendsen"""
from mindspore import ops
from .thermostat import Thermostat


class Berendsen(Thermostat):
    """berendsen"""
    def __init__(
            self,
            system,
            temperature=300,
            time_step=1e-3,
            coupling_time=4,
            scale_min=0.8,
            scale_max=1.25,
    ):
        super().__init__(
            system=system,
            temperature=temperature,
            time_step=time_step,
            coupling_time=coupling_time,
        )

        self.scale_min = scale_min
        self.scale_max = scale_max

        self.ratio = self.time_step / self.coupling_time

    def construct(self, v, kinetic):
        scale = self.velocity_scale(self.target_kinetic, kinetic, self.ratio)
        scale = ops.clip_by_value(scale, self.scale_min, self.scale_max)
        return v * scale
