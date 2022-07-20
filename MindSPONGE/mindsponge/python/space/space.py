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
"""space"""
import numpy as np
from mindspore import nn
from mindspore import Tensor, Parameter


class Space(nn.Cell):
    """space"""
    def __init__(self, system=None):
        super(Space, self).__init__()
        if system:
            self._init_system(system)

    def _init_system(self, system):
        _type_list = [int, float, str, np.int32, np.float32]
        for key in vars(system):
            if type(vars(system)[key]) is np.ndarray:
                if "coordinates" in key or "velocities" in key:
                    setattr(self, key, Parameter(Tensor(vars(system)[key])))
                else:
                    setattr(self, key, Tensor(vars(system)[key]))
            if type(vars(system)[key]) in _type_list:
                setattr(self, key, vars(system)[key])
