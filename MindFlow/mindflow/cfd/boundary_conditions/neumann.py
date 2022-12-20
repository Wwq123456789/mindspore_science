# Copyright 2022 Huawei Technologies Co., Ltd
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
# ==============================================================================
"""neumann boundary condition"""
import mindspore.numpy as mnp
from mindspore import jit_class

from .base import Boundary


@jit_class
class Neumann(Boundary):
    """Neumann boundary condition"""

    def __init__(self, config):
        super(Neumann, self).__init__(config)

    def fill_values_head(self, pri_var, axis, pad_size):
        val = pri_var.copy()[:, :1, :, :]
        return mnp.tile(val, (1, pad_size, 1, 1))

    def fill_values_tail(self, pri_var, axis, pad_size):
        val = pri_var.copy()[:, -1:, :, :]
        return mnp.tile(val, (1, pad_size, 1, 1))
