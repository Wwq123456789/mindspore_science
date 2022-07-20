# Copyright 2021-2022 The AIMM Group at Shenzhen Bay Laboratory & Peking University
#
# Developer: Yi Isaac Yang, Dechin Chen, Jun Zhang, Yijie Xia, Yupeng Huang
#
# Contact: yangyi@szbl.ac.cn
#
# This code is a part of MindSPONGE.
#
# The Cybertron-Code is open-source software based on the AI-framework:
# MindSpore (https://www.mindspore.cn/)
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
"""
Metrics for collective variables
"""

from mindspore.nn import Metric

from ..colvar import Colvar


class CV(Metric):
    """Metric to output collective variables"""
    def __init__(self,
                 colvar: Colvar,
                 indexes: tuple = (2, 3),
                 ):

        super().__init__()
        self._indexes = indexes
        self.colvar = colvar

    def clear(self):
        self._cv_value = 0

    def update(self, *inputs):
        coordinate = inputs[self._indexes[0]]
        pbc_box = inputs[self._indexes[1]]
        self._cv_value = self.colvar(coordinate, pbc_box)

    def eval(self):
        return self._cv_value
