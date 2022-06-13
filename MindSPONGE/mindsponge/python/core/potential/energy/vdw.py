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
"""vdw"""
from mindspore.nn import Cell
from mindspore import ops
from mindspore.ops import functional as F

from ....common.functions import gather_values


class VandDerWaalsFunction(Cell):
    """vanderwaals function"""
    def __init__(self, atomic_radius, well_depth):
        super().__init__()

        self.atomic_radius = atomic_radius
        self.well_depth = well_depth
        self.identity = ops.Identity()

    def construct(self, inverse_distances, neighbour_index):
        atomic_radius = self.identity(self.atomic_radius)
        well_depth = self.identity(self.well_depth)

        # (B,A,1)
        ri = F.expand_dims(atomic_radius, -1)
        # (B,A,N)
        rj = gather_values(atomic_radius, neighbour_index)
        # (B,A,N) = (B,A,1) * (B,A,N)
        r0 = ri + rj

        # (B,A,1)
        eps_i = F.expand_dims(well_depth, -1)
        # (B,A,N)
        eps_j = gather_values(well_depth, neighbour_index)
        # (B,A,N) = (B,A,1) * (B,A,N)
        eps = F.sqrt(eps_i * eps_j)

        rij = r0 * inverse_distances

        r6 = F.pows(rij, 6)
        r12 = r6 * r6

        e_acoeff = eps * r12
        e_bcoeff = eps * r6 * 2

        # (B,A,N)
        e_vdw = e_acoeff - e_bcoeff

        return e_vdw
