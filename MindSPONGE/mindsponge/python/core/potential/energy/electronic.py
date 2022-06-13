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
"""electronic energy"""
from mindspore import Tensor
from mindspore.nn import Cell
from mindspore import ops
from mindspore.ops import functional as F
from mindspore import numpy as msnp

from ....common.functions import gather_values


class CoulombFunction(Cell):
    """coulomb function"""
    def __init__(self, charge):
        super().__init__()

        self.charge = charge
        self.identity = ops.Identity()

    def construct(self, inverse_distances, neighbour_index, pbc_box=None):
        charge = self.identity(self.charge)
        # (B,A,1)
        qi = F.expand_dims(charge, -1)
        # (B,A,N)
        qj = gather_values(charge, neighbour_index)
        # (B,A,N) = (B,A,1) * (B,A,N)
        qiqj = qi * qj

        e_coulomb = qiqj * inverse_distances

        return e_coulomb


class ParticleMeshEwald(Cell):
    """particle mesh ewald"""
    def __init__(self, charge):
        super().__init__()

        self.charge = charge

    def construct(self, inverse_distances, neighbour_index, pbc_box=None):
        return 0


def dsf_coulomb(r: Tensor,
                Q_sq: Tensor,
                alpha: Tensor = 0.25,
                cutoff: float = 8.0) -> Tensor:
    """Damped-shifted-force approximation of the coulombic interaction."""
    qqr2e = 332.06371  # Coulmbic conversion factor: 1/(4*pi*epo).

    cutoffsq = cutoff * cutoff
    erfcc = msnp.erfc(alpha * cutoff)
    erfcd = msnp.exp(-alpha * alpha * cutoffsq)
    f_shift = -(erfcc / cutoffsq + 2 / msnp.sqrt(msnp.pi) * alpha * erfcd / cutoff)
    e_shift = erfcc / cutoff - f_shift * cutoff

    e = qqr2e * Q_sq / r * (msnp.erfc(alpha * r) - r * e_shift - r ** 2 * f_shift)
    return msnp.where(r < cutoff, e, 0.0)
