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
"""base"""
import mindspore as ms
from mindspore.common import Tensor
from .colvar import Colvar
from ..functions import keep_norm_last_dim, calc_angle_between_vectors, calc_torsion_for_vectors, norm_last_dim


class Distance(Colvar):
    """distance"""
    def __init__(
            self,
            position0,
            position1,
            use_pbc=None,
            unit_length=None,
    ):
        super().__init__(
            dim_output=1,
            periodic=False,
            use_pbc=use_pbc,
            unit_length=unit_length,
        )

        self.position0 = position0
        self.position1 = position1

    def construct(self, coordinates, pbc_box=None):
        r"""Compute distance between two atoms.

        Args:
            coordinates (ms.Tensor[B,N,D])

        Returns:
            distance (ms.Tensor[B,n,1]):

        """

        pos0 = self.position0(coordinates)
        pos1 = self.position1(coordinates)

        vec = self.get_vector(pos0, pos1, pbc_box)
        return keep_norm_last_dim(vec)


class Angle(Colvar):
    """angle"""
    def __init__(
            self,
            position_a,
            position_b,
            position_c,
            use_pbc=None,
            unit_length=None,
    ):
        super().__init__(
            dim_output=1,
            periodic=False,
            use_pbc=use_pbc,
            unit_length=unit_length,
        )

        self.position_a = position_a
        self.position_b = position_b
        self.position_c = position_c

    def construct(self, coordinates, pbc_box=None):
        r"""Compute distance between two atoms.

        Args:
            coordinates (ms.Tensor[B,N,D])

        Returns:
            distance (ms.Tensor[B,n,1]):

        """

        pos_a = self.position_a(coordinates)
        pos_b = self.position_b(coordinates)
        pos_c = self.position_c(coordinates)

        vec_ba = self.get_vector(pos_b, pos_a, pbc_box)
        vec_bc = self.get_vector(pos_b, pos_c, pbc_box)

        return calc_angle_between_vectors(vec_ba, vec_bc)


class Torsion(Colvar):
    """torsion"""
    def __init__(
            self,
            position_a,
            position_b,
            position_c,
            position_d,
            use_pbc=None,
            unit_length=None,
    ):
        super().__init__(
            dim_output=1,
            periodic=True,
            use_pbc=use_pbc,
            unit_length=unit_length,
        )

        self.position_a = position_a
        self.position_b = position_b
        self.position_c = position_c
        self.position_d = position_d

    def construct(self, coordinates, pbc_box=None):
        r"""Compute distance between two atoms.

        Args:
            coordinates (ms.Tensor[B,N,D])

        Returns:
            distance (ms.Tensor[B,n,1]):

        """

        pos_a = self.position_a(coordinates)
        pos_b = self.position_b(coordinates)
        pos_c = self.position_c(coordinates)
        pos_d = self.position_d(coordinates)

        vec_ba = self.get_vector(pos_b, pos_a, pbc_box)
        vec_cb = self.get_vector(pos_c, pos_b, pbc_box)
        vec_dc = self.get_vector(pos_d, pos_c, pbc_box)

        return calc_torsion_for_vectors(vec_ba, vec_cb, vec_dc)


class AtomDistances(Colvar):
    """atom distance"""
    def __init__(
            self,
            index,
            use_pbc=None,
            unit_length=None,
    ):
        super().__init__(
            dim_output=1,
            periodic=False,
            use_pbc=use_pbc,
            unit_length=unit_length,
        )

        # (M,2)
        index = Tensor(index, ms.int32)

        if index.ndim < 1 or index.shape[-1] != 2:
            raise ValueError('The last dimension of index must be 2')

        # (M)
        self.idx0 = index[..., 0]
        self.idx1 = index[..., 1]

    def construct(self, coordinates, pbc_box=None):
        r"""Compute distance between two atoms.

        Args:
            coordinates (ms.Tensor[float]): coordinates of system with shape (B,A,D)

        Returns:
            distances (ms.Tensor[float]): distance between atoms with shape (B,M,1)

        """

        # (B,M,3)
        pos0 = coordinates[..., self.idx0, :]
        pos1 = coordinates[..., self.idx1, :]

        # (B,M,3)
        vec = self.get_vector(pos0, pos1, pbc_box)
        # (B,M)
        return norm_last_dim(vec)
