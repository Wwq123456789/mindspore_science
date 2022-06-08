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
"""index"""
import mindspore as ms
from mindspore.ops import functional as F
from mindspore import nn
from mindspore.common import Tensor
from .colvar import Colvar


class IndexColvar(Colvar):
    """index colvar"""
    def __init__(
            self,
            dim_output,
            periodic=False,
            use_pbc=None,
            unit_length=None,
    ):
        super().__init__(
            dim_output=dim_output,
            periodic=periodic,
            use_pbc=use_pbc,
            unit_length=unit_length,
        )

    def construct(self, coordinates: Tensor, index: Tensor, mask: Tensor = None, pbc_box: Tensor = None):
        raise NotImplementedError


class IndexDistances(IndexColvar):
    """index distance"""
    def __init__(
            self,
            use_pbc=None,
            unit_length=None,
            mask_fill=100,
            keep_dims=False,
    ):
        super().__init__(
            dim_output=1,
            periodic=False,
            use_pbc=use_pbc,
            unit_length=unit_length,
        )

        self.norm_last_dim = nn.Norm(-1, keep_dims=keep_dims)
        self.mask_fill = mask_fill

    def construct(self, coordinates: Tensor, index: Tensor, mask: Tensor = None, pbc_box: Tensor = None):
        r"""Compute distances between atoms according to index.

        Args:
            coordinates (ms.Tensor[float]): (B,A,D) coordinates of system
            index (ms.Tensor[int]): (B,A,N) neighbour index of atoms
            mask (ms.Tensor[bool]): (B,A,N) mask of neighbour index
            pbc_box (ms.Tensor[float]): (B,D) box of periodic boundary condition

        Returns:
            distances (ms.Tensor[float]): (B,A,N)

        """

        # (B,A,1,D) <- (B,A,D)
        atoms = F.expand_dims(coordinates, -2)
        # (B,A,N,D) <- (B,A,D)
        neighbours = func.gather_vectors(coordinates, index)
        vectors = self.get_vector(atoms, neighbours, pbc_box)

        # Add a non-zero value to the vectors whose mask value is False
        # to prevent them from becoming zero values after Norm operation,
        # which could lead to auto-differentiation errors
        if mask is not None:
            # (B,A,N)
            mask_fill = F.fill(ms.float32, mask.shape, self.mask_fill)
            mask_fill = F.select(mask, F.zeros_like(mask_fill), mask_fill)
            # (B,A,N,D) = (B,A,N,D) + (B,A,N,1)
            vectors += F.expand_dims(mask_fill, -1)

        # (B,A,N) = (B,A,N,D)
        return self.norm_last_dim(vectors)


class IndexVectors(IndexColvar):
    """index vector"""
    def __init__(
            self,
            use_pbc=None,
            unit_length=None,
    ):
        super().__init__(
            dim_output=1,
            periodic=False,
            use_pbc=use_pbc,
            unit_length=unit_length,
        )

    def construct(self, coordinates: Tensor, index: Tensor, mask: Tensor = None, pbc_box: Tensor = None):
        r"""Compute distances between atoms according to index.

        Args:
            coordinates (ms.Tensor[float]): (B,A,D) coordinates of system
            index (ms.Tensor[int]): (B,A,N) neighbour index of atoms
            mask (ms.Tensor[bool]): (B,A,N) mask of neighbour index
            pbc_box (ms.Tensor[float]): (B,D) box of periodic boundary condition

        Returns:
            distances (ms.Tensor[float]): (B,A,N)

        """

        # (B,A,1,D) <- (B,A,D)
        atoms = F.expand_dims(coordinates, -2)
        # (B,A,N,D) <- (B,A,D)
        neighbours = func.gather_vectors(coordinates, index)

        return self.get_vector(atoms, neighbours, pbc_box)
