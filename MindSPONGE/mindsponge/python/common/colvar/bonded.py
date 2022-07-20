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
"""bonded"""
from mindspore.ops import functional as F
from mindspore import numpy as msnp

from .colvar import Colvar
from ..functions import gather_values, gather_vectors, keep_norm_last_dim


class BondedColvar(Colvar):
    """bonded colvar"""
    def __init__(
            self,
            bond_index,
            system=None,
            unit_length=None,
    ):
        super().__init__(
            dim_output=1,
            periodic=False,
            use_pbc=None,
            system=system,
            unit_length=unit_length,
        )

        self.bond_index = bond_index

    def construct(self, bond_vectors, bond_distances):
        raise NotImplementedError


class BondedDistances(BondedColvar):
    """bonded distance"""
    def __init__(
            self,
            bond_index=None,
            system=None,
            unit_length=None,
    ):
        super().__init__(
            bond_index=bond_index,
            system=system,
            unit_length=unit_length,
        )

    def construct(self, bond_vectors, bond_distances):
        r"""Compute distance between two atoms.

        Args:
            coordinates (ms.Tensor[float]): coordinates of system with shape (B,A,D)

        Returns:
            distances (ms.Tensor[float]): distance between atoms with shape (B,M,1)

        """

        distances = bond_distances
        if self.bond_index is not None:
            distances = gather_values(bond_distances, self.bond_index)

        return distances


class BondedAngles(BondedColvar):
    """bonded angles"""
    def __init__(
            self,
            bond_index,
            system=None,
            unit_length=None,
    ):
        super().__init__(
            bond_index=bond_index,
            system=system,
            unit_length=unit_length,
        )

    def construct(self, bond_vectors, bond_distances):
        r"""Compute angles formed by three atoms.

        Args:
            coordinates (ms.Tensor[float]): coordinates of system with shape (B,N,D)

        Returns:
            angles (ms.Tensor[float]): angles of atoms with shape (B,n,1)

        """

        # (B,a,2,D) <- gather (B,a,2) from (B,b,D)
        vectors = gather_vectors(bond_vectors, self.bond_index)
        # (B,a,2) <- gather (B,a,2) from (B,b)
        distances = gather_values(bond_distances, self.bond_index)

        # (B,a) <- (B,a,D)
        vec1vec2 = F.reduce_sum(vectors[:, :, 0, :] * vectors[:, :, 1, :], -1)
        # (B,a) = (B,a) * (B,a)
        dis1dis2 = distances[:, :, 0] * distances[:, :, 1]
        # (B,a)/(B,a)
        costheta = vec1vec2 / dis1dis2

        # (B,a)
        return F.acos(costheta)


class BondedTorsions(BondedColvar):
    """bonded torsion"""
    def __init__(
            self,
            bond_index,
            system=None,
            unit_length=None,
    ):
        super().__init__(
            bond_index=bond_index,
            system=system,
            unit_length=unit_length,
        )

    def construct(self, bond_vectors, bond_distances):
        r"""Compute torision angles formed by four atoms.

        Args:
            coordinates (ms.Tensor[float]): coordinates of system with shape (B,A,D)

        Returns:
            angles (ms.Tensor[float]): (B,M,1) angles of atoms

        """

        # (B,a,3,D) <- gather (B,a,3) from (B,b,D)
        vectors = gather_vectors(bond_vectors, self.bond_index)

        vec_1 = vectors[:, :, 0, :]
        vec_2 = vectors[:, :, 1, :]
        vec_3 = vectors[:, :, 2, :]

        # (B,d,1) <- (B,M,D)
        v2norm = keep_norm_last_dim(vec_2)
        # (B,d,D) = (B,d,D) / (B,d,1)
        norm_vec2 = vec_2 / v2norm

        # (B,M,D)
        vec_a = msnp.cross(norm_vec2, vec_1)
        vec_b = msnp.cross(vec_3, norm_vec2)
        cross_ab = msnp.cross(vec_a, vec_b)

        # (B,M)
        sin_phi = F.reduce_sum(cross_ab * norm_vec2, -1)
        cos_phi = F.reduce_sum(vec_a * vec_b, -1)

        # (B,M)
        return F.atan2(-sin_phi, cos_phi)
