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
"""atoms"""
from mindspore import ops
from mindspore.ops import functional as F
from mindspore import numpy as msnp
from ..functions import gather_vectors, norm_last_dim
from .colvar import Colvar


class AtomDistances(Colvar):
    """atom distances"""
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

        # (1,b,2)
        self.index = index
        self.identity = ops.Identity()

    def construct(self, coordinates, pbc_box=None):
        r"""Compute distance between two atoms.

        Args:
            coordinates (ms.Tensor[float]): coordinates of system with shape (B,A,D)

        Returns:
            distances (ms.Tensor[float]): distance between atoms with shape (B,M,1)

        """

        # (B,b,2,D)
        index = self.identity(self.index)

        atoms = gather_vectors(coordinates, index)

        # (B,b,D)
        vec = self.get_vector(atoms[:, :, 0, :], atoms[:, :, 1, :], pbc_box)
        # (B,b)
        return norm_last_dim(vec)


class AtomAngles(Colvar):
    """atom angles"""
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

        self.index = index
        self.split = ops.Split(-2, 3)

    def construct(self, coordinates, pbc_box=None):
        r"""Compute angles formed by three atoms.

        Args:
            coordinates (ms.Tensor[float]): coordinates of system with shape (B,N,D)

        Returns:
            angles (ms.Tensor[float]): angles of atoms with shape (B,n,1)

        """

        # (B,a,3,D)
        atoms = gather_vectors(coordinates, self.index)

        # (B,a,1,D)
        atom0, atom1, atom2 = self.split(atoms)

        vec1 = self.get_vector(atom1, atom0, pbc_box).squeeze(-2)
        vec2 = self.get_vector(atom1, atom2, pbc_box).squeeze(-2)

        # (B,a) <- (B,a,D)
        dis1 = norm_last_dim(vec1)
        dis2 = norm_last_dim(vec2)

        # (B,a) <- (B,a,D)
        vec1vec2 = F.reduce_sum(vec1 * vec2, -1)
        # (B,a) = (B,a) * (B,a)
        dis1dis2 = dis1 * dis2
        # (B,a)/(B,a)
        costheta = vec1vec2 / dis1dis2

        # (B,a)
        return F.acos(costheta)


class AtomTorsions(Colvar):
    """atom torsion"""
    def __init__(
            self,
            index,
            use_pbc=None,
            unit_length=None,
    ):
        super().__init__(
            dim_output=1,
            periodic=True,
            use_pbc=use_pbc,
            unit_length=unit_length,
        )

        # (1,d,4)
        self.index = index

        self.split = ops.Split(-2, 4)

    def construct(self, coordinates, pbc_box=None):
        r"""Compute torision angles formed by four atoms.

        Args:
            coordinates (ms.Tensor[float]): coordinates of system with shape (B,A,D)

        Returns:
            angles (ms.Tensor[float]): (B,M,1) angles of atoms

        """

        # (B,d,4,D)
        atoms = gather_vectors(coordinates, self.index)

        # (B,d,1,D)
        atom_a, atom_b, atom_c, atom_d = self.split(atoms)

        # (B,d,1,D)
        vec_1 = self.get_vector(atom_b, atom_a, pbc_box).squeeze(-2)
        vec_2 = self.get_vector(atom_c, atom_b, pbc_box).squeeze(-2)
        vec_3 = self.get_vector(atom_d, atom_c, pbc_box).squeeze(-2)

        # (B,d,1) <- (B,M,D)
        # v2norm = keep_norm_last_dim(vec_2)
        v2norm = msnp.norm(vec_2, axis=-1, keepdims=True)

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
