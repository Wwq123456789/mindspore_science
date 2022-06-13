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
"""Common class"""

import mindspore as ms
from mindspore import Tensor, nn, ops
from mindspore.nn import Cell
import numpy as np
from . import functions as func


class GetVector(Cell):
    r"""The class to get vector with or without PBC box

    Args:

        use_pbc (bool): Whether to calculate vector under periodic boundary condition.
                        If this is "None", it will determine whether to calculate the vector under
                        periodic boundary condition based on whether the pbc_box is given.
                        Default: None

    """

    def __init__(self, use_pbc: bool = None):
        super().__init__()

        self.get_vector = self.get_vector_default

        self.use_pbc = use_pbc
        self.set_pbc(use_pbc)

    def get_vector_without_pbc(self, position0, position1, pbc_box=None):
        """get vector without periodic boundary condition"""
        return func.get_vector_without_pbc(position0, position1, pbc_box)

    def get_vector_with_pbc(self, position0, position1, pbc_box):
        """get vector with periodic boundary condition"""
        return func.get_vector_with_pbc(position0, position1, pbc_box)

    def get_vector_default(self, position0, position1, pbc_box=None):
        """get vector"""
        return func.get_vector(position0, position1, pbc_box)

    def set_pbc(self, use_pbc=None):
        """set whether to use periodic boundary condition"""
        self.use_pbc = use_pbc
        if use_pbc is None:
            self.get_vector = self.get_vector_default
        else:
            if use_pbc:
                self.get_vector = self.get_vector_with_pbc
            else:
                self.get_vector = self.get_vector_without_pbc
        return self

    def construct(self, initial: Tensor, terminal: Tensor, pbc_box: Tensor = None):
        r"""Compute vector from initial point to terminal point.

        Args:
            initial (Tensor):   Tensor of shape (B, ..., D). Data type is float.
                                Coordinate of initial point
            terminal (Tensor):  Tensor of shape (B, ..., D). Data type is float.
                                Coordinate of terminal point
            pbc_box (Tensor):   Tensor of shape (B, D). Data type is float.
                                Default: None

        Returns:
            vector (Tensor):    Tensor of shape (B, ..., D). Data type is float.

        Symbols:
            B:  Batchsize, i.e. number of walkers in simulation
            D:  Dimension of the simulation system. Usually is 3.

        """
        return self.get_vector(initial, terminal, pbc_box)


class GetDistance(Cell):
    r"""The class to calculate distance with or without PBC box

    Args:

        use_pbc (bool): Whether to calculate distance under periodic boundary condition.
                        If this is "None", it will determine whether to calculate the distance under
                        periodic boundary condition based on whether the pbc_box is given.
                        Default: None

    """

    def __init__(self, use_pbc=None):
        super().__init__()

        self.get_vector = GetVector(use_pbc)

    def set_pbc(self, use_pbc):
        """set whether to use periodic boundary condition"""
        self.get_vector.set_pbc(use_pbc)
        return self

    def construct(self, initial: Tensor, terminal: Tensor, pbc_box: Tensor = None):
        r"""Compute the distance from initial point to terminal point.

        Args:
            initial (Tensor):   Tensor of shape (B, ..., D). Data type is float.
                                Coordinate of initial point
            terminal (Tensor):  Tensor of shape (B, ..., D). Data type is float.
                                Coordinate of terminal point
            pbc_box (Tensor):   Tensor of shape (B, D). Data type is float.
                                Default: None

        Returns:
            distance (Tensor):  Tensor of shape (B, ..., D). Data type is float.

        Symbols:
            B:  Batchsize, i.e. number of walkers in simulation
            D:  Dimension of the simulation system. Usually is 3.

        """
        vector = self.get_vector(initial, terminal, pbc_box)
        return func.norm_last_dim(vector)


class GetDistanceShift(nn.Cell):
    """Module for calculating B matrix whose dimensions are: K.
    Args:
        crd(Tensor,N*3): The old coordinates of the system.
        bonds(Tensor,K*1): All the bonds need to optimize.
    Input:
        crd(Tensor,N*3): The new coordinates of the system.
    Return:
        A tensor(K*1) about all the bond distance.
    """

    def __init__(self, crd, bonds, use_pbc=None):
        super(GetDistanceShift, self).__init__()
        self.crd = crd
        self.bonds = bonds
        self.norm = nn.Norm(-1)
        mask = np.zeros((crd.shape[0], bonds.shape[-2], crd.shape[-2]))
        np.put_along_axis(mask, bonds.asnumpy()[:, :, None, 0], 1, axis=-1)
        self.mask0 = Tensor(mask, ms.int32)[:, :, :, None]
        mask = np.zeros((crd.shape[0], bonds.shape[-2], crd.shape[-2]))
        np.put_along_axis(mask, bonds.asnumpy()[:, :, None, 1], 1, axis=-1)
        self.mask1 = Tensor(mask, ms.int32)[:, :, :, None]

        self.get_distance = GetDistance(use_pbc=use_pbc)
        if use_pbc is not None:
            self.get_distance.set_pbc(use_pbc)

    def construct(self, crd, pbc_box=None):
        dis1 = self.get_distance((self.mask0 * crd).sum(-2), (self.mask1 * crd).sum(-2), pbc_box=pbc_box)
        dis2 = self.get_distance((self.mask0 * self.crd).sum(-2), (self.mask1 * self.crd).sum(-2), pbc_box=pbc_box)
        return dis1 - dis2


class GetShiftGrad(GetDistanceShift):
    """Module for calculating the differentiation of B matrix whose dimensions are: K*N*D.
    Args:
        crd(Tensor,N*3): The old coordinates of the system.
        bonds(Tensor,K*1): All the bonds need to optimize.
    Input:
        crd(Tensor,N*3): The new coordinates of the system.
    Return:
        A tensor(K*N*3) about all the grad value of bond distance.
    """

    def __init__(self, crd, bonds, use_pbc=None):
        super(GetShiftGrad, self).__init__(crd, bonds)
        self.grad = ops.grad
        shape = (crd.shape[0], bonds.shape[-2], crd.shape[-2], crd.shape[-1])
        self.broadcast = ops.BroadcastTo(shape)
        self.net = GetDistanceShift(self.broadcast(crd[:, None, :, :]), bonds, use_pbc=use_pbc)

    def construct(self, crd, pbc_box=None):
        return self.grad(self.net, grad_position=(0,))(self.broadcast(crd[:, None, :, :]), pbc_box)


def mapping_crd(crd, masks, ids, MAX_ATOMS=14, DIMENSIONs=3):
    """
    Mapping NATOMS*3 crd into NRES*14*3.
    Args:
         crd(ndarray, NATOMS*3): The initial coordinates.
         masks(ndarray, NATOMS): The mask of residues indexes.
         ids(ndarray, NATOMS): The index of atoms in a residue.
    Return:
         map_crd(Tensor, Nres*14*3): The new crd after mapping.
    """
    Nres = np.max(masks) + 1
    map_crd = np.zeros((Nres, MAX_ATOMS, DIMENSIONs))
    for i in range(masks.shape[0]):
        map_crd[masks[i]][ids[i]] = crd[i]
    return map_crd
