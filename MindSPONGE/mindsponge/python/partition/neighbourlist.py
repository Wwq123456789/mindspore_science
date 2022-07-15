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
"""neighbour list"""
import mindspore as ms
from mindspore import Tensor
from mindspore import Parameter
from mindspore import numpy as msnp
from mindspore.common.api import ms_function
from mindspore.common.initializer import initializer
from mindspore.nn import Cell
from mindspore import ops
from mindspore.ops import functional as F

from ..common.functions import get_full_connect_index
from ..space.system import SystemCell


class NeighbourList(Cell):
    """neighbourlist"""
    def __init__(self,
                 system: SystemCell,
                 cutoff: Tensor = None,
                 max_neighbours: int = 64,
                 exclude_index: Tensor = None,
                 cutoff_extend: Tensor = 0.2,
                 update_steps: int = 20
                 ):
        super().__init__()

        self.system = system

        self.num_walkers = self.system.num_walkers
        self.num_atoms = self.system.num_atoms

        self.coordinates = self.system.coordinates
        self.pbc_box = self.system.pbc_box

        self.units = self.system.units

        self.max_neighbours = max_neighbours

        self.cutoff = cutoff
        if cutoff is None:
            self.num_neighbours = self.num_atoms - 1
        else:
            self.num_neighbours = max_neighbours

        self.exclude_index = exclude_index
        self.no_mask = False
        if self.cutoff is None and self.exclude_index is None:
            self.no_mask = True
        # (1,A,n) or (B,A,n)
        if exclude_index is not None:
            exclude_index = Tensor(exclude_index, ms.int32)
            if exclude_index.ndim < 2:
                raise ValueError('The rank of exclude_index cannot be smaller than 2!')

            if exclude_index.shape[-2] != self.num_atoms:
                raise ValueError('The last dimension of exclude_index (' + str(exclude_index.shape[-1]) +
                                 ') must be equal to the num_atoms (' + str(self.num_atoms) + ')!')

            if exclude_index.ndim == 2:
                exclude_index = F.expand_dims(exclude_index, 0)
            if exclude_index.shape[0] != self.num_walkers and exclude_index.shape[0] != 1:
                raise ValueError('The first dimension of exclude_index (' + str(exclude_index.shape[0]) +
                                 ') must be equal to 1 or num_walkers (' + str(self.num_walkers) + ')!')

            if exclude_index.ndim > 3:
                raise ValueError('The rank of exclude_index cannot be larger than 3!')

            self.exclude_index = exclude_index

        self.shape = (self.num_walkers, self.num_atoms, self.num_neighbours)

        self.neighbour_index = Parameter(initializer('zero', self.shape, ms.int32), name='neighbour_index',
                                         requires_grad=False)

        if self.cutoff is None and self.exclude_index is None:
            self.neighbour_mask = None
            self.neighbour_shift = None
        else:
            self.neighbour_mask = Parameter(initializer('zero', self.shape, ms.bool_), name='neighbour_mask',
                                            requires_grad=False)
            self.neighbour_shift = Parameter(initializer('zero', self.shape + (1,), ms.float32), name='neighbour_shift',
                                             requires_grad=False)

        self.cutoff_extend = cutoff_extend
        if self.cutoff is not None and self.cutoff_extend <= 0:
            raise ValueError('cutoff_extend must be larger than 0!')

        if update_steps <= 0:
            raise ValueError('update_steps must be larger than 0!')
        self.update_steps = update_steps

        self.identity = ops.Identity()

        self.update()

    def set_cutoff(self, cutoff):
        self.cutoff = cutoff
        if cutoff is None:
            self.num_neighbours = self.num_atoms - 1
        else:
            self.num_neighbours = self.max_neighbours
            if self.cutoff_extend <= 0:
                raise ValueError('cutoff_extend must be larger than 0!')
        return self

    def set_cutoff_extend(self, extend):
        self.cutoff_extend = extend
        if self.cutoff is not None and self.cutoff_extend <= 0:
            raise ValueError('cutoff_extend must be larger than 0!')
        return self

    def set_update_steps(self, steps):
        if steps <= 0:
            raise ValueError('update_steps must be larger than 0!')
        self.update_steps = steps
        return steps

    def full_connect_neighbours(self, exclude_index=None):
        index, mask = get_full_connect_index(self.num_atoms, exclude_index)
        if self.num_walkers > 1:
            index = msnp.broadcast_to(index, self.shape)
            if mask is not None:
                mask = msnp.broadcast_to(mask, self.shape)
        return index, mask

    def _update(self):
        index = None
        mask = None
        if self.cutoff is None:
            index, mask = self.full_connect_neighbours(self.exclude_index)
        else:
            index, mask = self.full_connect_neighbours(self.exclude_index)  # TODO

        return index, mask

    @ms_function
    def update(self):
        index, mask = self._update()

        success = True
        success = F.depend(success, F.assign(self.neighbour_index, index))
        if self.neighbour_mask is not None:
            success = F.depend(success, F.assign(self.neighbour_mask, mask))

        return success

    def construct(self):
        r"""Gather coordinates of neighbours atoms.

        Args:
            coordinates (ms.Tensor[float]): (B,A,D)
            neighbours (ms.Tensor[int]): (B,...)

        Returns:
            neighbour_atoms (ms.Tensor[float]): (B,...,D)

        """

        index = self.identity(self.neighbour_index)
        mask = self.neighbour_mask
        if self.neighbour_mask is not None:
            mask = self.identity(self.neighbour_mask)

        return index, mask
