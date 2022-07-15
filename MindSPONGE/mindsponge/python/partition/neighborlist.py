# Copyright 2021 Huawei Technologies Co., Ltd
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
"""neighborlist"""

import numpy as np
import mindspore.numpy as msnp
import mindspore.common.dtype as mstype
import mindspore.ops as ops
from mindspore import nn
from mindspore import Tensor
from ..ops.neighborlistop import NeighborListOP
from ..potential.utils import get_periodic_displacement

class NeighborList(nn.Cell):
    """NeighborList"""
    def __init__(self, space):
        """
        Args:
            atom_number: the total number of atoms.
            grid_number: the total number of grids divided.
            cutoff_square: the suqare value of cutoff.
            half_skin_square: the suqare value of skin.
            max_atom_in_grid_numbers: the maximum number of atoms in one grid. Default: 64.
            max_neighbor_numbers: The maximum number of neighbors. Default: 800.
            box_length: the length of 3 dimensions of the simulation box.
            grid_length_inverse: the inverse value of grid length.
            grid_n: the number of grids divided of 3 dimensions of the simulation box.
            nxy: the total number of grids divided in xy plane.
            coordinates: the coordinates of each atom.
            bucket: the atom indices in each grid bucket.
            atom_numbers_in_grid_bucket: the number of atoms in each grid bucket.
            pointer: the 125 nearest neighbor grids (including self) of each grid.
            nl_atom_numbers: the number of atoms in neighbor list of each atom.
            nl_atom_serial: the indices of atoms in neighbor list of each atom.
        Returns:
            nl_atom_numbers: the number of atoms in neighbor list of each atom.
            nl_atom_serial: the indices of atoms in neighbor list of each atom.
            old_crd: the coordinates before update of each atom.
            crd: the coordinates of each atom.
        """
        super(NeighborList, self).__init__()
        self.put_atom_into_bucket_op, self.find_atom_neighbors_op, self.delete_excluded_atoms_op = \
            NeighborListOP().register(space.atom_numbers, space.grid_numbers, space.max_atom_in_grid_numbers, \
            space.max_neighbor_numbers)
        self.atom_numbers = space.atom_numbers
        self.grid_numbers = space.grid_numbers
        self.max_atom_in_grid_numbers = space.max_atom_in_grid_numbers
        self.cutoff_square = space.cutoff * space.cutoff
        self.half_skin_square = 0.25 * space.skin * space.skin
        self.box_length = Tensor(np.array(space.box_length[:3]), mstype.float32)
        self.grid_length_inverse = Tensor(space.grid_length_inverse, mstype.float32)
        self.grid_n = Tensor(space.grid_n, mstype.int32)
        self.nxy = space.nxy
        self.grid_coeff = Tensor(np.array([1, space.grid_n[0], space.nxy]).reshape(3, 1), mstype.float32)
        self.bucket = Tensor(np.asarray(space.bucket, np.int32).reshape( \
           [self.grid_numbers, self.max_atom_in_grid_numbers]), mstype.int32)
        self.atom_numbers_in_grid_bucket = Tensor(space.atom_numbers_in_grid_bucket, mstype.int32)
        self.pointer = Tensor(np.asarray(space.pointer, np.int32).reshape([self.grid_numbers, 125]), mstype.int32)
        self.excluded_list_start = Tensor(np.asarray(space.h_excluded_list_start, np.int32), mstype.int32)
        self.excluded_list = Tensor(np.asarray(space.h_excluded_list, np.int32), mstype.int32)
        self.excluded_numbers = Tensor(np.asarray(space.h_excluded_numbers, np.int32), mstype.int32)
        self.zeros = Tensor(np.zeros((space.atom_numbers, 3)), mstype.int32)

    def crd_periodic_map(self, crd, box_len, zeros):
        less_than_zeros = msnp.less(crd, zeros, mstype.int32)
        greater_than_box_len = msnp.negative(msnp.greater(crd, box_len, mstype.int32))
        crd = crd + box_len * greater_than_box_len + box_len * less_than_zeros
        return crd

    def set_atom_grid_serial(self, crd, grid_length_inverse, gridn, grid_coeff, zeros, atom_numbers):
        cast = ops.Cast()
        nxyz = crd * grid_length_inverse
        nxyz = cast(nxyz, mstype.int32)
        nxyz = nxyz * msnp.less((nxyz - gridn), zeros, mstype.int32)
        nxyz_f = cast(nxyz, mstype.float32)
        atom_in_grid_serial = msnp.dot(nxyz_f, grid_coeff)
        atom_in_grid_serial = cast(atom_in_grid_serial, mstype.int32)
        atom_in_grid_serial = msnp.reshape(atom_in_grid_serial, (atom_numbers,))
        return atom_in_grid_serial

    def is_need_refresh_neighbor_list(self, crd, old_crd, box_len, half_skin_square):
        refresh_flag = 0
        dr = get_periodic_displacement(crd, old_crd, box_len)
        dr2 = msnp.sum(dr * dr, axis=1)
        z = msnp.sum(dr2 > half_skin_square)
        if z > 0:
            refresh_flag = 1
        return refresh_flag

    def neighbor_list(self, nl_atom_numbers, nl_atom_serial, crd, old_crd):
        """neighbor_list"""
        crd = self.crd_periodic_map(crd, self.box_length, self.zeros)
        old_crd = msnp.copy(crd)
        atom_in_grid_serial = self.set_atom_grid_serial(crd, self.grid_length_inverse, self.grid_n,
                                                        self.grid_coeff, self.zeros, self.atom_numbers)
        bucket, atom_numbers_in_grid_bucket = self.put_atom_into_bucket_op(atom_in_grid_serial,
                                                                           self.bucket,
                                                                           self.atom_numbers_in_grid_bucket)
        nl_atom_numbers, nl_atom_serial = self.find_atom_neighbors_op(self.cutoff_square,
                                                                      atom_in_grid_serial,
                                                                      crd,
                                                                      self.box_length,
                                                                      self.pointer,
                                                                      bucket,
                                                                      atom_numbers_in_grid_bucket,
                                                                      nl_atom_numbers,
                                                                      nl_atom_serial)
        nl_atom_numbers, nl_atom_serial = self.delete_excluded_atoms_op(self.excluded_list_start,
                                                                        self.excluded_list,
                                                                        self.excluded_numbers,
                                                                        nl_atom_numbers,
                                                                        nl_atom_serial)
        return nl_atom_numbers, nl_atom_serial, crd, old_crd

    def update(self, nl_atom_numbers, nl_atom_serial, crd, old_crd):
        refresh_flag = self.is_need_refresh_neighbor_list(crd, old_crd, self.box_length, self.half_skin_square)
        if refresh_flag == 1:
            nl_atom_numbers, nl_atom_serial, crd, old_crd = self.neighbor_list(nl_atom_numbers,
                                                                               nl_atom_serial,
                                                                               crd,
                                                                               old_crd)
        return nl_atom_numbers, nl_atom_serial, old_crd, crd

    def constant_update(self, nl_atom_numbers, nl_atom_serial, crd, old_crd, refresh_count, refresh_interval):
        if refresh_count % refresh_interval == 0:
            nl_atom_numbers, nl_atom_serial, crd, old_crd = self.neighbor_list(nl_atom_numbers,
                                                                               nl_atom_serial,
                                                                               crd,
                                                                               old_crd)
        refresh_count += 1
        return nl_atom_numbers, nl_atom_serial, old_crd, crd, refresh_count

    def construct(self, nl_atom_numbers, nl_atom_serial, crd, old_crd, update_type, refresh_count, refresh_interval):
        """construct"""
        if update_type == 0:
            nl_atom_numbers, nl_atom_serial, crd, old_crd = self.neighbor_list(nl_atom_numbers,
                                                                               nl_atom_serial,
                                                                               crd,
                                                                               old_crd)
        elif update_type == 1:
            nl_atom_numbers, nl_atom_serial, crd, old_crd, refresh_count = self.constant_update(nl_atom_numbers,
                                                                                                nl_atom_serial,
                                                                                                crd,
                                                                                                old_crd,
                                                                                                refresh_count,
                                                                                                refresh_interval)
        else:
            nl_atom_numbers, nl_atom_serial, crd, old_crd = self.update(nl_atom_numbers, nl_atom_serial, crd, old_crd)
        return nl_atom_numbers, nl_atom_serial, old_crd, crd, refresh_count
