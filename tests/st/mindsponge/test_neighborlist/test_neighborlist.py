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
"""Test neighborlist"""

import sys
import time
import numpy as np
import mindspore.common.dtype as mstype
from mindspore import Tensor
from mindspore import context
from mindsponge import NeighborList

class Space():
    """Space"""
    def __init__(self):
        self.atom_numbers = 100
        self.max_atom_in_grid_numbers = 64    #TO DO
        self.max_neighbor_numbers = 800       #TO DO
        self.cutoff = 10.0
        self.skin = 2.0
        self.box_length = np.array([30, 30, 30]) #if box_length is modified, crd and old_crd need to modify
        half_cutoff = 0.5 * (self.cutoff + self.skin)
        self.nx = int(self.box_length[0] / half_cutoff)
        self.ny = int(self.box_length[1] / half_cutoff)
        self.nz = int(self.box_length[2] / half_cutoff)
        grid_length = [self.box_length[0] / self.nx,
                       self.box_length[1] / self.ny,
                       self.box_length[2] / self.nz]
        self.grid_length_inverse = [1.0 / grid_length[0], 1.0 / grid_length[1], 1.0 / grid_length[2]]
        self.nxy = self.nx * self.ny
        self.grid_numbers = self.nx * self.ny * self.nz
        self.grid_n = [self.nx, self.ny, self.nz]
        self.bucket = [-1] * (self.grid_numbers * self.max_atom_in_grid_numbers)
        self.atom_numbers_in_grid_bucket = [0] * self.grid_numbers
        self.pointer = self.pointer_init()
        self.h_excluded_list_start = [0] * self.atom_numbers
        self.h_excluded_list = 0
        self.h_excluded_numbers = [0] * self.atom_numbers

    def pointer_init(self):
        """pointer init"""
        pointer = []
        temp_grid_serial = [0] * 125
        for i in range(self.grid_numbers):
            nz = int(i / self.nxy)
            ny = int((i - self.nxy * nz) / self.nx)
            nx = i - self.nxy * nz - self.nx * ny
            count = 0
            for l in range(-2, 3):
                for m in range(-2, 3):
                    for n in range(-2, 3):
                        xx = nx + l
                        if xx < 0:
                            xx = xx + self.nx
                        elif xx >= self.nx:
                            xx = xx - self.nx
                        yy = ny + m
                        if yy < 0:
                            yy = yy + self.ny
                        elif yy >= self.ny:
                            yy = yy - self.ny
                        zz = nz + n
                        if zz < 0:
                            zz = zz + self.nz
                        elif zz >= self.nz:
                            zz = zz - self.nz
                        temp_grid_serial[count] = zz * self.nxy + yy * self.nx + xx
                        count += 1
            temp_grid_serial = sorted(temp_grid_serial)
            pointer.extend(temp_grid_serial)
        return pointer

if __name__ == "__main__":
    if '--backend' in sys.argv:
        index = sys.argv.index('--backend')
        sys.argv.pop(index)  # Removes the '--backend'
        backend = sys.argv.pop(index)
    else:
        print("Please input backend, eg: python test_neighborlist.py --backend [GPU|CPU|Ascend] --device_id [num]")
    context.set_context(mode=context.GRAPH_MODE, device_target=backend, device_id=4, save_graphs=False)
    space = Space()
    nl = NeighborList(space)
    start = time.time()
    refresh_count = 0
    refresh_intervate = 2
    nl_atom_numbers = Tensor(np.zeros([space.atom_numbers,], np.int32), mstype.int32)
    nl_atom_serial = Tensor(np.zeros([space.atom_numbers, space.max_neighbor_numbers], np.int32), mstype.int32)
    crd = Tensor(np.random.randint(0, space.box_length[0], size=(space.atom_numbers, 3), dtype='int32'), mstype.float32)
    old_crd = Tensor(np.zeros((space.atom_numbers, 3)), mstype.float32)
    #const update
    for steps in range(0, 4):
        if steps == 0:
            update_type = 0
        else:
            update_type = 1
        nl_atom_numbers, nl_atom_serial, old_crd, crd, refresh_count = nl(nl_atom_numbers,
                                                                          nl_atom_serial,
                                                                          crd,
                                                                          old_crd,
                                                                          update_type,
                                                                          refresh_count,
                                                                          refresh_intervate)
        print("steps:", steps)
        print("refresh_count:", refresh_count)
        print("nl_atom_numbers:", nl_atom_numbers)
        print("nl_atom_serial:", nl_atom_serial)
        print("old_crd:", old_crd)
        print("crd:", crd)
        refresh_count += 1
    end = time.time()
    print("Main time(s):", end - start)
    #update
