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
"""simulation cell"""
import mindspore as ms
from mindspore import Tensor
from mindspore import ops
from mindspore.ops import functional as F
from mindspore.nn import Cell

from ..common.functions import norm_last_dim, gather_vectors
from ..common.tools import GetVector
from ..partition.neighbourlist import NeighbourList


class SimulationCell(Cell):
    """simulation cell"""

    def __init__(
            self,
            system,
            potential,
            bias=None,
            neighbour_list=None,
            barostat=None,
    ):
        super().__init__(auto_prefix=False)

        self.system = system
        self.potential = potential
        self.bias = bias

        self.num_walkers = self.system.num_walkers
        self.num_atoms = self.system.num_atoms

        self.neighbour_list = neighbour_list
        if neighbour_list is None:
            self.neighbour_list = NeighbourList(system)

        self.num_neighbours = self.neighbour_list.num_neighbours

        self.cutoff = self.neighbour_list.cutoff
        self.nl_update_steps = self.neighbour_list.update_steps

        self.coordinates = self.system.coordinates
        self.pbc_box = self.system.pbc_box
        self.mass = self.system.mass

        self.pbc = self.system.pbc

        self.potential.set_pbc(self.pbc)

        for p in self.potential.trainable_params():
            p.requires_grad = False

        self.units = self.system.units

        self.potential_units = self.potential.units

        self.input_unit_scale = Tensor(self.units.convert_length_to(self.potential.length_unit()), ms.float32)
        self.output_unit_scale = Tensor(self.units.convert_energy_from(self.potential.energy_unit()), ms.float32)

        self.get_vector = GetVector(self.pbc)

        mask_fill = self.units.length(10, 'nm')
        fill_shape = (self.num_walkers, self.num_atoms, self.num_neighbours)
        self.mask_fill = F.fill(ms.float32, fill_shape, mask_fill)
        self.zeros = F.zeros_like(self.mask_fill)

        self.barostat = barostat

        self.identity = ops.Identity()

    def length_unit(self):
        return self.units.length_unit()

    def energy_unit(self):
        return self.units.energy_unit()

    def set_cutoff(self, cutoff):
        self.neighbour_list.set_cutoff(cutoff)
        return self

    def construct(self, index: Tensor, mask: Tensor = None):
        coordinates, pbc_box = self.system()

        coordinates *= self.input_unit_scale
        if pbc_box is not None:
            pbc_box *= self.input_unit_scale

        # (B,A,1,D) <- (B,A,D)
        atoms = F.expand_dims(coordinates, -2)
        # (B,A,N,D) <- (B,A,D)
        neighbours = gather_vectors(coordinates, index)
        vectors = self.get_vector(atoms, neighbours, pbc_box)

        # Add a non-zero value to the vectors whose mask value is False
        # to prevent them from becoming zero values after Norm operation,
        # which could lead to auto-differentiation errors
        if mask is not None:
            # (B,A,N)
            mask_fill = F.select(mask, self.zeros, self.mask_fill)
            # (B,A,N,D) = (B,A,N,D) + (B,A,N,1)
            vectors += F.expand_dims(mask_fill, -1)

        # (B,A,N) = (B,A,N,D)
        distances = norm_last_dim(vectors)

        if self.cutoff is not None:
            distance_mask = distances < self.cutoff
            if mask is None:
                mask = distance_mask
            else:
                mask = F.logical_and(distance_mask, mask)

        energy = self.potential(coordinates, vectors, distances, index, mask, pbc_box)

        energy *= self.output_unit_scale

        return energy
