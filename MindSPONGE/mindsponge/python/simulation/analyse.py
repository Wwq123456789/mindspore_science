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
"""analyse"""
import mindspore as ms
from mindspore import ops
from mindspore.nn import Cell
from mindspore.common import Tensor
from ..partition.neighbourlist import NeighbourList


class AnalyseCell(Cell):
    """analyse cell"""
    def __init__(
            self,
            system,
            potential,
            neighbour_list=None,
            calc_energy=False,
            calc_forces=False,
    ):
        super().__init__(auto_prefix=False)
        self.system = system
        self.potential = potential
        self.pbc_box = self.system.pbc_box

        self.neighbour_list = neighbour_list
        if neighbour_list is None:
            self.neighbour_list = NeighbourList(system)

        self.calc_energy = calc_energy
        self.calc_forces = calc_forces

        self.system_units = self.system.units
        self.potential_units = self.potential.units

        self.units = self.system.units

        self.input_unit_scale = Tensor(self.units.convert_length_to(self.potential.length_unit()), ms.float32)
        self.output_unit_scale = Tensor(self.units.convert_energy_from(self.potential.energy_unit()), ms.float32)

        self.grad = ops.GradOperation()

    def construct(self, coordinates=None, pbc_box=None):
        if coordinates is None:
            coordinates, pbc_box = self.system()

        coordinates *= self.input_unit_scale
        if self.pbc_box is not None:
            pbc_box *= self.input_unit_scale

        energy = None
        if self.calc_energy:
            energy = self.potential(coordinates, pbc_box)

        forces = None
        if self.calc_forces:
            forces = -self.grad(self.potential)(coordinates, pbc_box)

        return energy, forces, coordinates, pbc_box
