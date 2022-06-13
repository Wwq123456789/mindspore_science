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
"""energy"""
import mindspore as ms
from mindspore import Tensor
from mindspore.nn import Cell

from ....common.units import Units, global_units
from ....common.functions import gather_values, gather_vectors


class EnergyCell(Cell):
    """energy"""
    def __init__(
            self,
            pbc=None,
            unit_length=None,
            unit_energy=None,
    ):
        super().__init__()

        self.pbc = pbc

        if unit_length is None and unit_energy is None:
            self.units = global_units
        else:
            self.units = Units(unit_length, unit_energy)

        self.input_unit_scale = 1
        self.output_unit_scale = 1

        self.gather_values = gather_values
        self.gather_vectors = gather_vectors

    def set_unit_scale(self, units):
        if units is None:
            self.input_unit_scale = 1
            self.output_unit_scale = 1
        elif isinstance(units, Units):
            self.input_unit_scale = Tensor(self.units.convert_length_from(units), ms.float32)
            self.output_unit_scale = Tensor(self.units.convert_energy_to(units), ms.float32)
        else:
            raise TypeError('Unsupported type: ' + str(type(units)))
        return self

    def set_pbc(self, pbc=None):
        self.pbc = pbc
        return self

    def convert_energy_to(self, unit):
        return self.units.convert_energy_to(unit)

    def length_unit(self):
        return self.units.length_unit()

    def energy_unit(self):
        return self.units.energy_unit()

    def calculate(self, coordinates, pbc_box):
        raise NotImplementedError

    def construct(self, coordinates, pbc_box=None):
        coordinates *= self.input_unit_scale
        if pbc_box is not None:
            pbc_box *= self.input_unit_scale

        energy = self.calculate(coordinates, pbc_box)
        return energy * self.output_unit_scale
