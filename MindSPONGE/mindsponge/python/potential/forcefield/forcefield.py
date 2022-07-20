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
"""force field"""
from mindspore.nn import Cell

from ...common.units import Units, global_units
from ...common.functions import gather_vectors
from ...common.tools import GetVector


class ForceField(Cell):
    """force field"""
    def __init__(
            self,
            unit_length=None,
            unit_energy=None,
            pbc=None,
    ):
        super().__init__()

        if unit_length is None and unit_energy is None:
            self.units = global_units
        else:
            self.units = Units(unit_length, unit_energy)

        self.pbc = pbc

        self.get_vector = GetVector(pbc)
        self.gather_atoms = gather_vectors

    def length_unit(self):
        return self.units.length_unit()

    def energy_unit(self):
        return self.units.energy_unit()

    def set_pbc(self, pbc=None):
        self.pbc = pbc
        self.get_vector.set_pbc(pbc)
        return self

    def construct(self,
                  coordinates,
                  neighbour_vectors,
                  neighbour_distances,
                  neighbour_index,
                  neighbour_mask=None,
                  pbc_box=None
                  ):
        raise NotImplementedError
