# Copyright 2021 The AIMM Group at Shenzhen Bay Laboratory & Peking University
#
# Developer: Yi Isaac Yang, Dechin Chen, Jun Zhang, Yijie Xia
#
# Email: yangyi@szbl.ac.cn
#
# This code is a part of MindSPONGE.
#
# The Cybertron-Code is open-source software based on the AI-framework:
# MindSpore (https://www.mindspore.cn/)
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
"""
Water
"""

from mindspore.nn import CellList
from mindspore.common import Tensor

from .molecule import Molecule
from .residue import Residue
from ..function.functions import get_integer

class Water(Molecule):
    r"""Water molecule

    Args:


    """

    def __init__(self,
                 number: int = 1,
                 num_points: int = 3,
                 coordinate: float = None,
                 pbc_box: Tensor = None,
                 length_unit: str = None,
                 name: str = 'WAT',
                 ):

        super().__init__(
            pbc_box=pbc_box,
            length_unit=length_unit,
            name=name,
        )

        self.num_points = get_integer(num_points)
        if self.num_points == 3:
            atom_name = ['O', 'H1', 'H2']
            atom_type = ['OW', 'HW', 'HW']
            atom_mass = [16.0, 1.008, 1.008]
            atom_charge = [-0.834, 0.417, 0.417]
            atomic_number = [8, 1, 1]
            bond = [[0, 1], [0, 2]]
        elif self.num_points == 4:
            atom_name = ['O', 'H1', 'H2', 'EP']
            atom_type = ['OW', 'HW', 'HW', 'EP']
            atom_mass = [16.0, 1.008, 1.008, 0]
            atom_charge = [0, 0.53, 0.52, -1.04]
            atomic_number = [8, 1, 1, 0]
            bond = [[0, 1], [0, 2], [1, 2]]
        elif self.num_points == 5:
            atom_name = ['O', 'H1', 'H2', 'EP1', 'EP2']
            atom_type = ['OW', 'HW', 'HW', 'EP', 'EP']
            atom_mass = [16.0, 1.008, 1.008, 0, 0]
            atom_charge = [0, 0.241, 0.241, -0.241, -0.241]
            atomic_number = [8, 1, 1, 0, 0]
            bond = [[0, 1], [0, 2], [1, 2]]
        else:
            raise ValueError(
                'The points of water model must be 3, 4 or 5 but got: '+str(self.num_points))

        self.residue = CellList([
            Residue(
                atom_name=atom_name,
                atom_type=atom_type,
                atom_mass=atom_mass,
                atom_charge=atom_charge,
                atomic_number=atomic_number,
                bond=bond,
                name=name,
            ) for _ in range(number)
        ])

        self._build_system()

        self.set_coordianate(coordinate)
