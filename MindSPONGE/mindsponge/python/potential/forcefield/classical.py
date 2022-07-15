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
"""classical ff"""
from .forcefield import ForceField


class ClassicalFF(ForceField):
    """callsical ff"""
    def __init__(
            self,
            bond_energy=None,
            angle_energy=None,
            dihedral_energy=None,
            nonbond_energy=None,
            nb14_energy=None,
            unit_length=None,
            unit_energy=None,
            harmonic_energy=None,
            pbc=None,
    ):
        super().__init__(
            unit_length=unit_length,
            unit_energy=unit_energy,
            pbc=pbc,
        )

        self.bond_energy = bond_energy
        self.angle_energy = angle_energy
        self.dihedral_energy = dihedral_energy
        self.nonbond_energy = nonbond_energy
        self.nb14_energy = nb14_energy
        self.harmonic_energy = harmonic_energy

        if self.bond_energy is not None:
            self.bond_energy.set_unit_scale(self.units)
        if self.angle_energy is not None:
            self.angle_energy.set_unit_scale(self.units)
        if self.dihedral_energy is not None:
            self.dihedral_energy.set_unit_scale(self.units)
        if self.nonbond_energy is not None:
            self.nonbond_energy.set_unit_scale(self.units)
        if self.nb14_energy is not None:
            self.nb14_energy.set_unit_scale(self.units)

    def set_pbc(self, pbc=None):
        if self.bond_energy is not None:
            self.bond_energy.set_pbc(pbc)

        if self.angle_energy is not None:
            self.angle_energy.set_pbc(pbc)

        if self.dihedral_energy is not None:
            self.dihedral_energy.set_pbc(pbc)

        if self.nonbond_energy is not None:
            self.nonbond_energy.set_pbc(pbc)

        if self.nb14_energy is not None:
            self.nb14_energy.set_pbc(pbc)

        return self

    def construct(self, coordinates,
                  neighbour_vectors,
                  neighbour_distances,
                  neighbour_index,
                  neihbour_mask=None,
                  pbc_box=None
                  ):
        e_bond = 0.
        if self.bond_energy is not None:
            e_bond = self.bond_energy(coordinates, pbc_box)

        e_angle = 0.
        if self.angle_energy is not None:
            e_angle = self.angle_energy(coordinates, pbc_box)

        e_dihedral = 0.
        if self.dihedral_energy is not None:
            e_dihedral = self.dihedral_energy(coordinates, pbc_box)

        e_ele = 0.
        e_vdw = 0.
        if self.nonbond_energy is not None:
            e_ele, e_vdw = self.nonbond_energy(neighbour_distances, neighbour_index, neihbour_mask, pbc_box)

        e14_ele = 0.
        e14_vdw = 0.
        if self.nb14_energy is not None:
            e14_ele, e14_vdw = self.nb14_energy(coordinates, pbc_box)

        e_harmonic = 0.
        if self.harmonic_energy is not None:
            e_harmonic = self.harmonic_energy(coordinates)

        e_pot = e_bond + e_angle + e_dihedral + e_ele + e_vdw + e14_ele + e14_vdw + e_harmonic

        return e_pot
