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
Molecule
"""
from operator import itemgetter
import numpy as np
import mindspore as ms
from mindspore.ops import functional as F
from mindspore.common import Tensor
from mindspore import numpy as msnp

from ..data.elements import element_set, element_dict, elements
from .residue import Residue
from ..data.template import template as AATEMPLATE


class AminoAcid(Residue):
    r"""Residue of amino acid
    Args:
        atom_name (list):       Atom name. Can be ndarray or list of str. Defulat: None
        atom_type (list):       Atom type. Can be ndarray or list of str. Defulat: None
        atom_mass (Tensor):     Tensor of shape (B, A). Data type is float.
                                Atom mass. Defulat: None
        atom_charge (Tensor):   Tensor of shape (B, A). Data type is float.
                                Atom charge. Defulat: None
        atomic_number (Tensor): Tensor of shape (B, A). Data type is float.
                                Atomic number. Defulat: None
        bond (Tensor):          Tensor of shape (B, b, 2) or (1, b, 2). Data type is int.
                                Bond index. Defulat: None
        head_atom (int):        Index of the head atom to connect with the previous residue.
                                Default: None
        tail_atom (int):        Index of the tail atom to connect with the next residue.
                                Default: None
        start_index (int):      The start index of the first atom in this residue.
        name (str):             Name of the residue. Default: 'AMI'
        Symbols:
            B:  Batchsize, i.e. number of walkers in simulation
            A:  Number of atoms.
            b:  Number of bonds.
    """

    def __init__(self,
                 atom_name: str = None,
                 atom_type: str = None,
                 atom_mass: float = None,
                 atom_charge: float = None,
                 atomic_number: str = None,
                 bond: int = None,
                 head_atom: int = None,
                 tail_atom: int = None,
                 start_index: int = 0,
                 name: str = 'AMI',
                 template: str = None
                 ):

        super().__init__(
            atom_name=atom_name,
            atom_type=atom_type,
            atom_mass=atom_mass,
            atom_charge=atom_charge,
            atomic_number=atomic_number,
            bond=bond,
            head_atom=head_atom,
            tail_atom=tail_atom,
            start_index=start_index,
            name=name,
        )

        self._name = name

        if atom_name is None and atomic_number is None:
            raise ValueError('atom_name and atomic_number cannot both be None')

        if atom_name is not None:
            self.atom_name = np.array(atom_name, np.str_)
            if self.atom_name.ndim == 1:
                self.atom_name = np.expand_dims(self.atom_name, 0)
            if self.atom_name.ndim != 2:
                raise ValueError('The rank of "atom_name" must be 1 or 2!')

        if atomic_number is not None:
            self.atomic_number = Tensor(atomic_number, ms.int32)
            if self.atomic_number.ndim == 1:
                self.atomic_number = F.expand_dims(self.atomic_number, 0)
            if self.atomic_number.ndim != 2:
                raise ValueError('The rank of "atomic_number" must be 1 or 2!')

        if atom_name is None:
            self.atom_name = elements[self.atomic_number.asnumpy()]

        if atomic_number is None:
            atom_name_list = self.atom_name.reshape(-1).tolist()
            if len(set(atom_name_list) - element_set) == 0:
                atomic_number = itemgetter(*atom_name_list)(element_dict)
                self.atomic_number = Tensor(
                    atomic_number, ms.int32).reshape(self.atom_name.shape)
            else:
                self.atomic_number = msnp.ones(self.atom_name.shape, ms.int32)

        if self.atomic_number.shape != self.atom_name.shape:
            if self.atomic_number.shape[-1] == self.atom_name.shape[-1]:
                if self.atomic_number.shape[0] == 1:
                    self.atomic_number = msnp.broadcast_to(
                        self.atomic_number, self.atom_name.shape)
                elif self.atom_name.shape[0] == 1:
                    self.atom_name = msnp.broadcast_to(
                        self.atom_name, self.atomic_number.shape)

            raise ValueError('The shape of "atomic_number" ' + str(self.atomic_number) +
                             ' does not match the shape of "atom_name" ' + str(self.atom_name) + '!')

        if template is None:
            self.template = AATEMPLATE

        else:
            self.template = template

        if atomic_number is None:
            atom_name_list = self.atom_name.reshape(-1).tolist()
            if len(set(atom_name_list) - element_set) == 0:
                atomic_number = itemgetter(*atom_name_list)(element_dict)
                self.atomic_number = Tensor(
                    atomic_number, ms.int32).reshape(self.atom_name.shape)
            else:
                self.atomic_number = msnp.ones(self.atom_name.shape, ms.int32)

        if self.atomic_number.shape != self.atom_name.shape:
            if self.atomic_number.shape[-1] == self.atom_name.shape[-1]:
                if self.atomic_number.shape[0] == 1:
                    self.atomic_number = msnp.broadcast_to(
                        self.atomic_number, self.atom_name.shape)
                elif self.atom_name.shape[0] == 1:
                    self.atom_name = msnp.broadcast_to(
                        self.atom_name, self.atomic_number.shape)

            raise ValueError('The shape of "atomic_number" ' + str(self.atomic_number) +
                             ' does not match the shape of "atom_name" ' + str(self.atom_name) + '!')

        # (1,A') or (B,A')
        if atom_mass is None:
            self.atom_mass = None
            self.build_atom_mass(self.template)
            if self.atom_mass.ndim == 1:
                self.atom_mass = F.expand_dims(self.atom_mass, 0)
            if self.atom_mass.ndim != 2:
                raise ValueError('The rank of "atom_mass" must be 1 or 2!')
            if self.atom_mass.shape[-1] != self.num_atoms:
                raise ValueError('The last dimension of atom_mass (' + str(self.atom_mass.shape[-1]) +
                                 ') must be equal to the number of atoms (' + str(self.num_atoms) + ')!')
            if self.atom_mass.shape[0] > 1 and self.atom_mass.shape[0] != self.multi_system:
                raise ValueError('The first dimension of atom_mass (' + str(self.atom_mass.shape[0]) +
                                 ') does not match the number of the number of system multi_system (' +
                                 str(self.multi_system) + ')!')
        else:
            self.atom_mass = Tensor(atom_mass, ms.float32)
            if self.atom_mass.ndim == 1:
                self.atom_mass = F.expand_dims(self.atom_mass, 0)
            if self.atom_mass.ndim != 2:
                raise ValueError('The rank of "atom_mass" must be 1 or 2!')
            if self.atom_mass.shape[-1] != self.num_atoms:
                raise ValueError('The last dimension of atom_mass (' + str(self.atom_mass.shape[-1]) +
                                 ') must be equal to the number of atoms (' + str(self.num_atoms) + ')!')
            if self.atom_mass.shape[0] > 1 and self.atom_mass.shape[0] != self.multi_system:
                raise ValueError('The first dimension of atom_mass (' + str(self.atom_mass.shape[0]) +
                                 ') does not match the number of the number of system multi_system (' +
                                 str(self.multi_system) + ')!')

        # (B,A')
        self.atom_mask = F.logical_and(
            self.atomic_number > 0, self.atom_mass > 0)
        self.inv_mass = msnp.where(
            self.atom_mask, msnp.reciprocal(self.atom_mass), 0)
        # (B,1)
        self.natom_tensor = msnp.sum(
            F.cast(self.atom_mask, ms.int32), -1, keepdims=True)
        self.total_mass = msnp.sum(self.atom_mass, -1, keepdims=True)

        # (B,C,2)
        self.bond = bond
        self.bond_mask = None
        if bond is not None:
            self.bond = Tensor(bond, ms.int32)
            if self.bond.shape[-1] != 2:
                raise ValueError('The last dimension of bond must 2!')
            if self.bond.ndim == 2:
                self.bond = F.expand_dims(self.bond, 0)
            self.bond_mask = self.bond < self.num_atoms
        else:
            self.build_bonds(self.template)
            if self.bond.shape[-1] != 2:
                raise ValueError('The last dimension of bond must 2!')
            if self.bond.ndim == 2:
                self.bond = F.expand_dims(self.bond, 0)
            self.bond_mask = self.bond < self.num_atoms

    def build_atom_mass(self, template: dict):
        if 'HIE' in self.name:
            self.name = self.name.replace('HIE', 'HIS')
        self.atom_mass = Tensor(np.array([template['mass_dict'][self.name][atom_name]
                                          for atom_name in self.atom_name[0]],
                                         np.float32).reshape(self.atom_name.shape))
        return self

    def build_atomic_number(self, template: dict):
        if 'HIE' in self.name:
            self.name = self.name.replace('HIE', 'HIS')
        self.atomic_number = Tensor(np.array([template['atomic_number'][self.name][atom_name]
                                              for atom_name in self.atom_name[0]],
                                             np.float32).reshape(self.atom_name.shape))
        return self

    def build_atom_types(self, force_parameters: dict):
        if 'HIE' in self.name:
            self.name = self.name.replace('HIE', 'HIS')
        self.atom_type = np.array([force_parameters['amino_dict'][self.name][atom_name]
                                   for atom_name in self.atom_name[0]],
                                  np.str_).reshape(self.atom_name.shape)
        return self

    def build_atom_charge(self, force_parameters: dict):
        if 'HIE' in self.name:
            self.name = self.name.replace('HIE', 'HIS')
        mapping_index = np.where(
            np.array(list(force_parameters['amino_dict'][self.name].keys())) == self.atom_name[0][:, None])[-1]
        self.atom_charge = Tensor(np.array(force_parameters['charge_dict'][self.name])[mapping_index].
                                  reshape(self.atom_name.shape))
        return self

    def build_bonds(self, template: dict):
        if 'HIE' in self.name:
            self.name = self.name.replace('HIE', 'HIS')
        mapping_index = np.where(self.atom_name[0] == np.array(
            list(template['order'][self.name]))[:, None])[-1]
        bonds = mapping_index[np.array(list(template['bond'][self.name]))]
        self.bond = Tensor(bonds.astype(np.int32))
        return self
