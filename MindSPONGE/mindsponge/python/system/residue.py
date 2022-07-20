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
Residue
"""

from operator import itemgetter
import numpy as np
import mindspore as ms
from mindspore.ops import functional as F
from mindspore.nn import Cell
from mindspore.common import Tensor
from mindspore import numpy as msnp

from ..function.functions import get_integer
from ..data.elements import elements, element_set, element_dict, atomic_mass


class Residue(Cell):
    r"""Cell for residue in molecule

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

        name (str):             Name of the residue. Default: 'MOL'

        Symbols:

            B:  Batchsize, i.e. number of walkers in simulation

            A:  Number of atoms.

            b:  Number of bonds.

    """

    def __init__(self,
                 atom_name: list = None,
                 atom_type: list = None,
                 atom_mass: Tensor = None,
                 atom_charge: Tensor = None,
                 atomic_number: Tensor = None,
                 bond: Tensor = None,
                 head_atom: int = None,
                 tail_atom: int = None,
                 start_index: int = 0,
                 name: str = 'MOL'
                 ):

        super().__init__()

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

        if atom_type is None:
            self.atom_type = self.atom_name.copy()
        else:
            self.atom_type = np.array(atom_type)
            if self.atom_type.ndim == 1:
                self.atom_type = np.expand_dims(self.atom_type, 0)
            if self.atom_type.shape != self.atom_name.shape:
                raise ValueError('The shape of "atom_type" ' + str(self.atom_type.shape) +
                                 ' must be equal to the shape of "atom_name" ' + str(self.atom_name.shape) + '!')

        self.num_atoms = self.atom_name.shape[-1]
        self.multi_system = self.atom_name.shape[0]

        self.start_index = get_integer(start_index)
        # (A'')
        self._index = msnp.arange(self.num_atoms)
        self.system_index = self._index + start_index

        # (1,A') or (B,A')
        if atom_mass is None:
            if atomic_number is None:
                self.atom_mass = msnp.ones(
                    self.atom_name.shape, dtype=np.float32)
            else:
                self.atom_mass = Tensor(
                    atomic_mass[self.atomic_number.asnumpy()], ms.float32)
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

        # (B,A')
        self.atom_charge = atom_charge
        if atom_charge is not None:
            self.atom_charge = Tensor(atom_charge, ms.float32)
            if self.atom_charge.ndim == 1:
                self.atom_charge = F.expand_dims(self.atom_charge, 0)
            if self.atom_charge.ndim != 2:
                raise ValueError('The rank of "atom_charge" must be 1 or 2!')
            if self.atom_charge.shape[-1] != self.num_atoms:
                raise ValueError('The last dimension of atom_charge (' + str(self.atom_charge.shape[-1]) +
                                 ') must be equal to the num_atoms (' + str(self.num_atoms) + ')!')
            if self.atom_charge.shape[0] != self.multi_system and self.atom_charge.shape[0] != 1:
                raise ValueError('The first dimension of atom_charge (' + str(self.atom_charge.shape[0]) +
                                 ') must be equal to 1 or the number of the number of system multi_system (' +
                                 str(self.multi_system) + ')!')

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

        # (B,1)
        self.head_atom = head_atom
        if head_atom is not None:
            self.head_atom = Tensor(head_atom, ms.int32).reshape(-1, 1)
            if self.head_atom.shape[0] != self.multi_system and self.head_atom.shape[0] != 1:
                raise ValueError('The first dimension of head_atom (' + str(self.head_atom.shape[0]) +
                                 ') does not match the number of system multi_system (' + str(self.multi_system) + ')!')
            if (self.head_atom >= self.num_atoms).any():
                raise ValueError(
                    'The value of head_atom has exceeds the number of atoms.')

        # (B,1)
        self.tail_atom = tail_atom
        if tail_atom is not None:
            self.tail_atom = Tensor(tail_atom, ms.int32).reshape(-1, 1)
            if self.tail_atom.shape[0] != self.multi_system and self.tail_atom.shape[0] != 1:
                raise ValueError('The first dimension of tail_atom (' + str(self.tail_atom.shape[0]) +
                                 ') does not match the number of system multi_system (' + str(self.multi_system) + ')!')
            if (self.tail_atom >= self.num_atoms).any():
                raise ValueError(
                    'The value of tail_atom has exceeds the number of atoms.')

    @property
    def name(self) -> str:
        return str(self._name)

    def add_atom(self,
                 atom_name: str = None,
                 atom_type: str = None,
                 atom_mass: float = None,
                 atom_charge: float = None,
                 atomic_number: str = None,
                 #  index: int = -1,
                 ):
        """set atom"""

        if atom_name is None and atomic_number is None:
            raise ValueError('atom_name and atomic_number cannot both be None')

        shape = (self.multi_system, 1)

        if atom_name is not None:
            atom_name = np.array(atom_name, np.str_)
            atom_name = np.broadcast_to(atom_name, shape)

        if atomic_number is not None:
            atomic_number = Tensor(atomic_number, ms.int32)
            atomic_number = msnp.broadcast_to(atomic_number, shape)

        if atom_name is None:
            atom_name = elements[atomic_number.asnumpy()]

        if atom_mass is None:
            if atomic_number is None:
                atom_mass = msnp.ones(atom_name.shape, dtype=np.float32)
            else:
                atom_mass = Tensor(
                    atomic_mass[atomic_number.asnumpy()], ms.float32)
        else:
            atom_mass = Tensor(atom_mass, ms.float32)
            atom_mass = np.broadcast_to(atom_mass, shape)

        if atomic_number is None:
            atom_name_list = atom_name.reshape(-1).tolist()
            if len(set(atom_name_list) - element_set) == 0:
                atomic_number = itemgetter(*atom_name_list)(element_dict)
                atomic_number = Tensor(
                    atomic_number, ms.int32).reshape(atom_name.shape)
            else:
                atomic_number = msnp.ones(atom_name.shape, ms.int32)

        if atomic_number.shape != atom_name.shape:
            if atomic_number.shape[-1] == atom_name.shape[-1]:
                if atomic_number.shape[0] == 1:
                    atomic_number = msnp.broadcast_to(
                        atomic_number, atom_name.shape)
                elif atom_name.shape[0] == 1:
                    atom_name = msnp.broadcast_to(
                        atom_name, atomic_number.shape)

            raise ValueError('The shape of "atomic_number" '+str(atomic_number) +
                             ' does not match the shape of "atom_name" '+str(atom_name)+'!')

        atom_mask = F.logical_and(atomic_number > 0, atom_mass > 0)
        inv_mass = msnp.where(atom_mask, msnp.reciprocal(atom_mass), 0)

        if atom_type is None:
            atom_type = atom_name.copy()
        else:
            atom_type = np.array(atom_type)
            atom_type = np.broadcast_to(atom_type, shape)

        if atom_charge is not None:
            atom_charge = Tensor(atom_charge, ms.float32)
            atom_charge = np.broadcast_to(atom_charge, shape)

        self.atom_name = np.concatenate((self.atom_name, atom_name), axis=-1)
        self.atom_type = np.concatenate((self.atom_type, atom_type), axis=-1)
        self.atom_mass = F.concat((self.atom_mass, atom_mass), -1)
        self.atom_mask = F.concat((self.atom_mask, atom_mask), -1)
        self.atomic_number = F.concat((self.atomic_number, atomic_number), -1)
        self.inv_mass = F.concat((self.inv_mass, inv_mass), -1)
        if self.atom_charge is None and atom_charge is not None:
            self.atom_charge = msnp.zeros(
                (self.multi_system, self.num_atoms), ms.float32)
        if self.atom_charge is not None and atom_charge is None:
            atom_charge = msnp.zeros((self.multi_system, 1), ms.float32)
        if atom_charge is not None or self.atom_charge is not None:
            self.atom_charge = F.concat((self.atom_charge, atom_charge), -1)

        self.num_atoms = self.atom_name.shape[-1]
        self._index = msnp.arange(self.num_atoms)
        self.system_index = self._index + self.start_index
        self.natom_tensor = msnp.sum(
            F.cast(self.atom_mask, ms.int32), -1, keepdims=True)
        self.total_mass = msnp.sum(self.atom_mass, -1, keepdims=True)

        # TODO: self.reorder_atoms(index)

        return self

    # def reorder_atoms(self, index):
    #     """reorder atoms"""
    #     # TODO
    #     return self

    # def build_atoms(self, template: dict):
    #     """build atoms"""
    #     return self

    # def build_atom_mass(self, template:dict):
    #     if 'HIE' in self.name:
    #         self.name = self.name.replace('HIE', 'HIS')
    #     self.atom_mass = Tensor(np.array([template['mass_dict'][self.name][atom_name] for atom_name in self.atom_name[0]],
    #                                      np.float32).reshape(self.atom_name.shape))
    #     return self
    #
    # def build_atomic_number(self, template:dict):
    #     if 'HIE' in self.name:
    #         self.name = self.name.replace('HIE', 'HIS')
    #     self.atomic_number = Tensor(np.array([template['atomic_number'][self.name][atom_name] for atom_name in self.atom_name[0]],
    #                                      np.float32).reshape(self.atom_name.shape))
    #     return self
    #
    # def build_atom_types(self, force_parameters: dict):
    #     if 'HIE' in self.name:
    #         self.name = self.name.replace('HIE', 'HIS')
    #     self.atom_type = np.array([force_parameters['amino_dict'][self.name][atom_name] for atom_name in self.atom_name[0]],
    #                               np.str_).reshape(self.atom_name.shape)
    #     return self
    #
    # def build_atom_charge(self, force_parameters: dict):
    #     if 'HIE' in self.name:
    #         self.name = self.name.replace('HIE', 'HIS')
    #     mapping_index = np.where(
    #         np.array(list(force_parameters['amino_dict'][self.name].keys())) == self.atom_name[0][:, None])[-1]
    #     self.atom_charge = Tensor(np.array(force_parameters['charge_dict'][self.name])[mapping_index].reshape(self.atom_name.shape))
    #     return self
    #
    # def build_bonds(self, template: dict):
    #     if 'HIE' in self.name:
    #         self.name = self.name.replace('HIE', 'HIS')
    #     mapping_index = np.where(self.atom_name[0] == np.array(
    #         list(template['order'][self.name]))[:, None])[-1]
    #     bonds = mapping_index[np.array(list(template['bond'][self.name]))]
    #     self.bond = Tensor(bonds.astype(np.int32))
    #     return self

    # def build_residue(self, template: dict):
    #     """build residue"""
    #     return self

    def broadcast_multiplicity(self, multi_system: int):
        """broadcast the information to the number of multiple system"""
        if multi_system <= 0:
            raise ValueError('multi_system must be larger than 0!')
        if self.multi_system > 1:
            raise ValueError('The current the number of system multi_system ('+str(self.multi_system) +
                             ') is larger than 1 and cannot be broadcast!')

        self.multi_system = multi_system
        self.atom_name = msnp.broadcast_to(
            self.atom_name, (self.multi_system, -1))
        self.atom_type = msnp.broadcast_to(
            self.atom_mass, (self.multi_system, -1))
        self.atomic_number = msnp.broadcast_to(
            self.atomic_number, (self.multi_system, -1))
        self.atom_mass = msnp.broadcast_to(
            self.atom_mass, (self.multi_system, -1))
        self.atom_mask = msnp.broadcast_to(
            self.atom_mask, (self.multi_system, -1))
        self.inv_mass = msnp.broadcast_to(
            self.inv_mass, (self.multi_system, -1))
        self.total_mass = msnp.broadcast_to(
            self.total_mass, (self.multi_system, -1))
        self.natom_tensor = msnp.broadcast_to(
            self.natom_tensor, (self.multi_system, -1))
        if self.atom_charge is not None:
            self.atom_charge = msnp.broadcast_to(
                self.atom_charge, (self.multi_system, -1))
        if self.bond is not None:
            bond_shape = (self.multi_system,) + self.bond.shape[1:]
            self.bond = msnp.broadcast_to(self.bond, bond_shape)
            self.bond_mask = msnp.broadcast_to(self.bond_mask, bond_shape)
        if self.head_atom is not None:
            self.head_atom = msnp.broadcast_to(
                self.head_atom, (self.multi_system, -1))
        if self.tail_atom is not None:
            self.tail_atom = msnp.broadcast_to(
                self.tail_atom, (self.multi_system, -1))

        return self

    def set_start_index(self, start_index: int):
        """set the start index"""
        if start_index < 0:
            raise ValueError('The start_index cannot be smaller than 0!')
        self.start_index = get_integer(start_index)
        index_shift = self.start_index - self.system_index[0]
        self.system_index += index_shift
        return self

    def construct(self, coordinate: Tensor):
        """get the coordinate of this residue"""
        return F.gather_d(coordinate, -2, self.system_index)
