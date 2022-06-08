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
"""
Module for getting Meta Parameters.
The correspond method in Xponge is:
>>> import Xponge
>>> import Xponge.forcefield.AMBER.ff14SB
>>> Save_SPONGE_Input(ALA,'ALA')
>>> Save_PDB(ALA, "ALA.pdb")
"""
import itertools
from itertools import product
from mindspore import load_checkpoint, nn, Tensor
import mindspore as ms
import numpy as np

from ..common.checkpoint import save_checkpoint
from ..common.functions import str_to_tensor, tensor_to_str
from ..common.ff14SB_constants import restype_name_to_atom14_names, amino_dict, bond_pairs, charge_dict
from ..common.ff14SB_constants import vdw_params, improper_params, dihedral_params, angle_params, bond_params

backbone_atoms = np.array(['N', 'CA', 'C', 'O'], np.str_)
include_backbone_atoms = np.array(['OXT'], np.str_)


class Params(object):
    """ Getting parameters for given bonds and atom types.
    Args:
        atom_types(np.str_): The atom types defined in forcefields.
        mass(np.float32): The atom mass.
        atom_names(np.str_): Unique atom names in an amino acid.
    Parameters:
        bonds(np.int32): The bond pairs defined for a given molecule.
    """

    def __init__(self, atom_types, mass=None, atom_names=None):
        self.atom_types = atom_types
        self.atom_names = atom_names
        atom_nums = atom_types.shape[-1]
        assert atom_nums > 0
        self.atom_nums = atom_nums
        self.amino_atoms = {key: len(amino_dict[key].keys()) for key in amino_dict.keys()}
        self.mass_dict = {'H': 1.008, 'C': 12.010, 'O': 16.000, 'N': 14.010,
                          'HC': 1.008, 'CT': 12.010, 'CX': 12.010, 'H1': 1.008,
                          'C8': 12.010, 'N2': 14.010, 'CA': 12.010, '2C': 12.010,
                          'CO': 12.010, 'O2': 16.000, 'SH': 32.060, 'HS': 1.008,
                          'CC': 12.010, 'NB': 14.010, 'CR': 12.010, 'H5': 1.008,
                          'NA': 14.010, 'CW': 12.010, 'H4': 1.008, '3C': 12.010,
                          'HP': 1.008, 'N3': 14.010, 'S': 32.06, 'HA': 1.008,
                          'OH': 16.000, 'HO': 1.008, 'C*': 12.010, 'CN': 12.010,
                          'CB': 12.010, 'CV': 12.010}
        self.htypes = np.array(['H', 'HC', 'H1', 'HS', 'H5', 'H4', 'HP', 'HA', 'HO'], np.str_)
        if mass is not None:
            self.mass = mass
        else:
            self.mass = np.empty(atom_nums)
            self.get_mass(atom_types)
        # self._data_path = this_directory
        self._bonds = bond_params
        self._angles = angle_params
        self._dihedrals = dihedral_params
        self._idihedrals = improper_params
        self._wildcard = np.array(['X'], dtype=np.str_)
        self.bond_params = None
        self.angle_params = None
        self.dihedral_params = None
        self.improper_dihedral_params = None
        self.excludes = np.empty(atom_nums)[:, None]
        self.vdw_param = np.empty((atom_nums, 2))
        self.nb14_index = None

    def get_bond_params(self, bonds, atom_types):
        """get bond params"""
        names = atom_types[bonds]
        bond_types = np.append(np.char.add(np.char.add(names[:, 0], '-'), names[:, 1])[None, :],
                               np.char.add(np.char.add(names[:, 1], '-'), names[:, 0])[None, :],
                               axis=0)
        bond_id = -1 * np.ones(bonds.shape[0], dtype=np.int32)
        mask_id = np.where(bond_types.reshape(bonds.shape[-2] * 2, 1) == self._bonds['name'])

        if mask_id[0].shape[0] < bonds.shape[0]:
            raise ValueError("Elements in atom types not recognized!")

        left_id = np.where(mask_id[0] < bonds.shape[0])[0]
        bond_id[mask_id[0][left_id]] = mask_id[1][left_id]
        right_id = np.where(mask_id[0] >= bonds.shape[0])[0]
        bond_id[mask_id[0][right_id] - bonds.shape[0]] = mask_id[1][right_id]
        bond_params = np.append(self._bonds['k'][bond_id][None, :],
                                self._bonds['b'][bond_id][None, :],
                                axis=0).T
        return bond_params

    def get_angle_params(self, angles, atom_types):
        """get angle params"""
        names = atom_types[angles]
        angle_types = np.append(np.char.add(np.char.add(np.char.add(np.char.add(names[:, 0], '-'),
                                                                    names[:, 1]), '-'), names[:, 2])[None, :],
                                np.char.add(np.char.add(np.char.add(np.char.add(names[:, 2], '-'),
                                                                    names[:, 1]), '-'), names[:, 0])[None, :],
                                axis=0)
        angle_id = -1 * np.ones(angles.shape[0], dtype=np.int32)
        mask_id = np.where(angle_types.reshape(angles.shape[0] * 2, 1) == self._angles['name'])

        left_id = np.where(mask_id[0] < angles.shape[0])[0]
        angle_id[mask_id[0][left_id]] = mask_id[1][left_id]
        right_id = np.where(mask_id[0] >= angles.shape[0])[0]
        angle_id[mask_id[0][right_id] - angles.shape[0]] = mask_id[1][right_id]
        angle_params = np.append(self._angles['k'][angle_id][None, :],
                                 self._angles['b'][angle_id][None, :],
                                 axis=0).T
        return angle_params

    def get_dihedral_params(self, dihedrals_in, atom_types, return_includes=False):
        """get dihedral params"""
        dihedrals = dihedrals_in.copy()
        standar_dihedrals = dihedrals.copy()
        names = atom_types[dihedrals]
        dihedral_types = np.append(np.char.add(np.char.add(np.char.add(np.char.add(
            np.char.add(np.char.add(names[:, 0], '-'), names[:, 1]), '-'), names[:, 2]), '-'), names[:, 3])[None, :],
            np.char.add(np.char.add(np.char.add(np.char.add(np.char.add(np.char.add(names[:, 3], '-'),
            names[:, 2]), '-'), names[:, 1]), '-'), names[:, 0])[None, :], axis=0)
        dihedral_id = -1 * np.ones(dihedrals.shape[0], dtype=np.int32)
        mask_id = np.where(dihedral_types.reshape(dihedrals.shape[0] * 2, 1) == self._dihedrals['name'])
        left_id = np.where(mask_id[0] < dihedrals.shape[0])[0]
        dihedral_id[mask_id[0][left_id]] = mask_id[1][left_id]
        right_id = np.where(mask_id[0] >= dihedrals.shape[0])[0]
        dihedral_id[mask_id[0][right_id] - dihedrals.shape[0]] = mask_id[1][right_id]
        include_1 = np.pad(np.where(dihedral_id > -1, 1, 0)[None, :], ((0, self._dihedrals['ks'].shape[1] - 1), (0, 0)),
                           mode='edge').T.flatten()[:, None]
        exclude_1 = include_1 - 1 < 0
        dihedral_params = np.pad(dihedrals[:, None, :], ((0, 0), (0, self._dihedrals['ks'].shape[1] - 1), (0, 0)),
                                 mode='edge'). \
            reshape(dihedrals.shape[0] * 4, self._dihedrals['ks'].shape[1])
        dihedral_params = np.append(dihedral_params, self._dihedrals['periodicitys'][dihedral_id].flatten()[:, None],
                                    axis=1)
        dihedral_params = np.append(dihedral_params, self._dihedrals['ks'][dihedral_id].flatten()[:, None], axis=1)
        dihedral_params = np.append(dihedral_params, self._dihedrals['phi0s'][dihedral_id].flatten()[:, None], axis=1)
        dihedral_params *= include_1
        dihedrals[:, 0] = -1 * np.ones_like(dihedrals[:, 0])
        names = atom_types[dihedrals]
        dihedral_types = np.append(np.char.add(np.char.add(np.char.add(np.char.add(np.char.add(
            np.char.add(names[:, 0], '-'), names[:, 1]), '-'), names[:, 2]), '-'), names[:, 3])[None, :],
            np.char.add(np.char.add(np.char.add(np.char.add(np.char.add(np.char.add(names[:, 3], '-'),
            names[:, 2]), '-'), names[:, 1]), '-'), names[:, 0])[None, :], axis=0)
        mask_id = np.where(dihedral_types.reshape(dihedrals.shape[0] * 2, 1) == self._dihedrals['name'])
        left_id = np.where(mask_id[0] < dihedrals.shape[0])[0]
        dihedral_id[mask_id[0][left_id]] = mask_id[1][left_id]
        right_id = np.where(mask_id[0] >= dihedrals.shape[0])[0]
        dihedral_id[mask_id[0][right_id] - dihedrals.shape[0]] = mask_id[1][right_id]
        include_2 = np.pad(np.where(dihedral_id > -1, 1, 0)[None, :], ((0, self._dihedrals['ks'].shape[1] - 1), (0, 0)),
                           mode='edge').T.flatten()[:, None]
        exclude_2 = include_2 - 1 < 0
        dihedral_params_1 = np.pad(standar_dihedrals[:, None, :],
                                   ((0, 0), (0, self._dihedrals['ks'].shape[1] - 1), (0, 0)), mode='edge'). \
            reshape(dihedrals.shape[0] * 4, self._dihedrals['ks'].shape[1])
        dihedral_params_1 = np.append(dihedral_params_1,
                                      self._dihedrals['periodicitys'][dihedral_id].flatten()[:, None], axis=1)
        dihedral_params_1 = np.append(dihedral_params_1, self._dihedrals['ks'][dihedral_id].flatten()[:, None], axis=1)
        dihedral_params_1 = np.append(dihedral_params_1, self._dihedrals['phi0s'][dihedral_id].flatten()[:, None],
                                      axis=1)
        dihedral_params_1 *= include_2 * exclude_1
        dihedrals = dihedrals_in.copy()
        dihedrals[:, -1] = -1 * np.ones_like(dihedrals[:, -1])
        names = atom_types[dihedrals]
        dihedral_types = np.append(np.char.add(np.char.add(np.char.add(np.char.add(np.char.add(np.char.add(
            names[:, 0], '-'), names[:, 1]), '-'), names[:, 2]), '-'), names[:, 3])[None, :], np.char.add(
            np.char.add(np.char.add(np.char.add(np.char.add(np.char.add(names[:, 3], '-'), names[:, 2]), '-'),
            names[:, 1]), '-'), names[:, 0])[None, :], axis=0)
        mask_id = np.where(dihedral_types.reshape(dihedrals.shape[0] * 2, 1) == self._dihedrals['name'])
        left_id = np.where(mask_id[0] < dihedrals.shape[0])[0]
        dihedral_id[mask_id[0][left_id]] = mask_id[1][left_id]
        right_id = np.where(mask_id[0] >= dihedrals.shape[0])[0]
        dihedral_id[mask_id[0][right_id] - dihedrals.shape[0]] = mask_id[1][right_id]
        include_4 = np.pad(np.where(dihedral_id > -1, 1, 0)[None, :], ((0, self._dihedrals['ks'].shape[1] - 1), (0, 0)),
                           mode='edge').T.flatten()[:, None]
        exclude_4 = include_4 - 1 < 0
        dihedral_params_3 = np.pad(standar_dihedrals[:, None, :],
                                   ((0, 0), (0, self._dihedrals['ks'].shape[1] - 1), (0, 0)), mode='edge'). \
            reshape(dihedrals.shape[0] * 4, self._dihedrals['ks'].shape[1])
        dihedral_params_3 = np.append(dihedral_params_3,
                                      self._dihedrals['periodicitys'][dihedral_id].flatten()[:, None], axis=1)
        dihedral_params_3 = np.append(dihedral_params_3, self._dihedrals['ks'][dihedral_id].flatten()[:, None], axis=1)
        dihedral_params_3 = np.append(dihedral_params_3, self._dihedrals['phi0s'][dihedral_id].flatten()[:, None],
                                      axis=1)
        dihedral_params_3 *= include_2 * exclude_1 * exclude_4
        dihedrals[:, 0] = -1 * np.ones_like(dihedrals[:, 0])
        dihedrals[:, -1] = -1 * np.ones_like(dihedrals[:, -1])
        names = atom_types[dihedrals]
        dihedral_types = np.append(np.char.add(np.char.add(np.char.add(np.char.add(np.char.add(np.char.add(
            names[:, 0], '-'), names[:, 1]), '-'), names[:, 2]), '-'), names[:, 3])[None, :], np.char.add(
            np.char.add(np.char.add(np.char.add(np.char.add(np.char.add(names[:, 3], '-'), names[:, 2]), '-'),
            names[:, 1]), '-'), names[:, 0])[None, :], axis=0)
        mask_id = np.where(dihedral_types.reshape(dihedrals.shape[0] * 2, 1) == self._dihedrals['name'])
        left_id = np.where(mask_id[0] < dihedrals.shape[0])[0]
        dihedral_id[mask_id[0][left_id]] = mask_id[1][left_id]
        right_id = np.where(mask_id[0] >= dihedrals.shape[0])[0]
        dihedral_id[mask_id[0][right_id] - dihedrals.shape[0]] = mask_id[1][right_id]
        include_3 = np.pad(np.where(dihedral_id > -1, 1, 0)[None, :], ((0, self._dihedrals['ks'].shape[1] - 1), (0, 0)),
                           mode='edge').T.flatten()[:, None]
        dihedral_params_2 = np.pad(standar_dihedrals[:, None, :],
                                   ((0, 0), (0, self._dihedrals['ks'].shape[1] - 1), (0, 0)), mode='edge'). \
            reshape(dihedrals.shape[0] * 4, self._dihedrals['ks'].shape[1])
        dihedral_params_2 = np.append(dihedral_params_2,
                                      self._dihedrals['periodicitys'][dihedral_id].flatten()[:, None], axis=1)
        dihedral_params_2 = np.append(dihedral_params_2, self._dihedrals['ks'][dihedral_id].flatten()[:, None], axis=1)
        dihedral_params_2 = np.append(dihedral_params_2, self._dihedrals['phi0s'][dihedral_id].flatten()[:, None],
                                      axis=1)
        dihedral_params_2 *= include_3 * exclude_1 * exclude_2 * exclude_4
        dihedral_params += dihedral_params_1 + dihedral_params_2 + dihedral_params_3
        ks0_condition = dihedral_params[:, -2] != 0
        if not return_includes:
            return dihedral_params[np.where(ks0_condition)[0]]
        return dihedral_params, (include_1, include_2, include_4, include_3)

    def _get_idihedral_func1(self, dihedrals, standar_dihedrals, dihedral_id, mask_id):
        """nloc func 1"""
        left_id = np.where(mask_id[0] < dihedrals.shape[0])[0]
        dihedral_id[mask_id[0][left_id]] = mask_id[1][left_id]
        right_id = np.where(mask_id[0] >= dihedrals.shape[0])[0]
        dihedral_id[mask_id[0][right_id] - dihedrals.shape[0]] = mask_id[1][right_id]
        include_1 = np.where(dihedral_id > -1, 1, 0)
        exclude_1 = include_1 - 1 < 0

        dihedral_params = standar_dihedrals
        dihedral_params = np.append(dihedral_params, self._idihedrals['periodicity'][dihedral_id].flatten()[:, None],
                                    axis=1)
        dihedral_params = np.append(dihedral_params, self._idihedrals['k'][dihedral_id].flatten()[:, None], axis=1)
        dihedral_params = np.append(dihedral_params, self._idihedrals['phi0'][dihedral_id].flatten()[:, None], axis=1)
        dihedral_params *= include_1[:, None]
        return include_1, exclude_1, dihedral_params

    def _get_idihedral_func2(self, dihedrals, standar_dihedrals, exclude_1, dihedral_id, mask_id):
        """ nloc func 2"""
        left_id = np.where(mask_id[0] < dihedrals.shape[0])[0]
        dihedral_id[mask_id[0][left_id]] = mask_id[1][left_id]
        right_id = np.where(mask_id[0] >= dihedrals.shape[0])[0]
        dihedral_id[mask_id[0][right_id] - dihedrals.shape[0]] = mask_id[1][right_id]
        include_2 = np.where(dihedral_id > -1, 1, 0)
        exclude_2 = include_2 - 1 < 0
        dihedral_params_1 = standar_dihedrals
        dihedral_params_1 = np.append(dihedral_params_1,
                                      self._idihedrals['periodicity'][dihedral_id].flatten()[:, None],
                                      axis=1)
        dihedral_params_1 = np.append(dihedral_params_1, self._idihedrals['k'][dihedral_id].flatten()[:, None], axis=1)
        dihedral_params_1 = np.append(dihedral_params_1, self._idihedrals['phi0'][dihedral_id].flatten()[:, None],
                                      axis=1)
        dihedral_params_1 *= include_2[:, None] * exclude_1[:, None]
        return include_2, exclude_2, dihedral_params_1

    def _get_dihedral_types(self, names):
        """get dihedral types"""
        dihedral_types = np.append(np.char.add(
            np.char.add(np.char.add(np.char.add(np.char.add(np.char.add(names[:, 0], '-'),
                                                            names[:, 1]), '-'), names[:, 2]),
                        '-'), names[:, 3])[None, :],
                                   np.char.add(
                                       np.char.add(np.char.add(np.char.add(np.char.add(np.char.add(names[:, 3], '-'),
                                                                                       names[:, 2]), '-'), names[:, 1]),
                                                   '-'), names[:, 0])[None, :],
                                   axis=0)
        return  dihedral_types

    def _get_idihedral_params(self, dihedrals_in, atom_types, return_includes=False):
        """get idihedral params"""
        dihedrals = dihedrals_in.copy()
        standar_dihedrals = dihedrals.copy()
        names = atom_types[dihedrals]
        dihedral_types = self._get_dihedral_types(names)
        dihedral_id = -1 * np.ones(dihedrals.shape[0], dtype=np.int32)
        mask_id = np.where(dihedral_types.reshape(dihedrals.shape[0] * 2, 1) == self._idihedrals['name'])

        # Constructing A-B-C-D
        include_1, exclude_1, dihedral_params = self._get_idihedral_func1(dihedrals, \
                standar_dihedrals, dihedral_id, mask_id)

        # Constructing X-B-C-D and D-C-B-X
        dihedrals[:, 0] = -1 * np.ones_like(dihedrals[:, 0])
        names = atom_types[dihedrals]
        dihedral_types = self._get_dihedral_types(names)
        mask_id = np.where(dihedral_types.reshape(dihedrals.shape[0] * 2, 1) == self._idihedrals['name'])
        include_2, exclude_2, dihedral_params_1 = self._get_idihedral_func2(dihedrals, \
                standar_dihedrals, exclude_1, dihedral_id, mask_id)

        # Constructing A-B-C-X and X-C-B-A
        dihedrals = dihedrals_in.copy()
        dihedrals[:, -1] = -1 * np.ones_like(dihedrals[:, -1])
        names = atom_types[dihedrals]
        dihedral_types = self._get_dihedral_types(names)
        mask_id = np.where(dihedral_types.reshape(dihedrals.shape[0] * 2, 1) == self._idihedrals['name'])
        left_id = np.where(mask_id[0] < dihedrals.shape[0])[0]
        dihedral_id[mask_id[0][left_id]] = mask_id[1][left_id]
        right_id = np.where(mask_id[0] >= dihedrals.shape[0])[0]
        dihedral_id[mask_id[0][right_id] - dihedrals.shape[0]] = mask_id[1][right_id]
        include_4 = np.where(dihedral_id > -1, 1, 0)
        exclude_4 = include_4 - 1 < 0
        dihedral_params_3 = standar_dihedrals
        dihedral_params_3 = np.append(dihedral_params_3,
                                      self._idihedrals['periodicity'][dihedral_id].flatten()[:, None],
                                      axis=1)
        dihedral_params_3 = np.append(dihedral_params_3, self._idihedrals['k'][dihedral_id].flatten()[:, None], axis=1)
        dihedral_params_3 = np.append(dihedral_params_3, self._idihedrals['phi0'][dihedral_id].flatten()[:, None],
                                      axis=1)
        dihedral_params_3 *= include_2[:, None] * exclude_1[:, None] * exclude_4[:, None]

        # Constructing X-A-B-X
        dihedrals = dihedrals_in.copy()
        dihedrals[:, 0] = -1 * np.ones_like(dihedrals[:, 0])
        dihedrals[:, -1] = -1 * np.ones_like(dihedrals[:, -1])
        names = atom_types[dihedrals]
        dihedral_types = self._get_dihedral_types(names)
        mask_id = np.where(dihedral_types.reshape(dihedrals.shape[0] * 2, 1) == self._idihedrals['name'])
        left_id = np.where(mask_id[0] < dihedrals.shape[0])[0]
        dihedral_id[mask_id[0][left_id]] = mask_id[1][left_id]
        right_id = np.where(mask_id[0] >= dihedrals.shape[0])[0]
        dihedral_id[mask_id[0][right_id] - dihedrals.shape[0]] = mask_id[1][right_id]
        include_3 = np.where(dihedral_id > -1, 1, 0)
        exclude_3 = include_3 - 1 < 0
        dihedral_params_2 = standar_dihedrals
        dihedral_params_2 = np.append(dihedral_params_2,
                                      self._idihedrals['periodicity'][dihedral_id].flatten()[:, None],
                                      axis=1)
        dihedral_params_2 = np.append(dihedral_params_2, self._idihedrals['k'][dihedral_id].flatten()[:, None], axis=1)
        dihedral_params_2 = np.append(dihedral_params_2, self._idihedrals['phi0'][dihedral_id].flatten()[:, None],
                                      axis=1)
        dihedral_params_2 *= include_3[:, None] * exclude_1[:, None] * exclude_2[:, None] * exclude_4[:, None]

        # Constructing X-X-C-D and D-C-X-X
        dihedrals = dihedrals_in.copy()
        dihedrals[:, 0] = -1 * np.ones_like(dihedrals[:, 0])
        dihedrals[:, 1] = -1 * np.ones_like(dihedrals[:, 1])
        names = atom_types[dihedrals]
        dihedral_types = self._get_dihedral_types(names)
        mask_id = np.where(dihedral_types.reshape(dihedrals.shape[0] * 2, 1) == self._idihedrals['name'])
        left_id = np.where(mask_id[0] < dihedrals.shape[0])[0]
        dihedral_id[mask_id[0][left_id]] = mask_id[1][left_id]
        right_id = np.where(mask_id[0] >= dihedrals.shape[0])[0]
        dihedral_id[mask_id[0][right_id] - dihedrals.shape[0]] = mask_id[1][right_id]
        include_5 = np.where(dihedral_id > -1, 1, 0)
        exclude_5 = include_5 - 1 < 0
        dihedral_params_4 = standar_dihedrals
        dihedral_params_4 = np.append(dihedral_params_4,
                                      self._idihedrals['periodicity'][dihedral_id].flatten()[:, None],
                                      axis=1)
        dihedral_params_4 = np.append(dihedral_params_4, self._idihedrals['k'][dihedral_id].flatten()[:, None], axis=1)
        dihedral_params_4 = np.append(dihedral_params_4, self._idihedrals['phi0'][dihedral_id].flatten()[:, None],
                                      axis=1)
        dihedral_params_4 *= include_5[:, None] * exclude_1[:, None] * exclude_2[:, None] * exclude_4[:, None] * \
                             exclude_3[:, None]

        dihedral_params += dihedral_params_1 + dihedral_params_2 + dihedral_params_3 + dihedral_params_4
        ks0_condition = dihedral_params[:, -2] != 0
        if not return_includes:
            return dihedral_params[np.where(ks0_condition)[0]]
        return dihedral_params, (include_1, include_2, include_4, include_3, include_5)

    def get_idihedral_params(self, idihedrals_in, atom_types, third_id):
        """get idihedral params"""
        try:
            idihedral_params, includes = self._get_idihedral_params(idihedrals_in.copy(),
                                                                    atom_types,
                                                                    return_includes=True)
        except AttributeError:
            return None

        priorities = includes[0] * 16 + includes[1] * 8 + includes[2] * 4 + includes[3] * 2 + includes[4] * 1
        for i, j, k, l in itertools.permutations(range(4), 4):
            idihedrals = np.ones_like(idihedrals_in.copy())
            idihedrals[:, 0] = idihedrals_in[:, i].copy()
            idihedrals[:, 1] = idihedrals_in[:, j].copy()
            idihedrals[:, 2] = idihedrals_in[:, k].copy()
            idihedrals[:, 3] = idihedrals_in[:, l].copy()
            this_idihedral_params, includes = self._get_idihedral_params(idihedrals, atom_types, return_includes=True)
            this_priorities = includes[0] * 16 + includes[1] * 8 + includes[2] * 4 + includes[3] * 2 + includes[4] * 1
            this_priorities *= (idihedrals[:, 2] == third_id)
            this_id = np.where(this_priorities >= priorities)[0]
            idihedral_params[this_id] = this_idihedral_params[this_id]
            priorities[this_id] = this_priorities[this_id]
        ks0_id = np.where(idihedral_params[:, -2] != 0)[0]
        return idihedral_params[ks0_id]

    def construct_angles(self, bonds, bonds_for_angle, middle_id):
        """construct angles"""
        adjacents = None
        for idx in middle_id:
            this_bonds = bonds[np.where(bonds_for_angle == idx)[0]]
            flatten_bonds = this_bonds.flatten()
            this_idx = np.delete(flatten_bonds, np.where(flatten_bonds == idx))
            yield this_idx

    def combinations(self, bonds, bonds_for_angle, middle_id):
        """combinations"""
        this_idx = self.construct_angles(bonds, bonds_for_angle, middle_id)
        id_selections = [[[0, 1]],
                         [[0, 1], [1, 2], [0, 2]],
                         [[0, 1], [1, 2], [2, 3], [0, 2], [0, 3], [1, 3]]]
        angles = None
        counter = 0
        for idx in this_idx:
            selections = id_selections[idx.size - 2]
            for selection in selections:
                if angles is None:
                    angles = np.insert(idx[selection], 1, middle_id[counter])[None, :]
                else:
                    angles = np.append(angles, np.insert(idx[selection], 1, middle_id[counter])[None, :], axis=0)
            counter += 1
        return angles

    def construct_hash(self, bonds):
        """construct hash"""
        hash_map = {}
        for i in range(len(bonds)):
            bond = tuple(bonds[i])
            hash_map[hash(bond)] = i
        return hash_map

    def trans_dangles(self, dangles, middle_id):
        """trans dangles"""
        left_id = np.isin(dangles[:, 0], middle_id[0])
        left_ele = dangles[:, 2][left_id]
        left_id = np.isin(dangles[:, 2], middle_id[0])
        left_ele = np.append(left_ele, dangles[:, 0][left_id])
        right_id = np.isin(dangles[:, 1], middle_id[0])
        right_ele = np.unique(dangles[right_id])
        right_ele = right_ele[np.where(np.isin(right_ele, middle_id, invert=True))[0]]
        sides = product(right_ele, left_ele)
        sides_array = np.array(list(sides))
        if sides_array.size == 0:
            return sides_array

        sides = sides_array[np.where(sides_array[:, 0] != sides_array[:, 1])[0]]
        left = np.append(sides[:, 0].reshape(sides.shape[0], 1),
                         np.broadcast_to(middle_id, (sides.shape[0],) + middle_id.shape), axis=1)
        dihedrals = np.append(left, sides[:, 1].reshape(sides.shape[0], 1), axis=1)
        return dihedrals

    def get_dihedrals(self, angles, dihedral_middle_id):
        """get dihedrals"""
        dihedrals = None
        for i in range(dihedral_middle_id.shape[0]):
            dangles = angles[np.where((np.isin(angles, dihedral_middle_id[i]).sum(axis=1) * \
                                       np.isin(angles[:, 1], dihedral_middle_id[i])) > 1)[0]]
            this_sides = self.trans_dangles(dangles, dihedral_middle_id[i])
            if this_sides.size == 0:
                continue
            if dihedrals is None:
                dihedrals = this_sides
            else:
                dihedrals = np.append(dihedrals, this_sides, axis=0)
        return dihedrals

    def check_idihedral(self, bonds, core_id):
        """check idihedral"""
        checked_core_id = core_id.copy()
        bonds_hash = [hash(tuple(x)) for x in bonds]
        for i in range(core_id.shape[0]):
            ids_for_idihedral = np.where(np.sum(np.isin(bonds, core_id[i]), axis=1) > 0)[0]
            bonds_for_idihedral = bonds[ids_for_idihedral]
            uniques = np.unique(bonds_for_idihedral.flatten())
            uniques = np.delete(uniques, np.where(uniques == core_id[i])[0])
            uniques_product = np.array([x for x in product(uniques, uniques)])
            uniques_hash = np.array([hash(tuple(x)) for x in product(uniques, uniques)])
            excludes = np.isin(uniques_hash, bonds_hash)
            exclude_size = np.unique(uniques_product[excludes]).size
            # Exclude condition
            if uniques.shape[0] - excludes.sum() > 2 and exclude_size <= 3:
                continue
            else:
                checked_core_id[i] == -1
        return checked_core_id[np.where(checked_core_id > -1)[0]]

    def get_idihedrals(self, bonds, core_id):
        """get idihedrals"""
        idihedrals = None
        new_id = None
        for i in range(core_id.shape[0]):
            ids_for_idihedral = np.where(np.sum(np.isin(bonds, core_id[i]), axis=1) > 0)[0]
            bonds_for_idihedral = bonds[ids_for_idihedral]
            if bonds_for_idihedral.shape[0] == 3:
                idihedral = np.unique(bonds_for_idihedral.flatten())[None, :]
                if idihedrals is None:
                    idihedrals = idihedral
                    new_id = core_id[i]
                else:
                    idihedrals = np.append(idihedrals, idihedral, axis=0)
                    new_id = np.append(new_id, core_id[i])
            else:
                # Only SP2 is considered.
                continue
        return idihedrals, new_id

    def get_excludes(self, bonds, angles, dihedrals, idihedrals):
        """get excludes"""
        excludes = []
        for i in range(self.atom_nums):
            bond_excludes = bonds[np.where(np.isin(bonds, i).sum(axis=1))[0]].flatten()
            angle_excludes = angles[np.where(np.isin(angles, i).sum(axis=1))[0]].flatten()
            dihedral_excludes = dihedrals[np.where(np.isin(dihedrals, i).sum(axis=1))[0]].flatten()
            if idihedrals is not None:
                idihedral_excludes = idihedrals[np.where(np.isin(idihedrals, i).sum(axis=1))[0]].flatten()
            this_excludes = np.append(bond_excludes, angle_excludes)
            this_excludes = np.append(this_excludes, dihedral_excludes)
            if idihedrals is not None:
                this_excludes = np.append(this_excludes, idihedral_excludes)
            this_excludes = np.unique(this_excludes)
            excludes.append(this_excludes[np.where(this_excludes != i)[0]].tolist())
        padding_length = 0
        for i in range(self.atom_nums):
            padding_length = max(padding_length, len(excludes[i]))
        self.excludes = np.empty((self.atom_nums, padding_length))
        for i in range(self.atom_nums):
            self.excludes[i] = np.pad(np.array(excludes[i]), (0, padding_length - len(excludes[i])), mode='constant',
                                      constant_values=self.atom_nums)
        return self.excludes

    def get_mass(self, atom_names):
        """get mass"""
        for i in range(self.atom_nums):
            self.mass[i] = self.mass_dict[atom_names[i]]

    def get_vdw_params(self, atom_names):
        """
        ['H','HO','HS','HC','H1','H2','H3','HP','HA','H4',
         'H5','HZ','O','O2','OH','OS','OP','C*','CI','C5',
         'C4','CT','CX','C','N','N3','S','SH','P','MG',
         'C0','F','Cl','Br','I','2C','3C','C8','CO']
        """
        for i in range(self.atom_nums):
            this_id = np.where(np.isin(vdw_params['name'], atom_names[i]))[0]
            if atom_names[i] in ['N2', 'NA', 'NB']:
                this_id = [24]
            if atom_names[i] in ['CA', 'CC', 'CR', 'CW', 'CN', 'CB', 'CV']:
                this_id = [17]
            self.vdw_param[i][0] = vdw_params['radius'][this_id]
            self.vdw_param[i][1] = vdw_params['well_depth'][this_id]

    def get_hbonds(self, bonds):
        """get hbonds"""
        hatoms = np.where(np.isin(self.atom_types, self.htypes))[0]
        bonds_with_h = np.where(np.isin(bonds, hatoms).sum(axis=-1))[0]
        non_hbonds = np.where(np.isin(bonds, hatoms).sum(axis=-1) == 0)[0]
        return bonds[bonds_with_h], bonds[non_hbonds]

    def get_nb14_index(self, dihedrals, angles, bonds):
        """get nb14 index"""
        nb14 = dihedrals[:, [0, -1]]
        nb14.sort()
        nb14_index = np.unique(nb14, axis=0)
        nb_hash = []
        for nb in nb14_index:
            if nb[0] < nb[1]:
                nb_hash.append(hash((nb[0], nb[1])))
            else:
                nb_hash.append(hash((nb[1], nb[0])))
        nb_hash = np.array(nb_hash)
        angle_hash = []
        for angle in angles:
            if angle[0] < angle[-1]:
                angle_hash.append(hash((angle[0], angle[-1])))
            else:
                angle_hash.append(hash((angle[-1], angle[0])))
        angle_hash = np.array(angle_hash)
        bond_hash = []
        for bond in bonds:
            b = sorted(bond)
            bond_hash.append(hash(tuple(b)))
        bond_hash = np.array(bond_hash)
        include_index = np.where(np.isin(nb_hash, angle_hash) + np.isin(nb_hash, bond_hash) == 0)[0]
        return nb14_index[include_index]

    def __call__(self, bonds):
        hbonds, non_hbonds = self.get_hbonds(bonds)
        atoms_types = self.atom_types
        self.get_vdw_params(atoms_types)
        atom_types = np.append(atoms_types, self._wildcard)
        bond_params = self.get_bond_params(bonds, atoms_types)
        self.bond_params = np.append(bonds, bond_params, axis=1)
        middle_id = np.where(np.bincount(bonds.flatten()) > 1)[0]
        ids_for_angle = np.where(np.sum(np.isin(bonds, middle_id), axis=1) > 0)[0]
        bonds_for_angle = bonds[ids_for_angle]
        angles = self.combinations(bonds, bonds_for_angle, middle_id)
        angle_params = self.get_angle_params(angles, atoms_types)
        self.angle_params = np.append(angles, angle_params, axis=1)
        dihedral_middle_id = bonds[np.where(np.isin(bonds, middle_id).sum(axis=1) == 2)[0]]
        dihedrals = self.get_dihedrals(angles, dihedral_middle_id)
        self.dihedral_params = self.get_dihedral_params(dihedrals, atom_types)
        core_id = np.where(np.bincount(bonds.flatten()) > 2)[0]
        checked_core_id = self.check_idihedral(bonds, core_id)
        idihedrals, third_id = self.get_idihedrals(bonds, checked_core_id)
        self.improper_dihedral_params = self.get_idihedral_params(idihedrals, atom_types, third_id)
        self.nb14_index = self.get_nb14_index(dihedrals, angles, bonds)
        self.excludes = self.get_excludes(bonds, angles, dihedrals, idihedrals)
        return self.bond_params, \
               self.angle_params, \
               self.dihedral_params, \
               self.improper_dihedral_params, \
               angles, \
               dihedrals, \
               idihedrals, \
               self.excludes, \
               self.vdw_param, \
               hbonds, non_hbonds


class HyperParam(nn.Cell):
    """ Module for get force field parameters.
    Args:
        atom_type(np.array[np.str_]): Types of all atoms. If not given, transform from atom name will be used.
        atom_name(np.array[np.str_]): Names of all atoms. If not given, transform from atomic number will be used.
        atomic_number(np.array): Just give atomic number and no additional information.
        mass(np.array): The atom mass with unit u. If not given, the default mass correspond to atomic_number will be
                        used.
        num_molecule(int): The number of molecules in input atoms, default to be 1.
        mol_name(np.array[np.str_]): All the different molecular names.
        bond_index(np.array): N*2 array contained the connection properties in the given molecule.
        residue_name(np.array[np.str_]): Names of residues/aminos.
        residue_pointer(np.array): The start atom index of correspond residue.
    Parameters:
        bond_index(Tensor): The same with input bond indexes.
        bond_params(Tensor): Bond force parameters from force field according to bond_index.
        angle_index(Tensor): The indexes where 2 different bonds connects the same atom.
        angle_params(Tensor): Angle force parameters from force field according to angle_index.
        dihedral_index(Tensor): The indexes where 2 different bonds with all different atoms are connected by another
                                bond.
        dihedral_params(Tensor): Dihedral force parameters from force field according to dihedral_index.
        idihedral_index(Tensor): The indexes where 3 different bonds are connected to the same atom, constructing a
                                 improper dihedral.
        idihedral_params(Tensor): Improper dihedral force parameters.
        exclude_index(Tensor): Atom indexes included in bond_index, angle_index and all dihedral_indexes, padded.
        vdw_index(Tensor): The 1-4 interactions in dihedrals.
        vdw_params(Tensor): The atomic radius and well depth parameters for each atom, not for vdw indexes.
        hbond_index(Tensor): The bond indexes with H atoms.
        nonhbond_index(Tensor): The bond indexes without H atoms.
    """

    def __init__(self, atom_type=None, bond_index=None, mass=None, charge=None, atom_name=None, atomic_number=None,
                 num_molecule=1, mol_name=None, residue_name=None, residue_pointer=None, template=None,
                 save_template=False):
        super(HyperParam, self).__init__()
        if template is None:
            residue_atom_num = {'VAL': 16, 'GLN': 17, 'PHE': 20, 'ASN': 14, 'PRO': 14, 'THR': 14, 'ALA': 10, 'ILE': 19,
                                'SER': 11, 'LEU': 19, 'TRP': 24, 'GLU': 15, 'LYS': 22, 'CYS': 11, 'TYR': 21, 'HIS': 17,
                                'GLY': 7, 'ACE': 6, 'ASP': 12, 'ARG': 24, 'MET': 17, 'NME': 6}

            self.residue_atom_num = None
            if residue_name is not None:
                if type(residue_name) is list:
                    residue_name = np.array(residue_name, np.str_)
                self.num_residue = residue_name.shape[-1]
                if residue_pointer is not None:
                    assert residue_name.shape == residue_pointer.shape
                    self.residue_index = np.argsort(residue_pointer)
                    residue_name = residue_name[self.residue_index]
                    residue_pointer = residue_pointer[self.residue_index]
                    self.residue_atom_num = [residue_atom_num[name] for name in residue_name]
                else:
                    self.residue_atom_num = [residue_atom_num[name] for name in residue_name]
                    residue_pointer = np.append(np.array([0]), np.cumsum(self.residue_atom_num)[:-1])

            if atom_type is None and residue_name is not None:
                atom_type = []
                for res in residue_name:
                    res_atoms = amino_dict[res].values()
                    atom_type.extend(res_atoms)
                atom_type = np.array(atom_type, np.str_)

            if bond_index is None and residue_name is not None:
                bond_index = []
                for i, res in enumerate(residue_name):
                    res_pairs = bond_pairs[res]
                    bond_index.extend(res_pairs + residue_pointer[i])
                    if i > 0:
                        bond_index.extend([[residue_pointer[i] - 2, residue_pointer[i]]])
                bond_index = np.array(bond_index, np.int32)

            self.bond_index = Tensor(bond_index, ms.int32)

            if atom_type is not None:
                atom_nums = atom_type.shape[-1]
                self.params = Params(atom_type, mass=mass)
            elif atom_name is None and atomic_number is None:
                raise ValueError('atom_type, atom_name and atomic_number can not be None at the same time.')
            elif atom_name is not None:
                atom_nums = atom_name.shape[-1]
            if charge is not None:
                self.charge = Tensor(charge, ms.float32)
            if mol_name is not None:
                assert mol_name.shape[-1] == num_molecule

            bond_params, angle_params, dihedral_params, idihedral_params, angles, dihedrals, idihedrals, \
            excludes, vdw, hbonds, non_hbonds = self.params(bond_index)

            self.bond_params = Tensor(bond_params, ms.float32)
            self.angle_params = Tensor(angle_params, ms.float32)
            self.dihedral_params = Tensor(dihedral_params, ms.float32)
            if idihedral_params is not None:
                self.idihedral_params = Tensor(idihedral_params, ms.float32)
            else:
                self.idihedral_params = None
            self.angle_index = Tensor(angles, ms.int32)
            self.dihedral_index = Tensor(dihedrals, ms.int32)
            if idihedrals is not None:
                self.idihedral_index = Tensor(idihedrals, ms.int32)
            else:
                self.idihedral_index = None
            self.excludes_index = Tensor(excludes, ms.int32)
            self.vdw_params = Tensor(vdw, ms.float32)
            self.vdw_index = Tensor(dihedrals[:, 0::3], ms.int32)
            self.hbond_index = Tensor(hbonds, ms.int32)
            self.nonhbond_index = Tensor(non_hbonds, ms.int32)
            self.atom_type = str_to_tensor(atom_type)
            if save_template:
                parameters = {'atom_type': self.atom_type,
                              'bond_index': self.bond_index,
                              'hbond_index': self.hbond_index,
                              'nonhbond_index': self.nonhbond_index,
                              'vdw_index': self.vdw_index,
                              'vdw_params': self.vdw_params,
                              'angle_index': self.angle_index,
                              'angle_params': self.angle_params,
                              'dihedral_index': self.dihedral_index,
                              'dihedral_params': self.dihedral_params,
                              'idihedral_index': self.idihedral_index,
                              'idihedral_params': self.idihedral_params
                              }
                save_checkpoint(self, 'ckp_hyperparam.ckpt', append_dict=parameters)
        else:
            assert type(template) is str and template.endswith('.ckpt')
            params = load_checkpoint(template)
            for param in params:
                params[param] = Tensor(params[param])
            params['atom_type'] == np.array(tensor_to_str(params['atom_type']).split('_'), np.str_)
            self.__dict__.update(params)


class Molecule(nn.Cell):
    """ Module for get force field parameters.
        Args:
            atom_type(np.array[np.str_]): Types of all atoms. If not given, transform from atom name will be used.
            atom_name(np.array[np.str_]): Names of all atoms. If not given, transform from atomic number will be used.
            atomic_number(np.array): Just give atomic number and no additional information.
            mass(np.array): The atom mass with unit u. If not given, the default mass correspond to atomic_number will
                            be used.
            num_molecule(int): The number of molecules in input atoms, default to be 1.
            mol_name(np.array[np.str_]): All the different molecular names.
            bond_index(np.array): N*2 array contained the connection properties in the given molecule.
        Parameters:
            bond_index(Tensor): The same with input bond indexes.
            bond_params(Tensor): Bond force parameters from force field according to bond_index.
            angle_index(Tensor): The indexes where 2 different bonds connects the same atom.
            angle_params(Tensor): Angle force parameters from force field according to angle_index.
            dihedral_index(Tensor): The indexes where 2 different bonds with all different atoms are connected by
                                     another bond.
            dihedral_params(Tensor): Dihedral force parameters from force field according to dihedral_index.
            idihedral_index(Tensor): The indexes where 3 different bonds are connected to the same atom, constructing a
                                     improper dihedral.
            idihedral_params(Tensor): Improper dihedral force parameters.
            exclude_index(Tensor): Atom indexes included in bond_index, angle_index and all dihedral_indexes, padded.
            vdw_index(Tensor): The 1-4 interactions in dihedrals.
            vdw_params(Tensor): The atomic radius and well depth parameters for each atom, not for vdw indexes.
            hbond_index(Tensor): The bond indexes with H atoms.
            nonhbond_index(Tensor): The bond indexes without H atoms.
        """

    def __init__(self, atom_type=None, bond_index=None, mass=None, charge=None, atom_name=None, atomic_number=None,
                 num_molecule=1, mol_name=None, template=None, save_template=False):
        super(Molecule, self).__init__()
        if template is None:
            self.bond_index = Tensor(bond_index, ms.int32)

            if atom_type is not None:
                atom_nums = atom_type.shape[-1]
                self.params = Params(atom_type, mass=mass, atom_names=atom_name)
            elif atom_name is None and atomic_number is None:
                raise ValueError('atom_type, atom_name and atomic_number can not be None at the same time.')
            elif atom_name is not None:
                atom_nums = atom_name.shape[-1]
            if charge is not None:
                self.charge = Tensor(charge, ms.float32)
            if mol_name is not None:
                assert mol_name.shape[-1] == num_molecule

            bond_params, angle_params, dihedral_params, idihedral_params, angles, dihedrals, \
            idihedrals, excludes, vdw, hbonds, non_hbonds = self.params(bond_index)

            self.mass = Tensor(self.params.mass, ms.float32)
            self.bond_params = Tensor(bond_params, ms.float32)
            self.angle_params = Tensor(angle_params, ms.float32)
            self.dihedral_params = Tensor(dihedral_params, ms.float32)
            if idihedral_params is not None:
                self.idihedral_params = Tensor(idihedral_params, ms.float32)
            else:
                self.idihedral_params = None
            self.angle_index = Tensor(angles, ms.int32)
            self.dihedral_index = Tensor(dihedrals, ms.int32)
            if self.params.nb14_index is not None:
                self.nb14_index = Tensor(self.params.nb14_index, ms.int32)
            else:
                self.nb14_index = None
            if idihedrals is not None:
                self.idihedral_index = Tensor(idihedrals, ms.int32)
            else:
                self.idihedral_index = None
            self.excludes_index = Tensor(excludes, ms.int32)
            self.vdw_params = Tensor(vdw, ms.float32)
            self.vdw_index = Tensor(dihedrals[:, 0::3], ms.int32)
            self.hbond_index = Tensor(hbonds, ms.int32)
            self.nonhbond_index = Tensor(non_hbonds, ms.int32)
            if save_template:
                parameters = {'atom_type': str_to_tensor(atom_type),
                              'bond_index': self.bond_index,
                              'hbond_index': self.hbond_index,
                              'nonhbond_index': self.nonhbond_index,
                              'vdw_index': self.vdw_index,
                              'vdw_params': self.vdw_params,
                              'angle_index': self.angle_index,
                              'angle_params': self.angle_params,
                              'dihedral_index': self.dihedral_index,
                              'dihedral_params': self.dihedral_params,
                              'idihedral_index': self.idihedral_index,
                              'idihedral_params': self.idihedral_params
                              }
                save_checkpoint(self, 'ckp_hyperparam.ckpt', append_dict=parameters)
        else:
            assert type(template) is str and template.endswith('.ckpt')
            params = load_checkpoint(template)
            for param in params:
                params[param] = Tensor(params[param])
            params['atom_type'] == np.array(tensor_to_str(params['atom_type']).split('_'), np.str_)
            self.__dict__.update(params)


class Protein(Molecule):
    """ Module for get force field parameters.
        Args:
            num_molecule(int): The number of molecules in input atoms, default to be 1.
            mol_name(np.array[np.str_]): All the different molecular names.
            residue_name(np.array[np.str_]): Names of residues/aminos.
            residue_pointer(np.array): The start atom index of correspond residue.
        Parameters:
            bond_index(Tensor): The same with input bond indexes.
            bond_params(Tensor): Bond force parameters from force field according to bond_index.
            angle_index(Tensor): The indexes where 2 different bonds connects the same atom.
            angle_params(Tensor): Angle force parameters from force field according to angle_index.
            dihedral_index(Tensor): The indexes where 2 different bonds with all different atoms are connected by
                                    another bond.
            dihedral_params(Tensor): Dihedral force parameters from force field according to dihedral_index.
            idihedral_index(Tensor): The indexes where 3 different bonds are connected to the same atom, constructing a
                                     improper dihedral.
            idihedral_params(Tensor): Improper dihedral force parameters.
            exclude_index(Tensor): Atom indexes included in bond_index, angle_index and all dihedral_indexes, padded.
            vdw_index(Tensor): The 1-4 interactions in dihedrals.
            vdw_params(Tensor): The atomic radius and well depth parameters for each atom, not for vdw indexes.
            hbond_index(Tensor): The bond indexes with H atoms.
            nonhbond_index(Tensor): The bond indexes without H atoms.
        """

    def __init__(self,
                 num_molecule=1, mol_name=None, residue_name=None, residue_pointer=None, template=None,
                 save_template=False):
        if template is None:
            residue_atom_num = {'VAL': 16, 'GLN': 17, 'PHE': 20, 'ASN': 14, 'PRO': 14, 'THR': 14, 'ALA': 10, 'ILE': 19,
                                'SER': 11, 'LEU': 19, 'TRP': 24, 'GLU': 15, 'LYS': 22, 'CYS': 11, 'TYR': 21, 'HIS': 17,
                                'GLY': 7, 'ACE': 6, 'ASP': 12, 'ARG': 24, 'MET': 17, 'NME': 6}

            self.residue_atom_num = None
            if residue_name is not None:
                if type(residue_name) is list:
                    residue_name = np.array(residue_name, np.str_)
                self.num_residue = residue_name.shape[-1]
                if residue_pointer is not None:
                    assert residue_name.shape == residue_pointer.shape
                    self.residue_index = np.argsort(residue_pointer)
                    residue_name = residue_name[self.residue_index]
                    residue_pointer = residue_pointer[self.residue_index]
                    self.residue_atom_num = [residue_atom_num[name] for name in residue_name]
                else:
                    self.residue_atom_num = [residue_atom_num[name] for name in residue_name]
                    residue_pointer = np.append(np.array([0]), np.cumsum(self.residue_atom_num)[:-1])

            atom_type = []
            for res in residue_name:
                res_atoms = amino_dict[res].values()
                atom_type.extend(res_atoms)
            self.atom_type = np.array(atom_type, np.str_)

            bond_index = []
            for i, res in enumerate(residue_name):
                res_pairs = bond_pairs[res]
                bond_index.extend(res_pairs + residue_pointer[i])
                if i > 0:
                    bond_index.extend([[residue_pointer[i] - 2, residue_pointer[i]]])
            bond_index = np.array(bond_index, np.int32)

            super(Protein, self).__init__(atom_type=self.atom_type, bond_index=bond_index)
            self.atom_type = str_to_tensor(self.atom_type)
            if save_template:
                parameters = {'atom_type': self.atom_type,
                              'bond_index': self.bond_index,
                              'hbond_index': self.hbond_index,
                              'nonhbond_index': self.nonhbond_index,
                              'vdw_index': self.vdw_index,
                              'vdw_params': self.vdw_params,
                              'angle_index': self.angle_index,
                              'angle_params': self.angle_params,
                              'dihedral_index': self.dihedral_index,
                              'dihedral_params': self.dihedral_params,
                              'idihedral_index': self.idihedral_index,
                              'idihedral_params': self.idihedral_params
                              }
                save_checkpoint(self, 'ckp_hyperparam.ckpt', append_dict=parameters)
        else:
            super(Protein, self).__init__(template=template)
            assert type(template) is str and template.endswith('.ckpt')
            params = load_checkpoint(template)
            for param in params:
                params[param] = Tensor(params[param])
            params['atom_type'] == np.array(tensor_to_str(params['atom_type']).split('_'), np.str_)
            self.__dict__.update(params)


class ReconstructProtein(Molecule):
    """Reconstruct a protein from given parameters.
    Args:
        res_names(list): The series of residue names.
        res_pointers(numpy.int32): The pointer as start atom index of each residue.
        atom_names(numpy.str_): An array of atom names.
        init_res_names(list): The residue name of each given atom.
        init_res_ids(list): The residue id of each given atom.
    """

    def __init__(self,
                 res_names: list,
                 res_pointers: np.ndarray,
                 atom_names: np.ndarray,
                 init_res_names: list = None,
                 init_res_ids: list = None):
        assert len(res_names) == res_pointers.shape[0]
        if res_names[0] != 'ACE' and res_names[0] != 'NME':
            res_names[0] = 'N' + res_names[0]
        if res_names[-1] != 'ACE' and res_names[-1] != 'NME':
            res_names[-1] = 'C' + res_names[-1]
        self.res_nums = res_pointers.shape[0]
        self.res_names = res_names
        self.res_pointers = np.append(res_pointers, len(atom_names))
        self.res_id = np.array([j for i in range(1, len(res_pointers)) for j in [i - 1] * res_pointers[i]], np.int32)
        self.atom_names = atom_names
        self.backbone_mask = np.isin(self.atom_names, backbone_atoms)
        self.oxt_id = np.where(np.isin(self.atom_names, include_backbone_atoms))[0][-1]
        self.backbone_mask[self.oxt_id] = True
        self.bond_index = self.get_bonds
        self.atom_types = self.get_types
        self.atomic_numbers = self.get_atomic_number
        self.charges = self.get_charge
        self.crd_mapping_ids = self.get_crd_mask_ids
        if init_res_names is not None:
            self.init_res_names = np.array(init_res_names, np.str_)
        if init_res_ids is not None:
            self.init_res_ids = np.array(init_res_ids, np.int32)
        super(ReconstructProtein, self).__init__(self.atom_types, self.bond_index, atom_name=self.atom_names,
                                                 charge=self.charges)

    @property
    def get_bonds(self):
        """get bonds"""
        bond_index = None
        c_list = []
        n_list = []
        for i in range(self.res_nums):
            atoms = self.atom_names[self.res_pointers[i]:self.res_pointers[i + 1]]
            if i != self.res_nums - 1:
                c_list.append(np.where(atoms == 'C')[0][0] + self.res_pointers[i])
            if i != 0:
                n_list.append(np.where(atoms == 'N')[0][0] + self.res_pointers[i])
            res_name = self.res_names[i]
            if 'HIE' in res_name:
                res_name = res_name.replace('HIE', 'HIS')
            mapping_index = np.where(atoms == np.array(list(amino_dict[res_name].keys()))[:, None])[-1]
            bonds = mapping_index[np.array(list(bond_pairs[res_name]))]
            if bond_index is None:
                bond_index = bonds.astype(np.int32)
            else:
                bond_index = np.append(bond_index, np.sort(bonds) + self.res_pointers[i], axis=0)
        bond_index = np.append(bond_index, np.array([c_list, n_list]).T, axis=0)
        return bond_index

    @property
    def get_types(self):
        """get types"""
        atom_types = None
        for i in range(self.res_nums):
            atoms = self.atom_names[self.res_pointers[i]:self.res_pointers[i + 1]]
            res_name = self.res_names[i]
            if 'HIE' in res_name:
                res_name = res_name.replace('HIE', 'HIS')
            mapping_index = np.where(np.array(list(amino_dict[res_name].keys())) == atoms[:, None])[-1]
            atom_type = np.array(list(amino_dict[res_name].values()))[mapping_index]
            if atom_types is None:
                atom_types = atom_type
            else:
                atom_types = np.append(atom_types, atom_type, axis=0)
        return atom_types

    @property
    def get_atomic_number(self):
        """get atomic number"""
        atomic_numbers = []
        for atom in self.atom_names:
            if atom.startswith('C'):
                atomic_numbers.append(6)
            elif atom.startswith('O'):
                atomic_numbers.append(8)
            elif atom.startswith('H'):
                atomic_numbers.append(1)
            elif atom.startswith('N'):
                atomic_numbers.append(7)
            elif atom.startswith('S'):
                atomic_numbers.append(16)
            elif atom == '1C' or atom == '2C' or atom == '3C' or atom == '4C':
                atomic_numbers.append(6)
            else:
                print('Not recgonized atom.')
        return np.array(atomic_numbers, np.int32)

    @property
    def get_charge(self):
        """get charge"""
        charges = None
        for i in range(self.res_nums):
            atoms = self.atom_names[self.res_pointers[i]:self.res_pointers[i + 1]]
            res_name = self.res_names[i]
            if 'HIE' in res_name:
                res_name = res_name.replace('HIE', 'HIS')
            mapping_index = np.where(np.array(list(amino_dict[res_name].keys())) == atoms[:, None])[-1]
            charge = np.array(charge_dict[res_name])[mapping_index]
            if charges is None:
                charges = charge
            else:
                charges = np.append(charges, charge, axis=0)
        return charges

    @property
    def get_crd_mask_ids(self):
        """get crd mask ids"""
        ids = None
        for i in range(self.res_nums):
            atoms = self.atom_names[self.res_pointers[i]:self.res_pointers[i + 1]]
            res_name = self.res_names[i]
            if len(res_name) == 4:
                res_name = res_name[1:]
            if 'HIE' in res_name:
                res_name = res_name.replace('HIE', 'HIS')
            mapping_index = np.where(np.array(restype_name_to_atom14_names[res_name]) == atoms[:, None])[-1]
            if ids is None:
                ids = mapping_index
            else:
                ids = np.append(ids, mapping_index, axis=0)
        return ids
