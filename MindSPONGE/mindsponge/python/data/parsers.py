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
Read information from a pdb format file.
"""
import re
import string
import collections
import io
import dataclasses
from typing import Any, Mapping, Optional, Sequence, Tuple, List
import h5py
import numpy as np

from absl import logging
from Bio import PDB
from Bio.Data import SCOPData


from .add_h import addH

restypes = [
    'A', 'R', 'N', 'D', 'C',
    'Q', 'E', 'G', 'H', 'I',
    'L', 'K', 'M', 'F', 'P',
    'S', 'T', 'W', 'Y', 'V'
]
resdict = {'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
           'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
           'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
           'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19,
           'CALA': 0, 'CARG': 1, 'CASN': 2, 'CASP': 3, 'CCYS': 4,
           'CGLN': 5, 'CGLU': 6, 'CGLY': 7, 'CHIS': 8, 'CILE': 9,
           'CLEU': 10, 'CLYS': 11, 'CMET': 12, 'CPHE': 13, 'CPRO': 14,
           'CSER': 15, 'CTHR': 16, 'CTRP': 17, 'CTYR': 18, 'CVAL': 19,
           'NALA': 0, 'NARG': 1, 'NASN': 2, 'NASP': 3, 'NCYS': 4,
           'NGLN': 5, 'NGLU': 6, 'NGLY': 7, 'NHIS': 8, 'NILE': 9,
           'NLEU': 10, 'NLYS': 11, 'NMET': 12, 'NPHE': 13, 'NPRO': 14,
           'NSER': 15, 'NTHR': 16, 'NTRP': 17, 'NTYR': 18, 'NVAL': 19,
           'CHIE': 8, 'HIE': 8, 'NHIE': 8
           }

atom_types = [
    'N', 'CA', 'C', 'CB', 'O', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD',
    'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3',
    'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2',
    'CZ3', 'NZ', 'OXT'
]
atom_order = {atom_type: i for i, atom_type in enumerate(atom_types)}

restype_name_to_atom14_names = {
    'ALA': ['N', 'CA', 'C', 'O', 'CB', '', '', '', '', '', '', '', '', ''],
    'ARG': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2', '', '', ''],
    'ASN': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'ND2', '', '', '', '', '', ''],
    'ASP': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2', '', '', '', '', '', ''],
    'CYS': ['N', 'CA', 'C', 'O', 'CB', 'SG', '', '', '', '', '', '', '', ''],
    'GLN': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'NE2', '', '', '', '', ''],
    'GLU': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2', '', '', '', '', ''],
    'GLY': ['N', 'CA', 'C', 'O', '', '', '', '', '', '', '', '', '', ''],
    'HIS': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2', '', '', '', ''],
    'ILE': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1', '', '', '', '', '', ''],
    'LEU': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', '', '', '', '', '', ''],
    'LYS': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ', '', '', '', '', ''],
    'MET': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE', '', '', '', '', '', ''],
    'PHE': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', '', '', ''],
    'PRO': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', '', '', '', '', '', '', ''],
    'SER': ['N', 'CA', 'C', 'O', 'CB', 'OG', '', '', '', '', '', '', '', ''],
    'THR': ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2', '', '', '', '', '', '', ''],
    'TRP': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
    'TYR': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH', '', ''],
    'VAL': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', '', '', '', '', '', '', ''],
    'UNK': ['', '', '', '', '', '', '', '', '', '', '', '', '', ''],
}
restype_name_to_atom14_masks = {
    'ALA': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'ARG': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    'ASN': [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    'ASP': [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    'CYS': [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    'GLN': [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    'GLU': [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    'GLY': [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'HIS': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    'HIE': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    'ILE': [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    'LEU': [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    'LYS': [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    'MET': [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    'PHE': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    'PRO': [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    'SER': [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    'THR': [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    'TRP': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    'TYR': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    'VAL': [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    'UNK': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}

atom14_order_dict = {'ALA': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4},
                     'ARG': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG': 5, 'CD': 6, 'NE': 7, 'CZ': 8, 'NH1': 9,
                             'NH2': 10},
                     'ASN': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG': 5, 'OD1': 6, 'ND2': 7},
                     'ASP': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG': 5, 'OD1': 6, 'OD2': 7},
                     'CYS': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'SG': 5},
                     'GLN': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG': 5, 'CD': 6, 'OE1': 7, 'NE2': 8},
                     'GLU': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG': 5, 'CD': 6, 'OE1': 7, 'OE2': 8},
                     'GLY': {'N': 0, 'CA': 1, 'C': 2, 'O': 3},
                     'HIS': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG': 5, 'ND1': 6, 'CD2': 7, 'CE1': 8, 'NE2': 9},
                     'HIE': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG': 5, 'ND1': 6, 'CD2': 7, 'CE1': 8, 'NE2': 9},
                     'ILE': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG1': 5, 'CG2': 6, 'CD1': 7},
                     'LEU': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG': 5, 'CD1': 6, 'CD2': 7},
                     'LYS': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG': 5, 'CD': 6, 'CE': 7, 'NZ': 8},
                     'MET': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG': 5, 'SD': 6, 'CE': 7},
                     'PHE': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG': 5, 'CD1': 6, 'CD2': 7, 'CE1': 8, 'CE2': 9,
                             'CZ': 10},
                     'PRO': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG': 5, 'CD': 6},
                     'SER': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'OG': 5},
                     'THR': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'OG1': 5, 'CG2': 6},
                     'TRP': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG': 5, 'CD1': 6, 'CD2': 7, 'NE1': 8, 'CE2': 9,
                             'CE3': 10, 'CZ2': 11, 'CZ3': 12, 'CH2': 13},
                     'TYR': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG': 5, 'CD1': 6, 'CD2': 7, 'CE1': 8, 'CE2': 9,
                             'CZ': 10, 'OH': 11},
                     'VAL': {'N': 0, 'CA': 1, 'C': 2, 'O': 3, 'CB': 4, 'CG1': 5, 'CG2': 6},
                     'UNK': {}}

atom14_to_atom37_dict = {'ALA': [0, 1, 2, 4, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         'ARG': [0, 1, 2, 4, 3, 5, 11, 23, 32, 29, 30, 0, 0, 0],
                         'ASN': [0, 1, 2, 4, 3, 5, 16, 15, 0, 0, 0, 0, 0, 0],
                         'ASP': [0, 1, 2, 4, 3, 5, 16, 17, 0, 0, 0, 0, 0, 0],
                         'CYS': [0, 1, 2, 4, 3, 10, 0, 0, 0, 0, 0, 0, 0, 0],
                         'GLN': [0, 1, 2, 4, 3, 5, 11, 26, 25, 0, 0, 0, 0, 0],
                         'GLU': [0, 1, 2, 4, 3, 5, 11, 26, 27, 0, 0, 0, 0, 0],
                         'GLY': [0, 1, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         'HIS': [0, 1, 2, 4, 3, 5, 14, 13, 20, 25, 0, 0, 0, 0],
                         'HIE': [0, 1, 2, 4, 3, 5, 14, 13, 20, 25, 0, 0, 0, 0],
                         'ILE': [0, 1, 2, 4, 3, 6, 7, 12, 0, 0, 0, 0, 0, 0],
                         'LEU': [0, 1, 2, 4, 3, 5, 12, 13, 0, 0, 0, 0, 0, 0],
                         'LYS': [0, 1, 2, 4, 3, 5, 11, 19, 35, 0, 0, 0, 0, 0],
                         'MET': [0, 1, 2, 4, 3, 5, 18, 19, 0, 0, 0, 0, 0, 0],
                         'PHE': [0, 1, 2, 4, 3, 5, 12, 13, 20, 21, 32, 0, 0, 0],
                         'PRO': [0, 1, 2, 4, 3, 5, 11, 0, 0, 0, 0, 0, 0, 0],
                         'SER': [0, 1, 2, 4, 3, 8, 0, 0, 0, 0, 0, 0, 0, 0],
                         'THR': [0, 1, 2, 4, 3, 9, 7, 0, 0, 0, 0, 0, 0, 0],
                         'TRP': [0, 1, 2, 4, 3, 5, 12, 13, 24, 21, 22, 33, 34, 28],
                         'TYR': [0, 1, 2, 4, 3, 5, 12, 13, 20, 21, 32, 31, 0, 0],
                         'VAL': [0, 1, 2, 4, 3, 6, 7, 0, 0, 0, 0, 0, 0, 0],
                         'UNK': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}


def read_pdb_via_xponge(pdb_name, addh):
    """read pdb via xponge"""
    if addh:
        import Xponge
        import Xponge.forcefield.AMBER.ff14SB
        from Xponge.LOAD import pdb as loadpdb
        from Xponge.BUILD import Save_PDB
        t = loadpdb(pdb_name, ignoreH=True)
        t.Add_Missing_Atoms()
        t_name = pdb_name.replace('.pdb', '_addH_by_xponge.pdb')
        Save_PDB(t, t_name)
        return read_pdb(t_name, add_h=False)
    return read_pdb(pdb_name, add_h=False)


def read_pdb(pdb_name, add_h=True):
    """Read a pdb file and return atom information with numpy array format.
    Args:
        pdb_name(str): The pdb file name, absolute path is suggested.
        add_h(bool): weather add H atom or not
    Returns:
        atom_names(list): 1-dimension list contain all atom names in each residue.
        res_names(list): 1-dimension list of all residue names.
        res_ids(numpy.int32): Unique id for each residue names.
        crds(list): The list format of coordinates.
        res_pointer(numpy.int32): The pointer where the residue starts.
        flatten_atoms(numpy.str_): The flatten atom names.
        flatten_crds(numpy.float32): The numpy array format of coordinates.
        init_res_names(list): The residue name information of each atom.
        init_res_ids(list): The residue id of each atom.
    """
    if add_h:
        new_name = pdb_name.replace('.pdb', '_addH_by_mindponge.pdb')
        addH(pdb_name, new_name)
        pdb_name = new_name
    with open(pdb_name, 'r') as pdb:
        lines = pdb.readlines()[1:]
    atom_names = []
    atom_group = []
    res_names = []
    res_ids = []
    init_res_names = []
    init_res_ids = []
    crds = []
    crd_group = []
    res_pointer = []
    flatten_atoms = []
    flatten_crds = []
    atom14_positions = []
    atom14_atom_exists = []
    residx_atom14_to_atom37 = []
    atom_pos = []
    res_name = ''
    for index, line in enumerate(lines):
        if 'END' in line or 'TER' in line:
            atom_names.append(atom_group)
            crds.append(crd_group)
            atom14_positions.append(atom_pos)
            residx_atom14_to_atom37.append(atom14_to_atom37_dict.get(res_name))
            break
        atom_name = line[12:16].strip()
        res_name = line[17:20].strip()
        res_id = int(line[22:26].strip())
        crd = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        pointer = int(line[6:11].strip()) - 1
        flatten_atoms.append(atom_name)
        flatten_crds.append(crd)
        init_res_names.append(res_name)
        init_res_ids.append(res_id)
        if res_ids == []:
            res_ids.append(res_id)
            res_names.append(res_name)
            atom14_atom_exists.append(restype_name_to_atom14_masks.get(res_name))
            atom_group.append(atom_name)
            crd_group.append(crd)
            res_pointer.append(0)
            atom_pos = np.zeros((14, 3))
            if not atom_name.startswith('H') and atom_name != 'OXT':
                atom14_order_name = atom14_order_dict.get(res_name)
                a_name = atom14_order_name.get(atom_name)
                atom_pos[a_name] = np.array(crd)
        elif res_id != res_ids[-1]:
            atom14_positions.append(atom_pos)
            residx_atom14_to_atom37.append(atom14_to_atom37_dict.get(res_name))
            atom_pos = np.zeros((14, 3))
            if not atom_name.startswith('H') and atom_name != 'OXT':
                atom14_order_name = atom14_order_dict.get(res_name)
                a_name = atom14_order_name.get(atom_name)
                atom_pos[a_name] = np.array(crd)
            atom_names.append(atom_group)
            crds.append(crd_group)
            atom_group = []
            crd_group = []
            res_ids.append(res_id)
            res_names.append(res_name)
            atom14_atom_exists.append(restype_name_to_atom14_masks.get(res_name))
            atom_group.append(atom_name)
            crd_group.append(crd)
            res_pointer.append(pointer)
        else:
            atom_group.append(atom_name)
            crd_group.append(crd)
            if not atom_name.startswith('H') and atom_name != 'OXT':
                atom14_order_name = atom14_order_dict.get(res_name)
                a_name = atom14_order_name.get(atom_name)
                atom_pos[a_name] = np.array(crd)
        if index == len(lines) - 1:
            atom_names.append(atom_group)
            crds.append(crd_group)
            atom14_positions.append(atom_pos)
            residx_atom14_to_atom37.append(atom14_to_atom37_dict.get(res_name))
    res_ids = np.array(res_ids, np.int32)
    flatten_atoms = np.array(flatten_atoms, np.str_)
    flatten_crds = np.array(flatten_crds, np.float32)
    init_res_names = np.array(init_res_names)
    init_res_ids = np.array(init_res_ids, np.int32)
    res_pointer = np.array(res_pointer, np.int32)
    # Violation loss parameters
    residue_index = np.arange(res_pointer.shape[0])
    aatype = np.zeros_like(residue_index)
    for i in range(res_pointer.shape[0]):
        aatype[i] = resdict.get(res_names[i])
    atom14_atom_exists = np.array(atom14_atom_exists, np.float32)

    atom14_positions = np.array(atom14_positions, np.float32)
    residx_atom14_to_atom37 = np.array(residx_atom14_to_atom37, np.float32)
    result_pack = (atom_names, res_names, res_ids, crds, res_pointer, flatten_atoms, flatten_crds, init_res_names,
                   init_res_ids, residue_index, aatype, atom14_positions, atom14_atom_exists, residx_atom14_to_atom37)
    return result_pack


@dataclasses.dataclass(frozen=True)
class HhrHit:
    """Class representing a hit in an hhr file."""
    index: int
    name: str
    prob_true: float
    e_value: float
    score: float
    aligned_cols: int
    identity: float
    similarity: float
    sum_probs: float
    neff: float
    query: str
    hit_sequence: str
    hit_dssp: str
    column_score_code: str
    confidence_scores: str
    indices_query: List[int]
    indices_hit: List[int]


# Type aliases:
ChainId = str
PdbHeader = Mapping[str, Any]
PDBSTRUCTURE = PDB.Structure.Structure
SeqRes = str
MmCIFDict = Mapping[str, Sequence[str]]


@dataclasses.dataclass(frozen=True)
class Monomer:
    id: str
    num: int


# Note - mmCIF format provides no guarantees on the type of author-assigned
# sequence numbers. They need not be integers.
@dataclasses.dataclass(frozen=True)
class AtomSite:
    residue_name: str
    author_chain_id: str
    mmcif_chain_id: str
    author_seq_num: str
    mmcif_seq_num: int
    insertion_code: str
    hetatm_atom: str
    model_num: int


# Used to map SEQRES index to a residue in the structure.
@dataclasses.dataclass(frozen=True)
class ResiduePosition:
    chain_id: str
    residue_number: int
    insertion_code: str


@dataclasses.dataclass(frozen=True)
class ResidueAtPosition:
    position: Optional[ResiduePosition]
    name: str
    is_missing: bool
    hetflag: str


@dataclasses.dataclass(frozen=True)
class MmcifObject:
    """Representation of a parsed mmCIF file.

    Contains:
      file_id: A meaningful name, e.g. a pdb_id. Should be unique amongst all
        files being processed.
      header: Biopython header.
      structure: Biopython structure.
      chain_to_seqres: Dict mapping chain_id to 1 letter amino acid sequence. E.g.
        {'A': 'ABCDEFG'}
      seqres_to_structure: Dict; for each chain_id contains a mapping between
        SEQRES index and a ResidueAtPosition. e.g. {'A': {0: ResidueAtPosition,
                                                          1: ResidueAtPosition,
                                                          ...}}
      raw_string: The raw string used to construct the MmcifObject.
    """
    file_id: str
    header: PdbHeader
    structure: PDBSTRUCTURE
    chain_to_seqres: Mapping[ChainId, SeqRes]
    seqres_to_structure: Mapping[ChainId, Mapping[int, ResidueAtPosition]]
    raw_string: Any


@dataclasses.dataclass(frozen=True)
class ParsingResult:
    """Returned by the parse function.

    Contains:
      mmcif_object: A MmcifObject, may be None if no chain could be successfully
        parsed.
      errors: A dict mapping (file_id, chain_id) to any exception generated.
    """
    mmcif_object: Optional[MmcifObject]
    errors: Mapping[Tuple[str, str], Any]


def _update_hhr_residue_indices_list(
        sequence, start_index, indices_list):
    """Computes the relative indices for each residue with respect to the original sequence."""
    counter = start_index
    for symbol in sequence:
        if symbol == '-':
            indices_list.append(-1)
        else:
            indices_list.append(counter)
            counter += 1


def _get_hhr_line_regex_groups(
        regex_pattern: str, line: str):
    match = re.match(regex_pattern, line)
    if match is None:
        raise RuntimeError(f'Could not parse query line {line}')
    return match.groups()


def parse_fasta(fasta_string: str):
    """Parses FASTA string and returns list of strings with amino-acid sequences.

    Arguments:
      fasta_string: The string contents of a FASTA file.

    Returns:
      A tuple of two lists:
      * A list of sequences.
      * A list of sequence descriptions taken from the comment lines. In the
        same order as the sequences.
    """
    sequences = []
    descriptions = []
    index = -1
    for line in fasta_string.splitlines():
        line = line.strip()
        if line.startswith('>'):
            index += 1
            descriptions.append(line[1:])  # Remove the '>' at the beginning.
            sequences.append('')
            continue
        elif not line:
            continue  # Skip blank lines.
        sequences[index] += line

    return sequences, descriptions


def _parse_hhr_hit(detailed_lines):
    """Parses the detailed HMM HMM comparison section for a single Hit.

    This works on .hhr files generated from both HHBlits and HHSearch.

    Args:
      detailed_lines: A list of lines from a single comparison section between 2
        sequences (which each have their own HMM's)

    Returns:
      A dictionary with the information from that detailed comparison section

    Raises:
      RuntimeError: If a certain line cannot be processed
    """
    # Parse first 2 lines.
    number_of_hit = int(detailed_lines[0].split()[-1])
    name_hit = detailed_lines[1][1:]

    # Parse the summary line.
    pattern = (
        'Probab=(.*)[\t ]*E-value=(.*)[\t ]*Score=(.*)[\t ]*Aligned_cols=(.*)[\t'
        ' ]*Identities=(.*)%[\t ]*Similarity=(.*)[\t ]*Sum_probs=(.*)[\t '
        ']*Template_Neff=(.*)')
    match = re.match(pattern, detailed_lines[2])
    if match is None:
        raise RuntimeError(
            'Could not parse section: %s. Expected this: \n%s to contain summary.' %
            (detailed_lines, detailed_lines[2]))
    (prob_true, e_value, score, aligned_cols, identity, similarity, sum_probs,
     neff) = [float(x) for x in match.groups()]

    # The next section reads the detailed comparisons. These are in a 'human
    # readable' format which has a fixed length. The strategy employed is to
    # assume that each block starts with the query sequence line, and to parse
    # that with a regexp in order to deduce the fixed length used for that
    # block.
    query = ''
    hit_sequence = ''
    hit_dssp = ''
    column_score_code = ''
    confidence_scores = ''
    indices_query = []
    indices_hit = []
    length_block = None

    for line in detailed_lines[3:]:
        # Parse the query sequence line
        if (line.startswith('Q ') and not line.startswith('Q ss_dssp') and not line.startswith('Q ss_pred') \
                and not line.startswith('Q Consensus')):
            # Thus the first 17 characters must be 'Q <query_name> ', and we can parse
            # everything after that.
            # start    sequence       end       total_sequence_length
            patt = r'[\t ]*([0-9]*) ([A-Z-]*)[\t ]*([0-9]*) \([0-9]*\)'
            groups = _get_hhr_line_regex_groups(patt, line[17:])

            # Get the length of the parsed block using the start and finish indices,
            # and ensure it is the same as the actual block length.
            start = int(groups[0]) - 1  # Make index zero based.
            delta_query = groups[1]
            end = int(groups[2])
            num_insertions = len([x for x in delta_query if x == '-'])
            length_block = end - start + num_insertions
            assert length_block == len(delta_query)

            # Update the query sequence and indices list.
            query += delta_query
            _update_hhr_residue_indices_list(delta_query, start, indices_query)

        elif line.startswith('T '):
            # Parse the hit dssp line.
            if line.startswith('T ss_dssp'):
                #        T ss_dssp      hit_dssp
                patt = r'T ss_dssp[\t ]*([A-Z-]*)'
                groups = _get_hhr_line_regex_groups(patt, line)
                assert len(groups[0]) == length_block
                hit_dssp += groups[0]

            # Parse the hit sequence.
            elif (not line.startswith('T ss_pred') and
                  not line.startswith('T Consensus')):
                # Thus the first 17 characters must be 'T <hit_name> ', and we can
                # parse everything after that.
                # start    sequence       end     total_sequence_length
                patt = r'[\t ]*([0-9]*) ([A-Z-]*)[\t ]*[0-9]* \([0-9]*\)'
                groups = _get_hhr_line_regex_groups(patt, line[17:])
                start = int(groups[0]) - 1  # Make index zero based.
                delta_hit_sequence = groups[1]
                assert length_block == len(delta_hit_sequence)

                # Update the hit sequence and indices list.
                hit_sequence += delta_hit_sequence
                _update_hhr_residue_indices_list(
                    delta_hit_sequence, start, indices_hit)

        # Parse the column score line.
        elif line.startswith(' ' * 22):
            assert length_block
            column_score_code += line[22:length_block + 22]

        # Update confidence score.
        elif line.startswith('Confidence'):
            assert length_block
            confidence_scores += line[22:length_block + 22]

    return HhrHit(
        index=number_of_hit,
        name=name_hit,
        prob_true=prob_true,
        e_value=e_value,
        score=score,
        aligned_cols=int(aligned_cols),
        identity=identity,
        similarity=similarity,
        sum_probs=sum_probs,
        neff=neff,
        query=query,
        hit_sequence=hit_sequence,
        hit_dssp=hit_dssp,
        column_score_code=column_score_code,
        confidence_scores=confidence_scores,
        indices_query=indices_query,
        indices_hit=indices_hit,
    )


def parse_hhr(hhr_string: str):
    """Parses the content of an entire HHR file."""
    lines = hhr_string.splitlines()

    # Each .hhr file starts with a results table, then has a sequence of hit
    # "paragraphs", each paragraph starting with a line 'No <hit number>'. We
    # iterate through each paragraph to parse each hit.

    block_starts = [i for i, line in enumerate(lines) if line.startswith('No ')]

    hits = []
    if block_starts:
        block_starts.append(len(lines))  # Add the end of the final block.
        for i in range(len(block_starts) - 1):
            hits.append(_parse_hhr_hit(lines[block_starts[i]:block_starts[i + 1]]))
    return hits


def parse_a3m(a3m_string: str):
    """Parses sequences and deletion matrix from a3m format alignment.

    Args:
      a3m_string: The string contents of a a3m file. The first sequence in the
        file should be the query sequence.

    Returns:
      A tuple of:
        * A list of sequences that have been aligned to the query. These
          might contain duplicates.
        * The deletion matrix for the alignment as a list of lists. The element
          at `deletion_matrix[i][j]` is the number of residues deleted from
          the aligned sequence i at residue position j.
    """
    sequences, _ = parse_fasta(a3m_string)
    deletion_matrix = []
    for msa_sequence in sequences:
        deletion_vec = []
        deletion_count = 0
        for j in msa_sequence:
            if j.islower():
                deletion_count += 1
            else:
                deletion_vec.append(deletion_count)
                deletion_count = 0
        deletion_matrix.append(deletion_vec)

    # Make the MSA matrix out of aligned (deletion-free) sequences.
    deletion_table = str.maketrans('', '', string.ascii_lowercase)
    aligned_sequences = [s.translate(deletion_table) for s in sequences]
    return aligned_sequences, deletion_matrix


class ParseError(Exception):
    """An error indicating that an mmCIF file could not be parsed."""


def mmcif_loop_to_list(prefix, parsed_info):
    """Extracts loop associated with a prefix from mmCIF data as a list.

    Reference for loop_ in mmCIF:
      http://mmcif.wwpdb.org/docs/tutorials/mechanics/pdbx-mmcif-syntax.html

    Args:
      prefix: Prefix shared by each of the data items in the loop.
        e.g. '_entity_poly_seq.', where the data items are _entity_poly_seq.num,
        _entity_poly_seq.mon_id. Should include the trailing period.
      parsed_info: A dict of parsed mmCIF data, e.g. _mmcif_dict from a Biopython
        parser.

    Returns:
      Returns a list of dicts; each dict represents 1 entry from an mmCIF loop.
    """
    cols = []
    data = []
    for key, value in parsed_info.items():
        if key.startswith(prefix):
            cols.append(key)
            data.append(value)

    assert all([len(xs) == len(data[0]) for xs in data]), ('mmCIF error: Not all loops are the same length: %s' % cols)

    return [dict(zip(cols, xs)) for xs in zip(*data)]


def mmcif_loop_to_dict(prefix, index, parsed_info):
    """Extracts loop associated with a prefix from mmCIF data as a dictionary.

    Args:
      prefix: Prefix shared by each of the data items in the loop.
        e.g. '_entity_poly_seq.', where the data items are _entity_poly_seq.num,
        _entity_poly_seq.mon_id. Should include the trailing period.
      index: Which item of loop data should serve as the key.
      parsed_info: A dict of parsed mmCIF data, e.g. _mmcif_dict from a Biopython
        parser.

    Returns:
      Returns a dict of dicts; each dict represents 1 entry from an mmCIF loop,
      indexed by the index column.
    """
    entries = mmcif_loop_to_list(prefix, parsed_info)
    return {entry[index]: entry for entry in entries}


def parse_mmcif(*,
                file_id: str,
                mmcif_string: str,
                catch_all_errors: bool = True):
    """Entry point, parses an mmcif_string.

    Args:
      file_id: A string identifier for this file. Should be unique within the
        collection of files being processed.
      mmcif_string: Contents of an mmCIF file.
      catch_all_errors: If True, all exceptions are caught and error messages are
        returned as part of the ParsingResult. If False exceptions will be allowed
        to propagate.

    Returns:
      A ParsingResult.
    """
    errors = {}
    try:
        parser = PDB.MMCIFParser(QUIET=True)
        handle = io.StringIO(mmcif_string)
        full_structure = parser.get_structure('', handle)
        first_model_structure = _get_first_model(full_structure)
        # Extract the _mmcif_dict from the parser, which contains useful fields not
        # reflected in the Biopython structure.
        parsed_info = parser._mmcif_dict  # pylint:disable=protected-access

        # Ensure all values are lists, even if singletons.
        for key, value in parsed_info.items():
            if not isinstance(value, list):
                parsed_info[key] = [value]

        header = _get_header(parsed_info)

        # Determine the protein chains, and their start numbers according to the
        # internal mmCIF numbering scheme (likely but not guaranteed to be 1).
        valid_chains = _get_protein_chains(parsed_info=parsed_info)
        if not valid_chains:
            return ParsingResult(None, {(file_id, ''): 'No protein chains found in this file.'})
        seq_start_num = {chain_id: min([monomer.num for monomer in seq]) for chain_id, seq in valid_chains.items()}

        # Loop over the atoms for which we have coordinates. Populate two mappings:
        # -mmcif_to_author_chain_id (maps internal mmCIF chain ids to chain ids used
        # the authors / Biopython).
        # -seq_to_structure_mappings (maps idx into sequence to ResidueAtPosition).
        mmcif_to_author_chain_id = {}
        seq_to_structure_mappings = {}
        for atom in _get_atom_site_list(parsed_info):
            if atom.model_num != '1':
                # We only process the first model at the moment.
                continue

            mmcif_to_author_chain_id[atom.mmcif_chain_id] = atom.author_chain_id

            if atom.mmcif_chain_id in valid_chains:
                hetflag = ' '
                if atom.hetatm_atom == 'HETATM':
                    # Water atoms are assigned a special hetflag of W in Biopython. We
                    # need to do the same, so that this hetflag can be used to fetch
                    # a residue from the Biopython structure by id.
                    if atom.residue_name in ('HOH', 'WAT'):
                        hetflag = 'W'
                    else:
                        hetflag = 'H_' + atom.residue_name
                insertion_code = atom.insertion_code
                if not _is_set(atom.insertion_code):
                    insertion_code = ' '
                position = ResiduePosition(chain_id=atom.author_chain_id, residue_number=int(
                    atom.author_seq_num), insertion_code=insertion_code)
                seq_idx = int(atom.mmcif_seq_num) - seq_start_num[atom.mmcif_chain_id]
                current = seq_to_structure_mappings.get(atom.author_chain_id, {})
                current[seq_idx] = ResidueAtPosition(position=position,
                                                     name=atom.residue_name,
                                                     is_missing=False,
                                                     hetflag=hetflag)
                seq_to_structure_mappings[atom.author_chain_id] = current

        # Add missing residue information to seq_to_structure_mappings.
        for chain_id, seq_info in valid_chains.items():
            author_chain = mmcif_to_author_chain_id.get(chain_id)
            current_mapping = seq_to_structure_mappings.get(author_chain)
            for idx, monomer in enumerate(seq_info):
                if idx not in current_mapping:
                    current_mapping[idx] = ResidueAtPosition(position=None,
                                                             name=monomer.id,
                                                             is_missing=True,
                                                             hetflag=' ')

        author_chain_to_sequence = {}
        for chain_id, seq_info in valid_chains.items():
            author_chain = mmcif_to_author_chain_id.get(chain_id)
            seq = []
            for monomer in seq_info:
                code = SCOPData.protein_letters_3to1.get(monomer.id, 'X')
                seq.append(code if len(code) == 1 else 'X')
            seq = ''.join(seq)
            author_chain_to_sequence[author_chain] = seq

        mmcif_object = MmcifObject(
            file_id=file_id,
            header=header,
            structure=first_model_structure,
            chain_to_seqres=author_chain_to_sequence,
            seqres_to_structure=seq_to_structure_mappings,
            raw_string=parsed_info)

        return ParsingResult(mmcif_object=mmcif_object, errors=errors)
    except Exception as e:  # pylint:disable=broad-except
        errors[(file_id, '')] = e
        if not catch_all_errors:
            raise
        return ParsingResult(mmcif_object=None, errors=errors)


def _get_first_model(structure: PDBSTRUCTURE) -> PDBSTRUCTURE:
    """Returns the first model in a Biopython structure."""
    return next(structure.get_models())


_MIN_LENGTH_OF_CHAIN_TO_BE_COUNTED_AS_PEPTIDE = 21


def get_release_date(parsed_info: MmCIFDict) -> str:
    """Returns the oldest revision date."""
    revision_dates = parsed_info['_pdbx_audit_revision_history.revision_date']
    return min(revision_dates)


def _get_header(parsed_info: MmCIFDict) -> PdbHeader:
    """Returns a basic header containing method, release date and resolution."""
    header = {}

    experiments = mmcif_loop_to_list('_exptl.', parsed_info)
    header['structure_method'] = ','.join([experiment['_exptl.method'].lower() for experiment in experiments])

    # Note: The release_date here corresponds to the oldest revision. We prefer to
    # use this for dataset filtering over the deposition_date.
    if '_pdbx_audit_revision_history.revision_date' in parsed_info:
        header['release_date'] = get_release_date(parsed_info)
    else:
        logging.warning('Could not determine release_date: %s', parsed_info['_entry.id'])

    header['resolution'] = 0.00
    for res_key in ('_refine.ls_d_res_high', '_em_3d_reconstruction.resolution', '_reflns.d_resolution_high'):
        if res_key in parsed_info:
            try:
                raw_resolution = parsed_info[res_key][0]
                header['resolution'] = float(raw_resolution)
            except ValueError:
                logging.warning('Invalid resolution format: %s', parsed_info[res_key])

    return header


def _get_atom_site_list(parsed_info: MmCIFDict) -> Sequence[AtomSite]:
    """Returns list of atom sites; contains data not present in the structure."""
    return [AtomSite(*site) for site in zip(  # pylint:disable=g-complex-comprehension
        parsed_info['_atom_site.label_comp_id'],
        parsed_info['_atom_site.auth_asym_id'],
        parsed_info['_atom_site.label_asym_id'],
        parsed_info['_atom_site.auth_seq_id'],
        parsed_info['_atom_site.label_seq_id'],
        parsed_info['_atom_site.pdbx_PDB_ins_code'],
        parsed_info['_atom_site.group_PDB'],
        parsed_info['_atom_site.pdbx_PDB_model_num'],
    )]


def _get_protein_chains(*, parsed_info: Mapping[str, Any]) -> Mapping[ChainId, Sequence[Monomer]]:
    """Extracts polymer information for protein chains only.

    Args:
      parsed_info: _mmcif_dict produced by the Biopython parser.

    Returns:
      A dict mapping mmcif chain id to a list of Monomers.
    """
    # Get polymer information for each entity in the structure.
    entity_poly_seqs = mmcif_loop_to_list('_entity_poly_seq.', parsed_info)

    polymers = collections.defaultdict(list)
    for entity_poly_seq in entity_poly_seqs:
        polymers[entity_poly_seq['_entity_poly_seq.entity_id']].append(
            Monomer(id=entity_poly_seq['_entity_poly_seq.mon_id'], num=int(entity_poly_seq['_entity_poly_seq.num'])))

    # Get chemical compositions. Will allow us to identify which of these polymers
    # are proteins.
    chem_comps = mmcif_loop_to_dict('_chem_comp.', '_chem_comp.id', parsed_info)

    # Get chains information for each entity. Necessary so that we can return a
    # dict keyed on chain id rather than entity.
    struct_asyms = mmcif_loop_to_list('_struct_asym.', parsed_info)

    entity_to_mmcif_chains = collections.defaultdict(list)
    for struct_asym in struct_asyms:
        chain_id = struct_asym['_struct_asym.id']
        entity_id = struct_asym['_struct_asym.entity_id']
        entity_to_mmcif_chains[entity_id].append(chain_id)

    # Identify and return the valid protein chains.
    valid_chains = {}
    for entity_id, seq_info in polymers.items():
        chain_ids = entity_to_mmcif_chains[entity_id]

        # Reject polymers without any peptide-like components, such as DNA/RNA.
        if any(['peptide' in chem_comps[monomer.id]['_chem_comp.type'] for monomer in seq_info]):
            for chain_id in chain_ids:
                valid_chains[chain_id] = seq_info
    return valid_chains


def _is_set(data: str) -> bool:
    """Returns False if data is a special mmCIF character indicating 'unset'."""
    return data not in ('.', '?')


def get_charge_from_file(file_name):
    """Function used to load charge from files generate from xponge.
    Args:
        file_name(str): The file path of charge file, absolute path is suggested.
    Returns:
        chargs(list): The builtin charge list with format float32.
    Example how to generate charge file:
        >>> import Xponge
        >>> import Xponge.forcefield.AMBER.ff14SB
        >>> Save_SPONGE_Input(ALA,'ALA')
        # The generated charge file name could be ALA_charge.txt
    """
    charges = []
    with open(file_name, 'r') as file:
        lines = file.readlines()
    nums = int(lines[0].strip())
    for line in lines[1:nums + 1]:
        charges.append(float(line.strip()))
    return charges


def load_h5(h5_name):
    """Function used to read the h5 format trajectory files.
    Args:
        h5_name(str): The file path of h5 file, absolute file path is suggested.
    """
    f = h5py.File(h5_name, 'r')
    return f['particles']['walker0']['position']['value'][-1]
