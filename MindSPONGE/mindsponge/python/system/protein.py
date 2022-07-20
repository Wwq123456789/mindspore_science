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
Protein modeling.
"""

import numpy as np
import mindspore as ms
from mindspore.common import Tensor, Parameter
from mindspore import numpy as msnp
from mindspore import nn, ops
from mindspore.ops import functional as F
from .amino import AminoAcid
from .molecule import Molecule
from .modeling.hadder import ReadPdbByMindsponge as read_pdb


backbone_atoms = np.array(['N', 'CA', 'C', 'O'], np.str_)
include_backbone_atoms = np.array(['OXT'], np.str_)


class Protein(Molecule):
    """
    Basic class of protein constructed by residues, based on Molecule module.
    """

    def __init__(self,
                 pdb_name: str = None,
                 residue_sequence: list = None,
                 crds=None,
                 pbc_box=None,
                 template=None
                 ):
        super().__init__()
        self.identity = ops.Identity()
        # If pdb file is given.
        if pdb_name is not None:
            addh = 1
            _, res_names, _, crds, res_pointers, flatten_atoms, flatten_crds, init_res_names,\
                init_res_ids, \
                residue_index, _, _, _, _ = read_pdb(
                    pdb_name, addh)

            if res_names[0] != 'ACE' and res_names[0] != 'NME':
                res_names[0] = 'N' + res_names[0]
            if res_names[-1] != 'ACE' and res_names[-1] != 'NME':
                res_names[-1] = 'C' + res_names[-1]

            self.init_resname = init_res_names
            self.init_resid = init_res_ids
            self.res_nums = len(res_names)
            self.num_residue = self.res_nums
            self.res_names = res_names
            self.res_pointers = np.append(res_pointers, len(flatten_atoms))
            self.res_id = np.array([j for i in range(1, len(res_pointers)) for j in [i - 1] * res_pointers[i]],
                                   np.int32)
            self.backbone_mask = np.isin(flatten_atoms, backbone_atoms)
            self.oxt_id = np.where(
                np.isin(flatten_atoms, include_backbone_atoms))[0][-1]
            self.backbone_mask[self.oxt_id] = True

            residue = []
            for residue_index in range(self.res_nums):
                this_residue_name = np.array(
                    self.res_names[residue_index], np.str_)
                this_atoms = flatten_atoms[self.res_pointers[residue_index]:
                                           self.res_pointers[residue_index + 1]][None, :]
                this_head_atom = self.get_head_atom(residue_index, this_atoms)
                this_tail_atom = self.get_tail_atom(this_atoms)
                this_residue = AminoAcid(atom_name=this_atoms,
                                         head_atom=this_head_atom,
                                         tail_atom=this_tail_atom,
                                         start_index=0,
                                         name=this_residue_name,
                                         template=template
                                         )
                residue.append(this_residue)

            self.residue = nn.CellList(residue)

            if self.residue is not None:
                self._build_system()

                # (B,A,D)
                self.coordinate = Parameter(
                    flatten_crds[None, :], name='coordinate')
                self.dimension = self.coordinate.shape[-1]
                self.num_walker = self.coordinate.shape[0]

                # (B,1)
                self.system_mass = msnp.sum(self.atom_mass, -1, keepdims=True)
                self.has_empty_atom = (not self.atom_mask.all())
                # (B,1) <- (B,A)
                self.system_natom = msnp.sum(self.atom_mask, -1, keepdims=True)

                self.keep_prod = ops.ReduceProd(keep_dims=True)
                self.identity = ops.Identity()

                # (B,D)
                if pbc_box is None:
                    self.pbc_box = None
                    self.use_pbc = False
                    self.num_com = self.dimension
                    self.image = None
                else:
                    self.use_pbc = True
                    self.num_com = self.dimension
                    pbc_box = Tensor(pbc_box, ms.float32)
                    if pbc_box.ndim == 1:
                        pbc_box = F.expand_dims(pbc_box, 0)
                    if pbc_box.ndim != 2:
                        raise ValueError('The rank of pbc_box must be 1 or 2!')
                    if pbc_box.shape[-1] != self.dimension:
                        raise ValueError('The last dimension of "pbc_box" (' + str(pbc_box.shape[-1]) +
                                         ') must be equal to the dimension of "coordinate" (' +
                                         str(self.dimension) + ')!')
                    if pbc_box.shape[0] > 1 and pbc_box.shape[0] != self.num_walker:
                        raise ValueError('The first dimension of "pbc_box" (' + str(pbc_box.shape[0]) +
                                         ') does not match the first dimension of "coordinate" (' +
                                         str(self.dimension) + ')!')
                    self.pbc_box = Parameter(pbc_box, name='pbc_box')

                    self.image = Parameter(msnp.zeros_like(self.coordinate, ms.int32), name='coordinate_image',
                                           requires_grad=False)
                    self.update_image()

                self.degrees_of_freedom = self.dimension * self.num_atoms - self.num_com

        elif residue_sequence is None:
            raise ValueError(
                'At least 1 of pdb name and residue sequence should be given.')
        elif crds is None:
            raise ValueError(
                'Coordinates should be given when only residue sequence is given.')
        else:
            self.crds = Tensor(crds, ms.float32)

    def get_head_atom(self, residue_index, this_atom_names):
        if residue_index == 0:
            return None
        for index, atom in enumerate(this_atom_names[0]):
            if atom == 'N':
                return np.array([index], np.int32)
        return self

    def get_tail_atom(self, this_atom_names):
        for index, atom in enumerate(this_atom_names[0]):
            if atom == 'C':
                return np.array([index], np.int32)
        return self
