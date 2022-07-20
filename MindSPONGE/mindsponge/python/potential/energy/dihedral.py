# Copyright 2021-2022 The AIMM Group at Shenzhen Bay Laboratory & Peking University
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
"""Torsion energy"""

import mindspore as ms
from mindspore import Tensor
from mindspore.ops import functional as F
from mindspore import Parameter

from .energy import EnergyCell
from ...colvar import AtomTorsions
from ...function import functions as func
from ...function.units import Units

class DihedralEnergy(EnergyCell):
    r"""Energy term of dihedral (torsion) angles.

    Args:

        index (Tensor):         Tensor of shape (B, d, 4). Data type is int.
                                Atom index of dihedral angles.

        pk_init (Tensor):       Tensor of shape (1, d). Data type is float.
                                The barrier height divided by a factor of 2.

        pn_init (Tensor):       Tensor of shape (1, d). Data type is int.
                                The periodicity of the torsional barrier.

        phase_init (Tensor):    Tensor of shape (1, d). Data type is float.
                                The phase shift angle in the torsional function.

        scale (float):          A constant value to scale the output. Default: 0.5

        use_pbc (bool):         Whether to use periodic boundary condition.

        energy_unit (str):      Energy unit. Default: None

        units (Units):          Units of length and energy. Default: None

    Symbols:

        B:  Batchsize, i.e. number of walkers in simulation

        d:  Number of dihedral angles.

        D:  Dimension of the simulation system. Usually is 3.

    """
    def __init__(self,
                 index: Tensor,
                 pk_init: Tensor,
                 pn_init: Tensor,
                 phase_init: Tensor,
                 scale: float = 0.5,
                 use_pbc: bool = None,
                 energy_unit: str = 'kj/mol',
                 units: Units = None,
                 ):

        super().__init__(
            label='dihedral',
            output_dim=1,
            use_pbc=use_pbc,
            energy_unit=energy_unit,
            units=units,
        )

        # (1,d,4)
        index = Tensor(index, ms.int32)
        if index.ndim == 2:
            index = F.expand_dims(index, 0)
        self.index = Parameter(
            index, name='dihedral_index', requires_grad=False)

        # (1,d)
        self.torsions = AtomTorsions(index, use_pbc=use_pbc)

        # d
        self.num_torsions = index.shape[-2]

        # (1,d)
        pk_init = Tensor(pk_init, ms.float32)
        if pk_init.shape[-1] != self.num_torsions:
            raise ValueError('The last shape of pk_init ('+str(pk_init.shape[-1]) +
                             ') must be equal to num_torsions ('+str(self.num_torsions)+')!')
        if pk_init.ndim == 1:
            pk_init = F.expand_dims(pk_init, 0)
        if pk_init.ndim > 2:
            raise ValueError('The rank of pk_init cannot be larger than 2!')
        self.dihedral_force_constant = Parameter(pk_init, name='dihedral_force_constant')

        pn_init = Tensor(pn_init, ms.int32)
        if pn_init.shape[-1] != self.num_torsions:
            raise ValueError('The last shape of pn_init ('+str(pn_init.shape[-1]) +
                             ') must be equal to num_torsions ('+str(self.num_torsions)+')!')
        if pn_init.ndim == 1:
            pn_init = F.expand_dims(pn_init, 0)
        if pn_init.ndim > 2:
            raise ValueError('The rank of pn_init cannot be larger than 2!')
        self.dihedral_periodicity = Parameter(pn_init, name='dihedral_periodicity')

        phase_init = Tensor(phase_init, ms.float32)
        if phase_init.shape[-1] != self.num_torsions:
            raise ValueError('The last shape of phase_init ('+str(phase_init.shape[-1]) +
                             ') must be equal to num_torsions ('+str(self.num_torsions)+')!')
        if phase_init.ndim == 1:
            phase_init = F.expand_dims(phase_init, 0)
        if phase_init.ndim > 2:
            raise ValueError('The rank of phase_init cannot be larger than 2!')
        self.dihedral_phase = Parameter(phase_init, name='dihedral_phase')

        self.scale = Tensor(scale, ms.float32)

    def set_pbc(self, use_pbc=None):
        self.use_pbc = use_pbc
        self.torsions.set_pbc(use_pbc)
        return self

    def construct(self,
                  coordinate: Tensor,
                  neighbour_index: Tensor = None,
                  neighbour_mask: Tensor = None,
                  neighbour_coord: Tensor = None,
                  neighbour_distance: Tensor = None,
                  inv_neigh_dis: Tensor = None,
                  pbc_box: Tensor = None,
                  ):
        r"""Calculate energy term.

        Args:
            coordinate (Tensor):            Tensor of shape (B, A, D). Data type is float.
                                            Position coordinate of atoms in system
            neighbour_index (Tensor):       Tensor of shape (B, A, N). Data type is int.
                                            Index of neighbour atoms.
            neighbour_mask (Tensor):        Tensor of shape (B, A, N). Data type is bool.
                                            Mask for neighbour index.
            neighbour_coord (Tensor):       Tensor of shape (B, A, N). Data type is bool.
                                            Position coorindates of neighbour atoms.
            neighbour_distance (Tensor):    Tensor of shape (B, A, N). Data type is float.
                                            Distance between neighbours atoms.
            inv_neigh_dis (Tensor):         Tensor of shape (B, A, N). Data type is float.
                                            Reciprocal of distances.
            pbc_box (Tensor):               Tensor of shape (B, D). Data type is float.
                                            Tensor of PBC box. Default: None

        Returns:
            energy (Tensor):    Tensor of shape (B, 1). Data type is float.

        Symbols:
            B:  Batchsize, i.e. number of walkers in simulation
            A:  Number of atoms.
            D:  Dimension of the simulation system. Usually is 3.

        """
        # (B,M)
        phi = self.torsions(coordinate, pbc_box)

        # (B,M) = (1,M) * (B,M)
        nphi = self.dihedral_periodicity * phi

        # (B,M)
        cosphi = F.cos(nphi - self.dihedral_phase) + 1

        # (B,M) = (1,M) + (B,M)
        energy = self.dihedral_force_constant * cosphi * self.scale

        # (B,1) <- (B,M)
        energy = func.keepdim_sum(energy, -1)

        return energy
