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
"""Non-bonded 1-4 energy"""

import mindspore as ms
import mindspore.numpy as msnp
from mindspore import Tensor
from mindspore import Parameter
from mindspore import ops
from mindspore.ops import functional as F

from .energy import EnergyCell
from ...colvar import AtomDistances
from ...function.units import Units
from ...function.functions import keepdim_sum


class NB14Energy(EnergyCell):
    r"""Non-bonded 1-4 energy.

    Args:

        index (Tensor):             Tensor of shape (B, n, 4). Data type is int.
                                    Atom index of dihedral angles.

        atom_charge (Parameter):    Parameter of atom charge for electronic interaction.

        atomic_radius (Parameter):  Parameter of atomic radius for LJ potential.

        well_depth (Parameter):     Parameter of well depth for LJ potential.

        one_scee (Tensor):          Tensor of shape (1, n). Data type is float.
                                    1-4 electrostatic scaling constant.

        one_scnb (Tensor):          Tensor of shape (1, n). Data type is float.
                                    1-4 LJ scaling constant.

        use_pbc (bool):             Whether to use periodic boundary condition.

        length_unit (str):          Length unit for position coordinates. Default: None

        energy_unit (str):          Energy unit. Default: None

        units (Units):              Units of length and energy. Default: None

    Symbols:

        B:  Batchsize, i.e. number of walkers in simulation

        d:  Number of dihedral angles.

        D:  Dimension of the simulation system. Usually is 3.

    """

    def __init__(self,
                 index: Tensor,
                 atom_charge: Parameter = None,
                 atomic_radius: Parameter = None,
                 well_depth: Parameter = None,
                 one_scee: Tensor = None,
                 one_scnb: Tensor = None,
                 use_pbc: bool = None,
                 length_unit: str = 'nm',
                 energy_unit: str = 'kj/mol',
                 units: Units = None,
                 ):

        super().__init__(
            label='nb14',
            output_dim=2,
            use_pbc=use_pbc,
            length_unit=length_unit,
            energy_unit=energy_unit,
            units=units,
        )

        # (1,n,2)
        index = Tensor(index, ms.int32)
        if index.ndim == 2:
            index = F.expand_dims(index, 0)
        self.index = index

        self.num_nb14 = index.shape[-2]

        self.atomic_radius = None
        if atomic_radius is not None:
            self.atomic_radius = F.cast(self.identity(atomic_radius), ms.float32)

        self.well_depth = None
        if well_depth is not None:
            self.well_depth = F.cast(self.identity(well_depth), ms.float32)

        self.atom_charge = None
        if atom_charge is not None:
            self.atom_charge = self.identity(atom_charge)

        self.get_nb14_distance = AtomDistances(index, use_pbc=use_pbc, length_unit=self.units)

        one_scee = Tensor(one_scee, ms.float32)
        if one_scee.shape[-1] != self.num_nb14:
            raise ValueError('The last shape of one_scee ('+str(one_scee.shape[-1]) +
                             ') must be equal to num_nb14 ('+str(self.num_nb14)+')!')
        if one_scee.ndim == 1:
            one_scee = F.expand_dims(one_scee, 0)
        if one_scee.ndim > 2:
            raise ValueError('The rank of one_scee cannot be larger than 2!')
        self.scee_scale_factor = Parameter(one_scee, name='scee_scale_factor', requires_grad=False)

        one_scnb = Tensor(one_scnb, ms.float32)
        if one_scnb.shape[-1] != self.num_nb14:
            raise ValueError('The last shape of one_scnb ('+str(one_scnb.shape[-1]) +
                             ') must be equal to num_nb14 ('+str(self.num_nb14)+')!')
        if one_scnb.ndim == 1:
            one_scnb = F.expand_dims(one_scnb, 0)
        if one_scnb.ndim > 2:
            raise ValueError('The rank of one_scnb cannot be larger than 2!')
        self.scnb_scale_factor = Parameter(one_scnb, name='scnb_scale_factor', requires_grad=False)

        self.coulomb_const = self.units.coulomb

        self.qiqj = self.get_qiqj()
        self.rij = self.get_rij()
        self.eps = self.get_eps()
        self.concat = ops.Concat(-1)

    def set_pbc(self, use_pbc=None):
        """set whether to use periodic boundary condition."""
        self.use_pbc = use_pbc
        self.get_nb14_distance.set_pbc(use_pbc)
        return self

    def get_qiqj(self) -> Tensor:
        """get the value of q_i and q_j"""
        if self.atom_charge is None:
            return 0
        q = self.gather_values(self.atom_charge, self.index)
        # (B,a) = (B,a,2)
        return F.cast(F.reduce_prod(q, -1), ms.float32)

    def get_rij(self) -> Tensor:
        """the value of distance between atoms"""
        if self.atomic_radius is None:
            return 0
        # (1,a,2)
        rij = self.gather_values(self.atomic_radius, self.index)
        # r_ij = r_i + r_j
        # (1,a) = (B,a,2)
        return F.cast(F.reduce_sum(rij, -1), ms.float32)

    def get_eps(self) -> Tensor:
        """get the value of eps"""
        if self.well_depth is None:
            return 0
        # (1,a,2)
        eps = self.gather_values(self.well_depth, self.index)
        # \eps_ij = sqrt(\eps_i * \eps_j)
        # (1,a) = (1,a,2)
        return F.cast(F.sqrt(F.reduce_prod(eps, -1)), ms.float32)

    def set_cutoff(self, cutoff: float):
        """set cutoff distance"""
        if cutoff is None:
            self.cutoff = None
        else:
            self.cutoff = Tensor(cutoff, ms.float32)
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

        distances = self.get_nb14_distance(coordinate, pbc_box) * self.input_unit_scale
        # (B,a)
        inv_dis = msnp.reciprocal(distances)

        energy_ele14 = 0
        if self.atom_charge is not None:
            # (B,a) = (1,a) * (B,a) * (1,a)
            energy_ele14 = self.qiqj * inv_dis * self.scee_scale_factor
            # (B,1) <- (B,a)
            energy_ele14 = keepdim_sum(energy_ele14, -1) * self.coulomb_const

        energy_vdw14 = 0
        if self.atomic_radius is not None:
            # (B,a) = (1,a)
            r0 = self.rij * inv_dis

            # (B,a)
            r6 = F.pows(r0, 6)
            r12 = r6 * r6

            # (B,a) = (1,a) * (B,a)
            ene_acoeff = self.eps * r12
            ene_bcoeff = self.eps * r6 * 2

            # (B,a)
            energy_vdw14 = ene_acoeff - ene_bcoeff
            energy_vdw14 *= self.scnb_scale_factor

            # (B,1) <- (B,a)
            energy_vdw14 = keepdim_sum(energy_vdw14, -1)

        # (B, 2)
        energy = self.concat((energy_ele14, energy_vdw14))

        return energy
