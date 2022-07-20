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
"""Lennard-Jones potential"""

import mindspore as ms
import mindspore.numpy as msnp
from mindspore import Tensor, Parameter
from mindspore.ops import functional as F

from .energy import NonbondEnergy
from ...function import functions as func
from ...function.functions import gather_values
from ...function.units import Units


class LennardJonesEnergy(NonbondEnergy):
    r"""Lennard-Jones potential

    Args:

        atomic_radius (Parameter):      Parameter of shape (B, A). Data type is float.
                                        Atomic radius for LJ potential.

        well_depth (Parameter):         Parameter of shape (B, A). Data type is float.
                                        Well depth for LJ potential.

        average_dispersion (Parameter): Parameter of shape (B, A). Data type is float.
                                        Average dispersion of the system. Default: None

        num_atoms (int):                A constant value to scale the output.
                                        Default: None

        cutoff (float):                 Cutoff distance. Default: None

    Symbols:

        B:  Batchsize, i.e. number of walkers in simulation

        A:  Number of atoms.

        N:  Maximum number of neighbour atoms.

        D:  Dimension of the simulation system. Usually is 3.

    """
    def __init__(self,
                 atomic_radius: Tensor,
                 well_depth: Tensor,
                 average_dispersion: Tensor = 0,
                 cutoff: float = None,
                 use_pbc: bool = None,
                 length_unit: str = 'nm',
                 energy_unit: str = 'kj/mol',
                 units: Units = None,
                 ):

        super().__init__(
            label='vdw',
            output_dim=1,
            cutoff=cutoff,
            use_pbc=use_pbc,
            length_unit=length_unit,
            energy_unit=energy_unit,
            units=units,
        )

        atomic_radius = Tensor(atomic_radius, ms.float32)
        well_depth = Tensor(well_depth, ms.float32)

        if atomic_radius.shape[-1] != well_depth.shape[-1]:
            raise ValueError('the last dimension of atomic_radius'+str(atomic_radius.shape[-1]) +
                             'must be equal to the last dimension of well_depth ('+str(well_depth.shape[-1])+')!')

        self.num_atoms = atomic_radius.shape[-1]

        if atomic_radius.ndim == 1:
            atomic_radius = F.expand_dims(atomic_radius, 0)
        if atomic_radius.ndim > 2:
            raise ValueError('The rank of atomic_raidus cannot be larger than 2!')
        self.atomic_radius = Parameter(atomic_radius, name='atomic_radius')

        if well_depth.ndim == 1:
            well_depth = F.expand_dims(well_depth, 0)
        if well_depth.ndim > 2:
            raise ValueError('The rank of well_depth cannot be larger than 2!')
        self.well_depth = Parameter(well_depth, name='well_depth')

        self.average_dispersion = Parameter(Tensor(average_dispersion, ms.float32),
                                            name='average dispersion', requires_grad=False)

        self.correct_factor = self._calc_correct_factor()

    def set_cutoff(self, cutoff: float):
        """set cutoff distance"""
        super().set_cutoff(cutoff)
        self.correct_factor = self._calc_correct_factor()
        return self

    def _calc_correct_factor(self) -> Tensor:
        """calculate the correct factor"""
        if self.cutoff is None:
            return 0
        return -2.0 / 3.0 * msnp.pi * self.num_atoms**2 / msnp.power(self.cutoff, 3)

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

        inv_neigh_dis *= self.inverse_input_scale

        atomic_radius = self.identity(self.atomic_radius)
        well_depth = self.identity(self.well_depth)

        # (B,A,1)
        ri = F.expand_dims(atomic_radius, -1)
        # (B,A,N)
        rj = gather_values(atomic_radius, neighbour_index)
        # (B,A,N) = (B,A,1) * (B,A,N)
        r0 = ri + rj

        # (B,A,1)
        eps_i = F.expand_dims(well_depth, -1)
        # (B,A,N)
        eps_j = gather_values(well_depth, neighbour_index)
        # (B,A,N) = (B,A,1) * (B,A,N)
        eps = F.sqrt(eps_i * eps_j)

        rij = r0 * inv_neigh_dis

        r6 = F.pows(rij, 6)
        r12 = r6 * r6

        ene_acoeff = eps * r12
        ene_bcoeff = eps * r6 * 2

        # (B,A,N)
        energy = ene_acoeff - ene_bcoeff

        # (B,A)
        energy = F.reduce_sum(energy, -1)
        # (B,1)
        energy = func.keepdim_sum(energy, -1) * 0.5

        if self.cutoff is not None and pbc_box is not None:
            # (B,1) <- (B,D)
            volume = func.keepdim_prod(pbc_box, -1)
            # E_corr = -2 / 3 * pi * N * \rho * C_6 * r_c^-3
            #        = -2 / 3 * pi * N * (N / V) * C_6 * r_c^-3
            #        = -2 / 3 * pi * N^2 * C_6 / V
            #        = k_corr * C_6 / V
            ene_corr = self.correct_factor * \
                       self.average_dispersion * msnp.reciprocal(volume)
            energy += ene_corr

        return energy
