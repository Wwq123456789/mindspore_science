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
"""Electroinc interaction"""

import mindspore as ms
import mindspore.numpy as msnp
from mindspore import Tensor
from mindspore import ops
from mindspore.ops import functional as F

from .energy import NonbondEnergy
from ...function import functions as func
from ...function.functions import gather_values
from ...function.units import Units


class CoulombEnergy(NonbondEnergy):
    r"""Coulomb potential

    Args:

        atom_charge (Tensor):   Tensor of shape (B, A). Data type is float.
                                Atom charge.

        cutoff (float):         Cutoff distance. Default: None

        use_pbc (bool):         Whether to use periodic boundary condition. Default: None

        length_unit (str):      Length unit for position coordinates. Default: None

        energy_unit (str):      Energy unit. Default: None

        units (Units):          Units of length and energy. Default: None

    """
    def __init__(self,
                 atom_charge: Tensor,
                 cutoff: float = None,
                 use_pbc: bool = None,
                 length_unit: str = 'nm',
                 energy_unit: str = 'kj/mol',
                 units: Units = None,
                 ):

        super().__init__(
            label='electronic',
            output_dim=1,
            cutoff=cutoff,
            use_pbc=use_pbc,
            length_unit=length_unit,
            energy_unit=energy_unit,
            units=units,
        )

        self.atom_charge = F.cast(self.identity(atom_charge), ms.float32)
        self.coulomb_const = Tensor(self.units.coulomb, ms.float32)

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
        #pylint: disable=unused-argument

        inv_neigh_dis *= self.inverse_input_scale

        # (B,A,1)
        qi = F.expand_dims(self.atom_charge, -1)
        # (B,A,N)
        qj = gather_values(self.atom_charge, neighbour_index)
        # (B,A,N) = (B,A,1) * (B,A,N)
        qiqj = qi * qj

        energy = qiqj * inv_neigh_dis

        # (B,A)
        energy = F.reduce_sum(energy, -1) * self.coulomb_const
        # (B,1)
        energy = func.keepdim_sum(energy, 1) * 0.5

        return energy


class DSFCoulombEnergy(CoulombEnergy):
    r"""Damped shifted force coulomb potential

    Args:

        atom_charge (Tensor):   Tensor of shape (B, A). Data type is float.
                                Atom charge.

        cutoff (float):         Cutoff distance. Default: None

        alpha (float):          Alpha. Default: 0.25

        use_pbc (bool):         Whether to use periodic boundary condition. Default: None

        length_unit (str):      Length unit for position coordinates. Default: None

        energy_unit (str):      Energy unit. Default: None

        units (Units):          Units of length and energy. Default: None

    """
    def __init__(self,
                 atom_charge: Tensor,
                 cutoff: float = None,
                 alpha: float = 0.25,
                 use_pbc: bool = None,
                 length_unit: str = 'nm',
                 energy_unit: str = 'kj/mol',
                 units: Units = None,
                 ):

        super().__init__(
            atom_charge=atom_charge,
            cutoff=cutoff,
            use_pbc=use_pbc,
            length_unit=length_unit,
            energy_unit=energy_unit,
            units=units,
        )

        self.alpha = Tensor(alpha, ms.float32)

        self.erfc = ops.Erfc()

        cutoffsq = self.cutoff * self.cutoff
        erfcc = self.erfc(self.alpha * self.cutoff)
        erfcd = msnp.exp(-self.alpha * self.alpha * cutoffsq)

        self.f_shift = -(erfcc / cutoffsq + 2 / msnp.sqrt(msnp.pi)
                         * self.alpha * erfcd / self.cutoff)
        self.e_shift = erfcc / self.cutoff - self.f_shift * self.cutoff

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
        neighbour_distance *= self.input_unit_scale
        inv_neigh_dis *= self.inverse_input_scale

        # (B,A,1)
        qi = F.expand_dims(self.atom_charge, -1)
        # (B,A,N)
        qj = gather_values(self.atom_charge, neighbour_index)
        # (B,A,N) = (B,A,1) * (B,A,N)
        qiqj = qi*qj
        energy = qiqj * inv_neigh_dis * \
            (self.erfc(self.alpha * neighbour_distance) - neighbour_distance *
             self.e_shift - neighbour_distance * neighbour_distance * self.f_shift)
        energy = msnp.where(
            neighbour_distance < self.cutoff, energy, 0.0)

        # (B,A)
        energy = F.reduce_sum(energy, -1) * self.coulomb_const
        # (B,1)
        energy = func.keepdim_sum(energy, 1) * 0.5

        return energy


class PMECoulombEnergy(CoulombEnergy):
    r"""Particle mesh ewald algorithm for electronic interaction

    Args:

        atom_charge (Tensor):   Tensor of shape (B, A). Data type is float.
                                Atom charge.

        cutoff (float):         Cutoff distance. Default: None

        use_pbc (bool):         Whether to use periodic boundary condition. Default: None

        length_unit (str):      Length unit for position coordinates. Default: None

        energy_unit (str):      Energy unit. Default: None

        units (Units):          Units of length and energy. Default: None

    """
    def __init__(self,
                 atom_charge: Tensor,
                 cutoff: float = None,
                 use_pbc: bool = None,
                 length_unit: str = 'nm',
                 energy_unit: str = 'kj/mol',
                 units: Units = None,
                 ):

        super().__init__(
            atom_charge=atom_charge,
            cutoff=cutoff,
            use_pbc=use_pbc,
            length_unit=length_unit,
            energy_unit=energy_unit,
            units=units,
        )

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

        # TODO
        raise NotImplementedError
