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
"""Force filed"""
import os
import numpy as np
import mindspore as ms
import mindspore.numpy as msnp
from mindspore import Tensor
from mindspore import ops
from mindspore.nn import CellList
from .energy import EnergyCell, BondEnergy, AngleEnergy, DihedralEnergy, NB14Energy
from .energy import CoulombEnergy, LennardJonesEnergy
from .potential import PotentialCell
from ..data import Params
from ..data.forcefield import get_yaml_dict
from ..system import Molecule
from ..function.units import Units


FORCE_FIELD_KEYS = ['ff14SB', 'test']
THIS_PATH = os.path.abspath(__file__)
BUILTIN_FF_PATH = THIS_PATH.replace(
    'potential/forcefield.py', 'data/forcefield/')


class ForceFieldBase(PotentialCell):
    r"""Basic cell for force filed

    Args:

        Energy (EnergyCell or list):    Energy terms. Default: None

        cutoff (float):                 Cutoff distance. Default: None

        exclude_index (Tensor):         Tensor of shape (B, A, Ex). Data type is int
                                        The indexes of atoms that should be excluded from neighbour list.
                                        Default: None

        length_unit (str):              Length unit for position coordinate. Default: None

        energy_unit (str):              Energy unit. Default: None

        units (Units):                  Units of length and energy. Default: None

        use_pbc (bool):                 Whether to use periodic boundary condition.

    """

    def __init__(self,
                 energy: EnergyCell = None,
                 cutoff: float = None,
                 exclude_index: Tensor = None,
                 length_unit: str = None,
                 energy_unit: str = None,
                 units: Units = None,
                 use_pbc: bool = None,
                 ):

        super().__init__(
            cutoff=cutoff,
            exclude_index=exclude_index,
            length_unit=length_unit,
            energy_unit=energy_unit,
            units=units,
            use_pbc=use_pbc,
        )

        self.num_energy = 0
        self.energy_network = self.set_energy_network(energy)

        self.energy_scale = 1

        self.output_unit_scale = self.set_unit_scale()

        self.concat = ops.Concat(-1)

    def set_energy_scale(self, scale: Tensor):
        """set energy scale"""
        scale = Tensor(scale, ms.float32)
        if scale.ndim != 1 and scale.ndim != 0:
            raise ValueError('The rank of energy scale must be 0 or 1.')
        if scale.shape[-1] != self.output_dim and self.shape[-1] != 1:
            raise ValueError('The dimension of energy scale must be equal to the dimension of energy ' +
                             str(self.output_dim)+' or 1, but got: '+str(self.shape[-1]))
        self.energy_scale = scale
        return self

    def set_energy_network(self, energy: EnergyCell) -> CellList:
        """set energy"""
        if energy is None:
            return None
        if isinstance(energy, EnergyCell):
            self.num_energy = 1
            energy = CellList([energy])
        elif isinstance(energy, list):
            self.num_energy = len(energy)
            energy = CellList(energy)
        else:
            raise TypeError(
                'The type of energy must be EnergyCell or list but got: '+str(type(energy)))

        self.output_dim = 0
        if energy is not None:
            for i in range(self.num_energy):
                self.output_dim += energy[i].output_dim
        return energy

    def set_unit_scale(self) -> Tensor:
        """set unit scale"""
        if self.energy_network is None:
            return 1
        output_unit_scale = ()
        for i in range(self.num_energy):
            self.energy_network[i].set_input_unit(self.units)
            dim = self.energy_network[i].output_dim
            scale = np.ones((dim,), np.float32) * \
                self.energy_network[i].convert_energy_to(self.units)
            output_unit_scale += (scale,)
        output_unit_scale = np.concatenate(output_unit_scale, axis=-1)
        return Tensor(output_unit_scale, ms.float32)

    def set_units(self, length_unit: str = None, energy_unit: str = None, units: Units = None):
        """set units"""
        if units is not None:
            self.units.set_units(units=units)
        else:
            if length_unit is not None:
                self.units.set_length_unit(length_unit)
            if energy_unit is not None:
                self.units.set_energy_unit(energy_unit)

        self.output_unit_scale = self.set_unit_scale()

        return self

    def set_pbc(self, use_pbc: Tensor = None):
        """set whether to use periodic boundary condition."""
        for i in range(self.num_energy):
            self.energy_network[i].set_pbc(use_pbc)
        return self

    def construct(self,
                  coordinate: Tensor,
                  neighbour_index: Tensor = None,
                  neighbour_mask: Tensor = None,
                  neighbour_coord: Tensor = None,
                  neighbour_distance: Tensor = None,
                  pbc_box: Tensor = None
                  ):
        r"""Calculate potential energy.

        Args:
            coordinate (Tensor):           Tensor of shape (B, A, D). Data type is float.
                                            Position coordinate of atoms in system.
            neighbour_index (Tensor):       Tensor of shape (B, A, N). Data type is int.
                                            Index of neighbour atoms. Default: None
            neighbour_mask (Tensor):        Tensor of shape (B, A, N). Data type is bool.
                                            Mask for neighbour atoms. Default: None
            neighbour_coord (Tensor):       Tensor of shape (B, A, N). Data type is bool.
                                            Position coorindates of neighbour atoms.
            neighbour_distance (Tensor):   Tensor of shape (B, A, N). Data type is float.
                                            Distance between neighbours atoms. Default: None
            pbc_box (Tensor):               Tensor of shape (B, D). Data type is float.
                                            Tensor of PBC box. Default: None

        Returns:
            potential (Tensor): Tensor of shape (B, 1). Data type is float.

        Symbols:
            B:  Batchsize, i.e. number of walkers in simulation
            A:  Number of atoms.
            N:  Maximum number of neighbour atoms.
            D:  Dimension of the simulation system. Usually is 3.

        """

        inv_neigh_dis = 0
        inv_neigh_dis = msnp.reciprocal(neighbour_distance)
        if neighbour_mask is not None:
            inv_neigh_dis = msnp.where(neighbour_mask, inv_neigh_dis, 0)

        potential = ()
        for i in range(self.num_energy):
            ene = self.energy_network[i](
                coordinate=coordinate,
                neighbour_index=neighbour_index,
                neighbour_mask=neighbour_mask,
                neighbour_coord=neighbour_coord,
                neighbour_distance=neighbour_distance,
                inv_neigh_dis=inv_neigh_dis,
                pbc_box=pbc_box
            )
            potential += (ene,)

        potential = self.concat(potential) * \
            self.energy_scale * self.output_unit_scale

        return potential


class ForceField(ForceFieldBase):
    r"""Potential of classical force field

    Args:

        system (Molecule):  Simulation system.

        cutoff (float):     Cutoff distance. Default: None

        length_unit (str):  Length unit for position coordinate. Default: None

        energy_unit (str):  Energy unit. Default: None

        units (Units):      Units of length and energy. Default: None

        use_pbc (bool):     Whether to use periodic boundary condition.

    """

    def __init__(self,
                 system: Molecule,
                 forceparams: dict,
                 cutoff: float = None,
                 length_unit: str = None,
                 energy_unit: str = None,
                 units: Units = None,
                 use_pbc: bool = None,
                 collect_dihedrals: bool = True,
                 ):

        super().__init__(
            cutoff=cutoff,
            exclude_index=None,
            length_unit=length_unit,
            energy_unit=energy_unit,
            units=units,
            use_pbc=use_pbc,
        )
        # Check Parameters
        self.check_params(system)

        # Generate Forcefield Parameters
        this_force_constants = None
        if isinstance(forceparams, dict):
            this_force_constants = forceparams
            for residue in system.residue:
                residue.build_atom_types(forceparams)
                residue.build_atom_charge(forceparams)

        if isinstance(forceparams, str):
            if forceparams in FORCE_FIELD_KEYS:
                this_force_constants = get_yaml_dict(
                    BUILTIN_FF_PATH+forceparams+'.yaml')
            else:
                this_force_constants = get_yaml_dict(forceparams)
            for residue in system.residue:
                residue.build_atom_types(this_force_constants)
                residue.build_atom_charge(this_force_constants)

        system._build_system()

        force_object = Params(
            system.atom_type[0], this_force_constants, atom_names=system.atom_name[0])

        if isinstance(system.bond, np.ndarray):
            force_params = force_object(system.bond[0])
        if isinstance(system.bond, Tensor):
            force_params = force_object(system.bond[0].asnumpy())

        ff_units = Units('A', 'kcal/mol')

        energy = []

        # Bond energy
        if system.bond is not None:
            bond_index = Tensor(system.bond[0], ms.int32)
            rk_init = Tensor(force_params[0][:, 2][None, :], ms.float32)
            req_init = Tensor(force_params[0][:, 3][None, :], ms.float32)
            bond_energy = BondEnergy(bond_index, rk_init=rk_init,
                                     req_init=req_init, scale=1, use_pbc=use_pbc, units=ff_units)
            energy.append(bond_energy)

        # Angle energy
        if force_params[4] is not None:
            angle_index = Tensor(force_params[4][None, :], ms.int32)
            tk_init = Tensor(force_params[1][:, 3][None, :], ms.float32)
            teq_init = Tensor(force_params[1][:, 4][None, :], ms.float32)
            angle_energy = AngleEnergy(angle_index, tk_init=tk_init,
                                       teq_init=teq_init, scale=1, use_pbc=use_pbc, units=ff_units)
            energy.append(angle_energy)

        # Dihedral energy
        if force_params[2] is not None:
            dihedral_index = Tensor(
                force_params[2][:, [0, 1, 2, 3]][None, :], ms.int32)
            pk_init = Tensor(force_params[2][:, 5][None, :], ms.float32)
            pn_init = Tensor(force_params[2][:, 4][None, :], ms.int32)
            phase_init = Tensor(force_params[2][:, 6][None, :], ms.float32)

            # Idihedral Parameters
            idihedral_index = Tensor(
                force_params[3][:, [0, 1, 2, 3]][None, :], ms.int32)
            if collect_dihedrals:
                dihedral_index = msnp.append(
                    dihedral_index, idihedral_index, axis=1)
                pk_init = msnp.append(pk_init, Tensor(
                    force_params[3][:, 5][None, :], ms.float32), axis=1)
                pn_init = msnp.append(pn_init, Tensor(
                    force_params[3][:, 4][None, :], ms.int32), axis=1)
                phase_init = msnp.append(phase_init, Tensor(
                    force_params[3][:, 6][None, :], ms.float32), axis=1)

            dihedral_energy = DihedralEnergy(dihedral_index, pk_init=pk_init, pn_init=pn_init,
                                             phase_init=phase_init, scale=1, use_pbc=use_pbc, units=ff_units)
            energy.append(dihedral_energy)

        # Electronic energy
        if system.atom_charge is not None:
            ele_energy = CoulombEnergy(
                atom_charge=system.atom_charge, use_pbc=use_pbc, units=ff_units)
            energy.append(ele_energy)

        # VDW energy
        atomic_radius = None
        well_depth = None
        if force_params[8] is not None:
            atomic_radius = Tensor(force_params[8][:, 0][None, :], ms.float32)
            well_depth = Tensor(force_params[8][:, 1][None, :], ms.float32)
            vdw_energy = LennardJonesEnergy(atomic_radius=atomic_radius, well_depth=well_depth,
                                            use_pbc=use_pbc, units=ff_units)
            energy.append(vdw_energy)

        # Non-bonded 1-4 energy
        if force_object.nb14_index is not None:
            nb14_index = Tensor(force_object.nb14_index[None, :], ms.int32)
            one_scee = np.array([5 / 6] * nb14_index.shape[-2])[None, :]
            one_scnb = np.array([.5] * nb14_index.shape[-2])[None, :]
            nb14_energy = NB14Energy(nb14_index, atom_charge=system.atom_charge, atomic_radius=atomic_radius,
                                     well_depth=well_depth, one_scee=one_scee, one_scnb=one_scnb,
                                     use_pbc=use_pbc, units=ff_units)
            energy.append(nb14_energy)

        # Exclude Parameters
        self._exclude_index = Tensor(force_params[7][None, :], ms.int32)

        self.energy_network = self.set_energy_network(energy)
        self.output_unit_scale = self.set_unit_scale()

    def check_params(self, system: Molecule):
        """Check if the input parameters for force field is legal.
        """
        # Dimension Checking
        if system.atom_mass.shape[0] != 1:
            raise ValueError('The first dimension of atom mass should be 1.')

        if system.atom_type.shape[0] != 1:
            raise ValueError('The first dimension of atom type should be 1.')

        if system.atom_name.shape[0] != 1:
            raise ValueError('The first dimension of atom name should be 1.')

        if system.bond.shape[0] != 1:
            raise ValueError('The first dimension of bond should be 1.')

        # Type Checking
        if not isinstance(system.atom_name, np.ndarray):
            raise ValueError(
                'The data type of atom name should be numpy.ndarray.')

        if not isinstance(system.atom_type, np.ndarray):
            raise ValueError(
                'The data type of atom type should be numpy.ndarray.')

        if not isinstance(system.atom_mass, np.ndarray) and not isinstance(system.atom_mass, Tensor):
            raise ValueError(
                'The data type of atom mass should be numpy.ndarray or Tensor.')

        if not isinstance(system.bond, np.ndarray) and not isinstance(system.bond, Tensor):
            raise ValueError(
                'The data type of bond should be numpy.ndarray or Tensor.')
