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
"""system"""
import mindspore as ms
from mindspore import Parameter, ParameterTuple
from mindspore.common import initializer
from mindspore import ops
from mindspore.ops import functional as F
from mindspore.nn import Cell
from mindspore.common import Tensor
from mindspore import numpy as msnp
from ...common.units import Units, global_units
from ...common.functions import concat_last_dim, concat_penulti, get_kinetic_energy


class SystemCell(Cell):
    """system cell"""
    def __init__(
            self,
            num_atoms,
            mass,
            resname=None,
            resid=None,
            charge=None,
            atom_name=None,
            atom_type=None,
            atomic_number=None,
            bond_index=None,
            residue_index=None,
            coordinates=None,
            velocities=None,
            pbc_box=None,
            num_walkers=1,
            dimension=3,
            unit_length=None,
            unit_energy=None,
    ):
        super().__init__()

        if unit_length is None and unit_energy is None:
            self.units = global_units
        else:
            self.units = Units(unit_length, unit_energy)

        if not isinstance(dimension,
                          (int, ms.int8, ms.int16, ms.int32, ms.int64, ms.uint8, ms.uint16, ms.uint32, ms.uint64)):
            raise ValueError('dimension must be int type!')

        self.num_atoms = num_atoms
        self.num_walkers = num_walkers
        self.dimension = dimension

        # (B,A)
        if atomic_number is not None:
            atomic_number = Tensor(atomic_number, ms.int32)
            if atomic_number.ndim > 2:
                raise ValueError('The rank of atomic_number cannot larger than 2!')
            if atomic_number.shape[-1] != num_atoms:
                raise ValueError('The last dimension of atomic number (' +
                                 str(atomic_number.shape[-1]) + ') of be equal to the num_atoms (' +
                                 str(num_atoms) + ')!')
            if atomic_number.ndim == 1:
                atomic_number = F.expand_dims(atomic_number, 0)
            if atomic_number.shape[0] != num_walkers:
                if atomic_number.shape[0] == 1:
                    print(atomic_number.shape)
                    atomic_number = msnp.broadcast_to(atomic_number, (self.num_walkers, self.num_atoms))
                else:
                    raise ValueError('The firt dimension of atomic_number (' +
                                     str(atomic_number.shape[0]) + ') must equal to 1 or num_walker (' +
                                     str(num_walkers) + ')!')
        self.atomic_number = atomic_number

        # (B,A,D)
        if coordinates is None:
            self.coordinates = Parameter(initializer('normal', [num_walkers, num_atoms, dimension], ms.float32),
                                         name='coordinates')
            self.coordinates_before_one_step = ParameterTuple((Parameter(
                initializer('normal', [num_walkers, num_atoms, dimension], ms.float32),
                name='coordinates_before_one_step'),))
        else:
            coordinates = Tensor(coordinates, ms.float32)
            if coordinates.ndim < 2:
                raise ValueError('The rank of coordinates must be 2 or 3!')

            if coordinates.shape[-1] != dimension:
                raise ValueError('The last dimension of coordinates (' + str(coordinates.shape[-1]) +
                                 ') is not equal to the dimension (' + str(dimension) + ') in SystemCell!')

            if coordinates.shape[-2] != num_atoms:
                raise ValueError('The atoms number in coordinates (' + str(coordinates.shape[-2]) +
                                 ') is not equal to the number of atoms (' + str(self.num_atoms) + ')!')

            if coordinates.ndim == 2:
                coordinates = F.expand_dims(coordinates, 0)
            if coordinates.ndim == 3:
                if coordinates.shape[0] != num_walkers:
                    if coordinates.shape[0] == 1:
                        coordinates = msnp.broadcast_to(coordinates, (num_walkers, self.num_atoms, dimension))
                    else:
                        raise ValueError('The walkers number in coordinates (' + str(coordinates.shape[0]) +
                                         ') is not equal to the num_walkers (' + str(num_walkers) + ')!')
            else:
                raise ValueError('The rank of coordinates cannot be larger than 2!')

            self.coordinates = Parameter(coordinates, name='coordinates')
            self.coordinates_before_one_step = (
                Parameter(coordinates, name='coordinates_before_one_step', requires_grad=False),)

        # (B,A,D)
        if velocities is None:
            velocities = msnp.zeros((num_walkers, num_atoms, dimension), dtype=ms.float32)
            self.velocities = ParameterTuple((Parameter(velocities, name='velocities', requires_grad=False),))
            self.velocities_before_one_step = ParameterTuple(
                (Parameter(velocities, name='velocities_before_one_step', requires_grad=False),))
            self.velocities_before_half_step = ParameterTuple(
                (Parameter(velocities, name='velocities_before_half_step', requires_grad=False),))
        else:
            velocities = Tensor(velocities, ms.float32)
            if velocities.ndim < 2:
                raise ValueError('The rank of velocities must be 2 or 3!')

            if velocities.shape[-1] != dimension:
                raise ValueError('The last dimension of velocities (' + str(velocities.shape[-1]) +
                                 ') is not equal to the dimension (' + str(dimension) + ') in SystemCell!')

            if velocities.shape[-2] != num_atoms:
                raise ValueError('The atoms number in velocities (' + str(velocities.shape[-2]) +
                                 ') is not equal to the number of atoms (' + str(self.num_atoms) + ')!')

            if velocities.ndim == 2:
                velocities = F.expand_dims(velocities, 0)
            if velocities.ndim == 3:
                if velocities.shape[0] != num_walkers:
                    if velocities.shape[0] == 1:
                        velocities = msnp.broadcast_to(velocities, (num_walkers, self.num_atoms, dimension))
                    else:
                        raise ValueError('The walkers number in velocities (' + str(velocities.shape[0]) +
                                         ') is not equal to the num_walkers (' + str(num_walkers) + ')!')
            else:
                raise ValueError('The rank of velocities cannot be larger than 3!')

            self.velocities = ParameterTuple((Parameter(velocities, name='velocities', requires_grad=False),))
            self.velocities_before_one_step = ParameterTuple(
                (Parameter(velocities, name='velocities_before_one_step', requires_grad=False),))
            self.velocities_before_half_step = ParameterTuple(
                (Parameter(velocities, name='velocities_before_half_step', requires_grad=False),))

        # (B,D)
        self.pbc_box = None
        self.pbc = False
        if pbc_box is not None:
            self.pbc = True
            pbc_box = Tensor(pbc_box, ms.float32)
            if pbc_box.ndim < 1:
                raise ValueError('The rank of pbc_box cannot be smaller than 1!')

            if pbc_box.shape[-1] != dimension:
                raise ValueError('The last dimension of pbc_box (' + str(pbc_box.shape[-1]) +
                                 ') must be equal to the dimension in SystemCell (' + str(dimension) + ')!')

            if pbc_box.ndim == 1:
                pbc_box = F.expand_dims(pbc_box, 0)
            if pbc_box.ndim == 2:
                if pbc_box.shape[0] != num_walkers:
                    if pbc_box.shape[0] == 1:
                        pbc_box = msnp.broadcast_to(pbc_box, (num_walkers, dimension))
                    else:
                        raise ValueError('The firt dimension of pbc_box (' + str(pbc_box.shape[0]) +
                                         ') should equal to num_walkers (' + str(num_walkers) + ')!')
            else:
                raise ValueError('The rank of pbc_box cannot be larger than 2!')

            self.pbc_box = Parameter(pbc_box, name='pbc_box', requires_grad=False)

        # (1,A) or (B,A)
        if mass is None:
            self.mass = (Parameter(initializer('one', [1, num_atoms], ms.float32), name='mass', requires_grad=False),)
            self.inv_mass = (
                Parameter(initializer('one', [1, num_atoms, 1], ms.float32), name='inv_mass', requires_grad=False),)
            self.inv_sqrt_mas = (
                Parameter(initializer('one', [1, num_atoms, 1], ms.float32), name='inv_sqrt_mas', requires_grad=False),)
            self.atom_mask = True
        else:
            mass = Tensor(mass, ms.float32)
            if mass.ndim > 2:
                raise ValueError('The rank of mass cannot be larger than 2!')

            if mass.shape[-1] != num_atoms:
                raise ValueError('The last dimension of mass (' + str(mass.shape[-1]) +
                                 ') must be equal to the num_atoms (' + str(num_atoms) + ')!')

            if mass.ndim == 1:
                mass = F.expand_dims(mass, 0)
            if mass.shape[0] != num_walkers and mass.shape[0] != 1:
                raise ValueError('The first dimension of mass (' + str(mass.shape[0]) +
                                 ') must be equal to 1 or num_walkers (' + str(num_walkers) + ')!')

            atom_mask = mass > 0
            inv_mass = F.select(atom_mask, 1.0 / mass, F.zeros_like(mass))
            inv_mass = F.expand_dims(inv_mass, -1)
            inv_sqrt_mass = F.sqrt(inv_mass)

            mass = Parameter(mass, name='mass', requires_grad=False)
            inv_mass = Parameter(inv_mass, name='inv_mass', requires_grad=False)
            inv_sqrt_mass = Parameter(inv_sqrt_mass, name='inv_sqrt_mass', requires_grad=False)

            self.mass = (mass,)
            self.inv_mass = (inv_mass,)
            self.inv_sqrt_mass = (inv_sqrt_mass,)

        self.system_mass = concat_last_dim(self.mass)
        self.system_inv_mass = concat_penulti(self.inv_mass)
        self.system_inv_sqrt_mass = concat_last_dim(self.inv_sqrt_mass)

        if charge is None:
            self.charge = Parameter(msnp.ones([1, num_atoms], ms.float32), name='charge', requires_grad=False)
        else:
            charge = Tensor(charge, ms.float32)
            if charge.ndim > 2:
                raise ValueError('The rank of charge cannot be larger than 2!')

            if charge.shape[-1] != num_atoms:
                raise ValueError('The last dimension of charge (' + str(charge.shape[-1]) +
                                 ') must be equal to the num_atoms (' + str(num_atoms) + ')!')

            if charge.ndim == 1:
                charge = F.expand_dims(charge, 0)
            if charge.shape[0] != num_walkers and charge.shape[0] != 1:
                raise ValueError('The first dimension of charge (' + str(charge.shape[0]) +
                                 ') must be equal to 1 or num_walkers (' + str(num_walkers) + ')!')
            self.charge = Parameter(charge, name='charge', requires_grad=False)

        self.atom_name = atom_name
        self.atom_type = atom_type

        self.bond_index = ParameterTuple(
            (Parameter(Tensor(bond_index, ms.int32), name='bond_index', requires_grad=False),))
        self.residue_index = residue_index

        self.resname = resname
        self.resid = resid

        self.num_constraint = 0

        self.num_com = 3 if self.pbc else 6
        self.degrees_of_freedom = 3 * self.num_atoms - self.num_constraint - self.num_com

        self.kb = Tensor(self.units.boltzmann(), ms.float32)

        self.kinetic_unit_scale = Tensor(self.units.kinetic_ref(), ms.float32)
        self.velocity_unit_scale = Tensor(self.units.velocity_ref(), ms.float32)

        kinetic = self.get_system_kinetic()
        temperature = 2 * kinetic / self.degrees_of_freedom / self.kb

        self.temperature = Parameter(temperature, name='temperature', requires_grad=False)
        self.kinetic = Parameter(kinetic, name='kinetic', requires_grad=False)

        self.keepdim_sum = ops.ReduceSum(keep_dims=True)

    def update_DOFs(self):
        """update DOFS"""
        self.num_com = 3 if self.pbc else 6
        self.degrees_of_freedom = 3 * self.num_atoms - self.num_constraint - self.num_com
        return self.degrees_of_freedom

    def set_pbc_box(self, pbc_box=None):
        """set pbc box"""
        self.pbc_box = pbc_box
        self.pbc = False
        if pbc_box is not None:
            self.pbc = True
            pbc_box = Tensor(pbc_box, ms.float32)
            if pbc_box.ndim < 1:
                raise ValueError('The rank of pbc_box cannot be smaller than 1!')

            if pbc_box.shape[-1] != self.dimension:
                raise ValueError('The last dimension of pbc_box (' + str(pbc_box.shape[-1]) +
                                 ') must be equal to the dimension in SystemCell (' + str(self.dimension) + ')!')

            if pbc_box.ndim == 1:
                pbc_box = F.expand_dims(pbc_box, 0)
            if pbc_box.ndim == 2:
                if pbc_box.shape[0] != self.num_walkers:
                    if pbc_box.shape[0] == 1:
                        pbc_box = msnp.broadcast_to(pbc_box, (self.num_walkers, self.dimension))
                    else:
                        raise ValueError('The firt dimension of pbc_box (' + str(pbc_box.shape[0]) +
                                         ') should equal to num_walkers (' + str(self.num_walkers) + ')!')
            else:
                raise ValueError('The rank of pbc_box cannot be larger than 2!')

            self.pbc_box = Parameter(pbc_box, name='pbc_box', requires_grad=False)
        self.update_DOFs()
        return self

    def align_pbc_box(self, shift=0):
        """align pbc box"""
        coord_align = position_in_pbc(self.coordinates, self.pbc_box, shift)
        return F.assign(self.coordinates, coord_align)

    def length_unit(self):
        """length unit"""
        return self.units.length_unit()

    def set_length_unit(self, unit):
        """set length unit"""
        self.units.set_length_unit(unit)
        return self

    def get_kinetic_energy(self, m, v):
        """get kinetic energy"""
        kinetic = get_kinetic_energy(m, v) * self.kinetic_unit_scale
        return kinetic

    def get_system_kinetic(self):
        """get system kinetic"""
        v = concat_penulti(self.velocities)
        kinetic = self.get_kinetic_energy(self.system_mass, v)
        return kinetic

    def update_thermo(self, kinetic, temperature):
        """update thermo"""
        success = True
        success = F.depend(success, F.assign(self.kinetic, kinetic))
        success = F.depend(success, F.assign(self.temperature, temperature))
        return success

    def construct(self):
        return self.coordinates, self.pbc_box
