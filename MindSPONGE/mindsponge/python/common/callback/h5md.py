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
"""h5md"""

import os
import numpy as np
import h5py
from mindspore.train._utils import _make_directory
from mindspore.train.callback import Callback, RunContext

from ..units import Units, global_units
from ...core.space.system import SystemCell
from ..elements import vmd_radius

_cur_dir = os.getcwd()


class WriteH5MD(Callback):
    '''write h5md'''
    def __init__(
            self,
            system: SystemCell,
            filename: str,
            directory: str = None,
            mode: str = 'w',
            write_velocity: bool = False,
            write_force: bool = False,
            compression='gzip',
            compression_opts=4,
            unit_length=None,
            unit_energy=None,
    ):

        if directory is not None:
            self._directory = _make_directory(directory)
        else:
            self._directory = _cur_dir
        self.filename = os.path.join(self._directory, filename)

        self.system = system
        if unit_length is None and unit_energy is None:
            self.units = global_units
        else:
            self.units = Units(unit_length, unit_energy)

        self.num_walkers = self.system.num_walkers
        self.num_atoms = self.system.num_atoms
        self.dimension = self.system.dimension
        self.coordinates = self.system.coordinates
        self.crd_shape = (None, self.num_atoms, self.dimension)
        self.pbc = self.system.pbc

        atomic_number = None
        atom_raidus = None
        if self.system.atomic_number is not None:
            atomic_number = self.system.atomic_number.asnumpy()
            atom_raidus = vmd_radius[atomic_number[0]] * system.units.convert_length_from('nm')
            if self.num_walkers > 1 and atomic_number.shape[0] == 1:
                atomic_number = np.broadcast_to(atomic_number, (self.num_walkers, self.num_atoms))

        self.length_unit_scale = self.units.convert_length_from(self.system.units)
        self.force_unit_scale = self.units.convert_energy_from(self.system.units) / self.length_unit_scale

        atom_name = None
        if self.system.atom_name is not None:
            atom_name = [s.encode('ascii', 'ignore') for s in self.system.atom_name]

        atom_type = None
        if self.system.atom_type is not None:
            atom_type = [s.encode('ascii', 'ignore') for s in self.system.atom_type]

        resname = None
        if self.system.resname is not None:
            resname = [s.encode('ascii', 'ignore') for s in self.system.resname]

        resid = None
        if self.system.resid is not None:
            resid = self.system.resid

        bond_index = None
        if self.system.bond_index is not None:
            bond_index = self.system.bond_index[0] + 1

        self.file = h5py.File(self.filename, mode)

        self.h5md = self.file.create_group('h5md')
        self.h5md.attrs['version'] = [1, 1]

        self.h5md_author = self.h5md.create_group('author')
        self.h5md_author.attrs['name'] = 'AIMM Group'
        self.h5md_author.attrs['email'] = 'yangyi@szbl.ac.cn'

        self.h5md_creator = self.h5md.create_group('creator')
        self.h5md_creator.attrs['name'] = 'MindSponge'
        self.h5md_creator.attrs['version'] = '2.0'

        species = np.arange(self.num_atoms, dtype=np.int32)

        self.parameters = self.file.create_group('parameters')
        self.vmd_structure = self.parameters.create_group('vmd_structure')
        self.vmd_structure.create_dataset('indexOfSpecies', dtype='int32', data=species, compression=compression,
                                          compression_opts=compression_opts)
        if atomic_number is not None:
            self.vmd_structure.create_dataset('atomicnumber', dtype='int32', data=atomic_number[0],
                                              compression=compression, compression_opts=compression_opts)
            self.vmd_structure.create_dataset('radius', dtype='float32', data=atom_raidus, compression=compression,
                                              compression_opts=compression_opts)
        if atom_name is not None:
            self.vmd_structure.create_dataset('name', data=atom_name, compression=compression,
                                              compression_opts=compression_opts)
        if atom_type is not None:
            self.vmd_structure.create_dataset('type', data=atom_type, compression=compression,
                                              compression_opts=compression_opts)

        if resname is not None:
            self.vmd_structure.create_dataset('resname', data=resname, compression=compression,
                                              compression_opts=compression_opts)
            self.vmd_structure.create_dataset('resid', dtype='int32', data=resid, compression=compression,
                                              compression_opts=compression_opts)

        if bond_index is not None:
            self.vmd_structure.create_dataset('bond_from', dtype='int32', data=bond_index[..., 0],
                                              compression=compression, compression_opts=compression_opts)
            self.vmd_structure.create_dataset('bond_to', dtype='int32', data=bond_index[..., 1],
                                              compression=compression, compression_opts=compression_opts)

        self.particles = self.file.create_group('particles')

        self.write_velocity = write_velocity
        self.write_force = write_force

        self.particles_walkers = []
        self.positions = []
        self.velocities = []
        self.forces = []
        self.boxes = []
        shape = (self.num_atoms, self.dimension)
        for i in range(self.num_walkers):
            name = 'walker' + str(i)
            walker = self.particles.create_group(name)

            walker.create_dataset('species', dtype='int32', data=species, compression=compression,
                                  compression_opts=compression_opts)

            position = walker.create_group('position')
            position.create_dataset('step', shape=(0,), dtype='int32', maxshape=(None,), compression=compression,
                                    compression_opts=compression_opts)
            position.create_dataset('time', shape=(0,), dtype='float32', maxshape=(None,), compression=compression,
                                    compression_opts=compression_opts).attrs['unit'] = 'ps'
            position.create_dataset('value', shape=(0,) + shape, dtype='float32', maxshape=(None,) + shape,
                                    compression=compression, compression_opts=compression_opts).attrs['unit'] \
                = self.units.length_unit_name()
            self.positions.append(position)

            box = walker.create_group('box')
            box.attrs['dimension'] = self.dimension
            if self.pbc:
                box.attrs['boundary'] = ['periodic',] * self.dimension
                self.boxes.append(box)
            else:
                box.attrs['boundary'] = ['none',] * self.dimension
            self.boxes.append(box)

            if self.write_velocity:
                velocity = walker.create_group('velocity')
                velocity.create_dataset('step', shape=(0,), dtype='int32', maxshape=(None,), compression=compression,
                                        compression_opts=compression_opts)
                velocity.create_dataset('time', shape=(0,), dtype='float32', maxshape=(None,), compression=compression,
                                        compression_opts=compression_opts).attrs['unit'] = 'ps'
                velocity.create_dataset('value', shape=(0,) + shape, dtype='float32', maxshape=(None,) + shape,
                                        compression=compression, compression_opts=compression_opts).attrs['unit'] = \
                    self.units.velocity_unit()
                velocity = self.velocities.append(velocity)

            if self.write_force:
                force = walker.create_group('force')
                force.create_dataset('step', shape=(0,), dtype='int32', maxshape=(None,), compression=compression,
                                     compression_opts=compression_opts)
                force.create_dataset('time', shape=(0,), dtype='float32', maxshape=(None,), compression=compression,
                                     compression_opts=compression_opts).attrs['unit'] = 'ps'
                force.create_dataset('value', shape=(0,) + shape, dtype='float32', maxshape=(None,) + shape,
                                     compression=compression, compression_opts=compression_opts).attrs['unit'] = \
                    self.units.force_unit()
                force = self.forces.append(force)

            self.particles_walkers.append(walker)

        self.observables = self.file.create_group('observables')

        self.step = 0
        self.time = 0

    def update_group(self, group: h5py.Group, step: int, time: float, value: np.ndarray):
        '''update group'''
        ds_step = group['step']
        ds_step.resize(ds_step.shape[0] + 1, axis=0)
        ds_step[-1] = step

        ds_time = group['time']
        ds_time.resize(ds_time.shape[0] + 1, axis=0)
        ds_time[-1] = time

        ds_value = group['value']
        ds_value.resize(ds_value.shape[0] + 1, axis=0)
        ds_value[-1] = value

        return self

    def __enter__(self):
        """Return the enter target."""
        return self

    def __exit__(self, *err):
        """Release resources here if have any."""

    def begin(self, run_context: RunContext):
        """
        Called once before the network executing.

        Args:
            run_context (RunContext): Include some information of the model.
        """

    def epoch_begin(self, run_context: RunContext):
        """
        Called before each epoch beginning.

        Args:
            run_context (RunContext): Include some information of the model.
        """

    def epoch_end(self, run_context: RunContext):
        """
        Called after each epoch finished.

        Args:
            run_context (RunContext): Include some information of the model.
        """

    def step_begin(self, run_context: RunContext):
        """
        Called before each step beginning.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        cb_params = run_context.original_args()
        self.step = cb_params.cur_step
        self.time = cb_params.cur_time
        coordinates = cb_params.coordinates.asnumpy() * self.length_unit_scale
        for i in range(self.num_walkers):
            self.update_group(self.positions[i], self.step, self.time, coordinates[i])

    def step_end(self, run_context: RunContext):
        """
        Called after each step finished.

        Args:
            run_context (RunContext): Include some information of the model.
        """

        cb_params = run_context.original_args()
        if self.write_velocity:
            velocities = cb_params.velocities[0].asnumpy() * self.length_unit_scale
            for i in range(self.num_walkers):
                self.update_group(self.velocities[i], self.step, self.time, velocities[i])
        if self.write_force:
            forces = cb_params.forces.asnumpy() * self.force_unit_scale
            for i in range(self.num_walkers):
                self.update_group(self.forces[i], self.step, self.time, forces[i])

    def end(self, run_context: RunContext):
        """
        Called once after network training.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        self.file.close()
