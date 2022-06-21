# Copyright 2021 Huawei Technologies Co., Ltd
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
'''main'''
import argparse
import os
import time
import warnings
import numpy as np

import mindspore as ms
from mindspore import context, nn
from mindspore import numpy as msnp
from mindspore import ops
from mindspore.common import Tensor
from mindsponge.common.callback import RunInfo
from mindsponge.common.units import set_global_units
from mindsponge.core.control.integrator.minimize import GradientDescent
from mindsponge.core.loss import get_violation_loss
from mindsponge.core.partition.neighbourlist import NeighbourList
from mindsponge.core.potential.energy import (AngleEnergy, BondEnergy,
                                              DihedralEnergy, NB14Energy,
                                              NonBondEnergy)
from mindsponge.core.potential.forcefield import ClassicalFF, Oscillator
from mindsponge.core.simulation import SimulationCell
from mindsponge.core.simulation.onestep import RunOneStepCell
from mindsponge.core.simulation.sponge import Sponge
from mindsponge.core.space.system import SystemCell
from mindsponge.data.hyperparam import ReconstructProtein as Protein
from mindsponge.data.pdb_generator import gen_pdb
from mindsponge.data.pdb_parser import read_pdb_via_xponge as read_pdb

os.environ['GLOG_v'] = '3'
warnings.filterwarnings("ignore")
context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id=1)

parser = argparse.ArgumentParser()
parser.add_argument("-i", help="Set the input pdb file path.")
parser.add_argument("-o", help="Set the output pdb file path.")
parser.add_argument("-addh", help="Set to 1 if need to add H atoms, default to be 1..", default=1)
args = parser.parse_args()
pdb_name = args.i
save_pdb_name = args.o
addh = args.addh

ms.set_seed(2333)
_, res_names, _, _, res_pointer, flatten_atoms, flatten_crds, init_res_names, init_res_ids, \
residue_index, aatype, atom14_positions, atom14_atom_exists, residx_atom14_to_atom37 = read_pdb(pdb_name, addh)

pdb_cell = Protein(res_names, res_pointer, flatten_atoms, init_res_names=init_res_names, init_res_ids=init_res_ids)

coordinates = flatten_crds
atomic_number = pdb_cell.atomic_numbers
nonh_mask = Tensor(np.where(atomic_number > 1, 0, 1)[None, :, None], ms.int32)
atom_name = pdb_cell.atom_names
atom_type = pdb_cell.atom_types
resname = pdb_cell.res_names
resid = pdb_cell.res_id
init_resname = pdb_cell.init_res_names
init_resid = pdb_cell.init_res_ids
mass = pdb_cell.mass
charge = pdb_cell.charge
crd_mapping_ids = pdb_cell.crd_mapping_ids
bond_index = pdb_cell.bond_index
bond_params = pdb_cell.bond_params
rk_init = bond_params[:, 2]
req_init = bond_params[:, 3]
angle_index = pdb_cell.angle_index
angle_params = pdb_cell.angle_params
tk_init = angle_params[:, 3]
teq_init = angle_params[:, 4]
dihedral_index = pdb_cell.dihedral_params[:, [0, 1, 2, 3]]
idihedral_index = pdb_cell.idihedral_params[:, [0, 1, 2, 3]]
dihedral_index = ops.Cast()(msnp.vstack((dihedral_index, idihedral_index)), ms.int32)
dihedral_params = pdb_cell.dihedral_params
idihedral_params = pdb_cell.idihedral_params
dihedral_params = msnp.vstack((dihedral_params, idihedral_params))
pk_init = dihedral_params[:, 5]
pn_init = ops.Cast()(dihedral_params[:, 4], ms.int32)
phase_init = dihedral_params[:, 6]
vdw_params = pdb_cell.vdw_params
atomic_radius = vdw_params[:, 0]
well_depth = vdw_params[:, 1]
exclude_index = pdb_cell.excludes_index
nb14_index = pdb_cell.nb14_index
one_scee = np.array([5 / 6] * nb14_index.shape[-2])
one_scnb = np.array([.5] * nb14_index.shape[-2])
set_global_units('A', 'kcal/mol')
num_atoms = atom_type.shape[-1]
coordinates = Tensor(coordinates, ms.float32)


def main():
    num_walkers = 1

    violation_loss = get_violation_loss(atom14_atom_exists=atom14_atom_exists, residue_index=residue_index,
                                        residx_atom14_to_atom37=residx_atom14_to_atom37,
                                        atom14_positions=atom14_positions, aatype=aatype)

    system = SystemCell(num_walkers=num_walkers, num_atoms=num_atoms, atomic_number=atomic_number, atom_name=atom_name,
                        atom_type=atom_type, resname=resname, resid=init_resid - 1, bond_index=bond_index, mass=mass,
                        coordinates=coordinates, charge=charge)

    bond_energy = BondEnergy(bond_index, rk_init=rk_init, req_init=req_init, scale=1, pbc=False)
    angle_energy = AngleEnergy(angle_index, tk_init=tk_init, teq_init=teq_init, scale=1, pbc=False)
    dihedral_energy = DihedralEnergy(dihedral_index, pk_init=pk_init, pn_init=pn_init, phase_init=phase_init, scale=1,
                                     pbc=False)
    nonbond_energy = NonBondEnergy(num_atoms, charge=system.charge, atomic_radius=atomic_radius, well_depth=well_depth)
    nb14_energy = NB14Energy(nb14_index, nonbond_energy, one_scee=one_scee, one_scnb=one_scnb)
    beg_time = time.time()

    # First Step Pipeline
    def first_try(system, gds, loops, ads, adm):
        energy = ClassicalFF(
            bond_energy=bond_energy,
            angle_energy=angle_energy,
            dihedral_energy=dihedral_energy,
            nonbond_energy=nonbond_energy,
            nb14_energy=nb14_energy,
        )
        neighbour_list = NeighbourList(system, cutoff=None, exclude_index=exclude_index)
        simulation_network = SimulationCell(system, energy, neighbour_list=neighbour_list)
        total_step = gds
        learning_rate = 1e-07
        factor = 1.003
        opt = GradientDescent(system.trainable_params(), learning_rate=learning_rate, factor=factor,
                              nonh_mask=nonh_mask)
        for i, param in enumerate(opt.trainable_params()):
            print(i, param.name, param.shape)
        onestep = RunOneStepCell(simulation_network, opt, sens=1)
        md = Sponge(onestep)
        run_info = RunInfo(system,
                           get_vloss=False,
                           atom14_atom_exists=atom14_atom_exists,
                           residue_index=residue_index,
                           residx_atom14_to_atom37=residx_atom14_to_atom37,
                           aatype=aatype,
                           crd_mapping_masks=init_res_ids - 1,
                           crd_mapping_ids=crd_mapping_ids,
                           nonh_mask=nonh_mask
                           )
        md.run(total_step, callbacks=[run_info])

        if msnp.isnan(md.energy().sum()):
            return 0

        for _ in range(loops):
            k_coe = 10
            harmonic_energy = Oscillator(1 * system.coordinates, k_coe, nonh_mask)

            total_step = ads
            learning_rate = 5e-02

            energy = ClassicalFF(
                bond_energy=bond_energy,
                angle_energy=angle_energy,
                dihedral_energy=dihedral_energy,
                nonbond_energy=nonbond_energy,
                nb14_energy=nb14_energy,
                harmonic_energy=harmonic_energy,
            )

            simulation_network = SimulationCell(system, energy, neighbour_list=neighbour_list)
            for _ in range(adm):
                opt = nn.Adam(system.trainable_params(), learning_rate=learning_rate)
                for i, param in enumerate(opt.trainable_params()):
                    print(i, param.name, param.shape)
                onestep = RunOneStepCell(simulation_network, opt, sens=1)
                md = Sponge(onestep)
                print(md.energy())
                run_info = RunInfo(system,
                                   get_vloss=False,
                                   atom14_atom_exists=atom14_atom_exists,
                                   residue_index=residue_index,
                                   residx_atom14_to_atom37=residx_atom14_to_atom37,
                                   aatype=aatype,
                                   crd_mapping_masks=init_res_ids - 1,
                                   crd_mapping_ids=crd_mapping_ids,
                                   nonh_mask=nonh_mask
                                   )
                md.run(total_step, callbacks=[run_info])
                if msnp.isnan(md.energy().sum()):
                    return 0

            energy = ClassicalFF(
                bond_energy=bond_energy,
                angle_energy=angle_energy,
                dihedral_energy=dihedral_energy,
                harmonic_energy=harmonic_energy,
            )

            simulation_network = SimulationCell(system, energy, neighbour_list=neighbour_list)
            for _ in range(adm):
                opt = nn.Adam(system.trainable_params(), learning_rate=learning_rate)
                for i, param in enumerate(opt.trainable_params()):
                    print(i, param.name, param.shape)
                onestep = RunOneStepCell(simulation_network, opt, sens=1)
                md = Sponge(onestep)
                print(md.energy())
                run_info = RunInfo(system,
                                   get_vloss=False,
                                   atom14_atom_exists=atom14_atom_exists,
                                   residue_index=residue_index,
                                   residx_atom14_to_atom37=residx_atom14_to_atom37,
                                   aatype=aatype,
                                   crd_mapping_masks=init_res_ids - 1,
                                   crd_mapping_ids=crd_mapping_ids,
                                   nonh_mask=nonh_mask
                                   )
                md.run(total_step, callbacks=[run_info])
                if msnp.isnan(md.energy().sum()):
                    return 0

        return system

    # Second Step Pipeline
    def second_try(system, gds, loops, ads, adm):
        energy = ClassicalFF(
            bond_energy=bond_energy,
            angle_energy=angle_energy,
            dihedral_energy=dihedral_energy,
            nonbond_energy=nonbond_energy,
            nb14_energy=nb14_energy,
        )
        neighbour_list = NeighbourList(system, cutoff=None, exclude_index=exclude_index)
        simulation_network = SimulationCell(system, energy, neighbour_list=neighbour_list)
        total_step = gds
        learning_rate = 1e-07
        factor = 1.003
        opt = GradientDescent(system.trainable_params(), learning_rate=learning_rate, factor=factor,
                              nonh_mask=nonh_mask)
        for i, param in enumerate(opt.trainable_params()):
            print(i, param.name, param.shape)
        onestep = RunOneStepCell(simulation_network, opt, sens=1)
        md = Sponge(onestep)
        run_info = RunInfo(system,
                           get_vloss=False,
                           atom14_atom_exists=atom14_atom_exists,
                           residue_index=residue_index,
                           residx_atom14_to_atom37=residx_atom14_to_atom37,
                           aatype=aatype,
                           crd_mapping_masks=init_res_ids - 1,
                           crd_mapping_ids=crd_mapping_ids,
                           nonh_mask=nonh_mask
                           )
        md.run(total_step, callbacks=[run_info])

        if msnp.isnan(md.energy().sum()):
            return 0

        for _ in range(loops):
            k_coe = 10
            harmonic_energy = Oscillator(1 * system.coordinates, k_coe, nonh_mask)

            total_step = ads
            learning_rate = 5e-02

            energy = ClassicalFF(
                bond_energy=bond_energy,
                angle_energy=angle_energy,
                dihedral_energy=dihedral_energy,
                nonbond_energy=nonbond_energy,
                nb14_energy=nb14_energy,
                harmonic_energy=harmonic_energy,
            )

            simulation_network = SimulationCell(system, energy, neighbour_list=neighbour_list)
            for _ in range(adm):
                opt = nn.Adam(system.trainable_params(), learning_rate=learning_rate)
                for i, param in enumerate(opt.trainable_params()):
                    print(i, param.name, param.shape)
                onestep = RunOneStepCell(simulation_network, opt, sens=1)
                md = Sponge(onestep)
                print(md.energy())
                run_info = RunInfo(system,
                                   get_vloss=False,
                                   atom14_atom_exists=atom14_atom_exists,
                                   residue_index=residue_index,
                                   residx_atom14_to_atom37=residx_atom14_to_atom37,
                                   aatype=aatype,
                                   crd_mapping_masks=init_res_ids - 1,
                                   crd_mapping_ids=crd_mapping_ids,
                                   nonh_mask=nonh_mask
                                   )
                md.run(total_step, callbacks=[run_info])
                if msnp.isnan(md.energy().sum()):
                    return 0

        return system

    # Third Step Pipeline
    def third_try(system, gds, loops, ads, adm):
        energy = ClassicalFF(
            bond_energy=bond_energy,
            angle_energy=angle_energy,
            dihedral_energy=dihedral_energy,
            nonbond_energy=nonbond_energy,
            nb14_energy=nb14_energy,
        )
        neighbour_list = NeighbourList(system, cutoff=None, exclude_index=exclude_index)
        simulation_network = SimulationCell(system, energy, neighbour_list=neighbour_list)
        total_step = gds
        learning_rate = 1e-07
        factor = 1.003
        opt = GradientDescent(system.trainable_params(), learning_rate=learning_rate, factor=factor,
                              nonh_mask=nonh_mask)
        for i, param in enumerate(opt.trainable_params()):
            print(i, param.name, param.shape)
        onestep = RunOneStepCell(simulation_network, opt, sens=1)
        md = Sponge(onestep)
        run_info = RunInfo(system,
                           get_vloss=False,
                           atom14_atom_exists=atom14_atom_exists,
                           residue_index=residue_index,
                           residx_atom14_to_atom37=residx_atom14_to_atom37,
                           aatype=aatype,
                           crd_mapping_masks=init_res_ids - 1,
                           crd_mapping_ids=crd_mapping_ids,
                           nonh_mask=nonh_mask
                           )
        md.run(total_step, callbacks=[run_info])

        if msnp.isnan(md.energy().sum()):
            return 0

        for _ in range(loops):
            k_coe = 10
            harmonic_energy = Oscillator(1 * system.coordinates, k_coe, nonh_mask)

            total_step = ads
            learning_rate = 5e-02

            energy = ClassicalFF(
                bond_energy=bond_energy,
                angle_energy=angle_energy,
                dihedral_energy=dihedral_energy,
                harmonic_energy=harmonic_energy,
            )

            simulation_network = SimulationCell(system, energy, neighbour_list=neighbour_list)
            for _ in range(adm):
                opt = nn.Adam(system.trainable_params(), learning_rate=learning_rate)
                for i, param in enumerate(opt.trainable_params()):
                    print(i, param.name, param.shape)
                onestep = RunOneStepCell(simulation_network, opt, sens=1)
                md = Sponge(onestep)
                print(md.energy())
                run_info = RunInfo(system,
                                   get_vloss=False,
                                   atom14_atom_exists=atom14_atom_exists,
                                   residue_index=residue_index,
                                   residx_atom14_to_atom37=residx_atom14_to_atom37,
                                   aatype=aatype,
                                   crd_mapping_masks=init_res_ids - 1,
                                   crd_mapping_ids=crd_mapping_ids,
                                   nonh_mask=nonh_mask
                                   )
                md.run(total_step, callbacks=[run_info])
                if msnp.isnan(md.energy().sum()):
                    return 0

        return system

    gds, loops, ads, adm = 100, 3, 200, 2
    system = first_try(system, gds, loops, ads, adm)
    try:
        gen_pdb(system.coordinates.asnumpy(), atom_name, init_resname, init_resid, pdb_name=save_pdb_name)
        violation_loss = get_violation_loss(save_pdb_name)
        print('The first try violation loss value is: {}'.format(violation_loss))
    except AttributeError:
        pass

    while system == 0:
        system = SystemCell(num_walkers=num_walkers, num_atoms=num_atoms, atomic_number=atomic_number,
                            atom_name=atom_name,
                            atom_type=atom_type, resname=resname, resid=init_resid - 1, bond_index=bond_index,
                            mass=mass,
                            coordinates=coordinates, charge=charge)
        gds = int(0.5 * gds)
        ads = int(0.8 * ads)
        system = first_try(system, gds, loops, ads, adm)
        try:
            gen_pdb(system.coordinates.asnumpy(), atom_name, init_resname, init_resid, pdb_name=save_pdb_name)
            violation_loss = get_violation_loss(save_pdb_name)
            print('The first try violation loss value is: {}'.format(violation_loss))
        except AttributeError:
            continue

    if violation_loss > 0:
        gds = 200
        system = SystemCell(num_walkers=num_walkers, num_atoms=num_atoms, atomic_number=atomic_number,
                            atom_name=atom_name, atom_type=atom_type, resname=resname, resid=init_resid - 1,
                            bond_index=bond_index, mass=mass, coordinates=coordinates, charge=charge)
        loops, ads, adm = 6, 200, 1
        system = second_try(system, gds, loops, ads, adm)

        gen_pdb(system.coordinates.asnumpy(), atom_name, init_resname, init_resid, pdb_name=save_pdb_name)
        violation_loss = get_violation_loss(save_pdb_name)
        print('The second try violation loss value is: {}'.format(violation_loss))

    if violation_loss > 0:
        gds = 200
        system = SystemCell(num_walkers=num_walkers, num_atoms=num_atoms, atomic_number=atomic_number,
                            atom_name=atom_name, atom_type=atom_type, resname=resname, resid=init_resid - 1,
                            bond_index=bond_index, mass=mass, coordinates=coordinates, charge=charge)
        loops, ads, adm = 6, 200, 1
        system = third_try(system, gds, loops, ads, adm)

        gen_pdb(system.coordinates.asnumpy(), atom_name, init_resname, init_resid, pdb_name=save_pdb_name)
        violation_loss = get_violation_loss(save_pdb_name)
        print('The third try violation loss value is: {}'.format(violation_loss))

    if violation_loss > 0:
        gds = 100
        system = SystemCell(num_walkers=num_walkers, num_atoms=num_atoms, atomic_number=atomic_number,
                            atom_name=atom_name, atom_type=atom_type, resname=resname, resid=init_resid - 1,
                            bond_index=bond_index, mass=mass, coordinates=coordinates, charge=charge)
        loops, ads, adm = 8, 100, 1
        system = third_try(system, gds, loops, ads, adm)

        gen_pdb(system.coordinates.asnumpy(), atom_name, init_resname, init_resid, pdb_name=save_pdb_name)
        violation_loss = get_violation_loss(save_pdb_name)
        print('The forth try violation loss value is: {}'.format(violation_loss))

    if violation_loss > 0:
        system = SystemCell(num_walkers=num_walkers, num_atoms=num_atoms, atomic_number=atomic_number,
                            atom_name=atom_name, atom_type=atom_type, resname=resname, resid=init_resid - 1,
                            bond_index=bond_index, mass=mass, coordinates=coordinates, charge=charge)
        gds, loops, ads, adm = 30, 2, 150, 2
        system = first_try(system, gds, loops, ads, adm)

        gen_pdb(system.coordinates.asnumpy(), atom_name, init_resname, init_resid, pdb_name=save_pdb_name)
        violation_loss = get_violation_loss(save_pdb_name)
        print('The final try violation loss value is: {}'.format(violation_loss))

    end_time = time.time()
    used_time = end_time - beg_time
    m, s = divmod(used_time, 60)
    h, m = divmod(m, 60)

    print("Run Time: %02d:%02d:%02d" % (h, m, s))


try:
    main()
except RuntimeError as e:
    import traceback

    traceback.print_exc()
    print('MindSponge relax pipeline running failed, please try it again!')
