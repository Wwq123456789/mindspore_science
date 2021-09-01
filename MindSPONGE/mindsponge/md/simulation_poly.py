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
'''Simulation'''
import numpy as np

import mindspore.common.dtype as mstype
from mindspore import Tensor
from mindspore import nn
from mindspore.common.parameter import Parameter
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindsponge import Angle
from mindsponge import Bond
from mindsponge import Dihedral
from mindsponge import LennardJonesInformation
from mindsponge import NonBond14
from mindsponge import ParticleMeshEwald
from mindsponge import NeighborList
from mindsponge import MdInformation
from mindsponge import LangevinLiujian


class Controller:
    '''controller'''

    def __init__(self, args_opt):
        self.input_file = args_opt.i
        self.initial_coordinates_file = args_opt.c
        self.amber_parm = args_opt.amber_parm
        self.restrt = args_opt.r
        self.mdcrd = args_opt.x
        self.mdout = args_opt.o
        self.mdbox = args_opt.box

        self.command_set = {}
        self.md_task = None
        self.commands_from_in_file()
        self.punctuation = ","

    def commands_from_in_file(self):
        '''command from in file'''
        file = open(self.input_file, 'r')
        context = file.readlines()
        file.close()
        self.md_task = context[0].strip()
        for val in context:
            val = val.strip()
            if val and val[0] != '#' and ("=" in val):
                val = val[:val.index(",")] if ',' in val else val
                assert len(val.strip().split("=")) == 2
                flag, value = val.strip().split("=")
                value = value.replace(" ", "")
                flag = flag.replace(" ", "")
                if flag not in self.command_set:
                    self.command_set[flag] = value
                else:
                    print("ERROR COMMAND FILE")


class Simulation(nn.Cell):
    '''simulation'''

    def __init__(self, args_opt):
        super(Simulation, self).__init__()
        self.control = Controller(args_opt)
        self.md_info = MdInformation(self.control)
        self.mode = self.md_info.mode
        self.bond = Bond(self.control)
        self.angle = Angle(self.control)
        self.dihedral = Dihedral(self.control)
        self.nb14 = NonBond14(self.control, self.dihedral, self.md_info.atom_numbers)
        self.nb_info = NeighborList(self.control, self.md_info.atom_numbers, self.md_info.box_length)
        self.lj_info = LennardJonesInformation(self.control, self.md_info.nb.cutoff, self.md_info.sys.box_length)
        self.liujian_info = LangevinLiujian(self.control, self.md_info.atom_numbers)
        self.pme_method = ParticleMeshEwald(self.control, self.md_info)
        self.bond_energy_sum = Tensor(0, mstype.int32)
        self.angle_energy_sum = Tensor(0, mstype.int32)
        self.dihedral_energy_sum = Tensor(0, mstype.int32)
        self.nb14_lj_energy_sum = Tensor(0, mstype.int32)
        self.nb14_cf_energy_sum = Tensor(0, mstype.int32)
        self.lj_energy_sum = Tensor(0, mstype.int32)
        self.ee_ene = Tensor(0, mstype.int32)
        self.total_energy = Tensor(0, mstype.int32)
        # Init scalar
        self.ntwx = self.md_info.ntwx
        self.atom_numbers = self.md_info.atom_numbers
        self.residue_numbers = self.md_info.residue_numbers
        self.bond_numbers = self.bond.bond_numbers
        self.angle_numbers = self.angle.angle_numbers
        self.dihedral_numbers = self.dihedral.dihedral_numbers
        self.nb14_numbers = self.nb14.nb14_numbers
        self.nxy = self.nb_info.Nxy
        self.grid_numbers = self.nb_info.grid_numbers
        self.max_atom_in_grid_numbers = self.nb_info.max_atom_in_grid_numbers
        self.max_neighbor_numbers = self.nb_info.max_neighbor_numbers
        self.excluded_atom_numbers = self.md_info.nb.excluded_atom_numbers
        self.refresh_count = Parameter(Tensor(self.nb_info.refresh_count, mstype.int32), requires_grad=False)
        self.refresh_interval = self.nb_info.refresh_interval
        self.skin = self.nb_info.skin
        self.cutoff = self.nb_info.cutoff
        self.cutoff_square = self.nb_info.cutoff_square
        self.cutoff_with_skin = self.nb_info.cutoff_with_skin
        self.half_cutoff_with_skin = self.nb_info.half_cutoff_with_skin
        self.cutoff_with_skin_square = self.nb_info.cutoff_with_skin_square
        self.half_skin_square = self.nb_info.half_skin_square
        self.beta = self.pme_method.beta
        self.fftx = self.pme_method.fftx
        self.ffty = self.pme_method.ffty
        self.fftz = self.pme_method.fftz
        self.random_seed = self.liujian_info.random_seed
        self.dt = self.liujian_info.dt
        self.half_dt = self.liujian_info.half_dt
        self.exp_gamma = self.liujian_info.exp_gamma
        self.init_tensor()
        self.op_define()
        self.update = False

    def init_tensor(self):
        '''init tensor'''
        self.uint_crd = Parameter(Tensor(np.zeros([self.atom_numbers, 3], dtype=np.uint32), mstype.uint32),
                                  requires_grad=False)
        self.crd = Parameter(
            Tensor(np.array(self.md_info.coordinate).reshape([self.atom_numbers, 3]), mstype.float32),
            requires_grad=False)
        self.crd_to_uint_crd_cof = Tensor(np.asarray(self.md_info.pbc.crd_to_uint_crd_cof, np.float32), mstype.float32)
        self.uint_dr_to_dr_cof = Parameter(Tensor(self.md_info.pbc.uint_dr_to_dr_cof, mstype.float32),
                                           requires_grad=False)
        self.box_length = Tensor(self.md_info.box_length, mstype.float32)
        self.charge = Parameter(Tensor(np.asarray(self.md_info.h_charge, dtype=np.float32), mstype.float32),
                                requires_grad=False)
        self.old_crd = Parameter(Tensor(np.zeros([self.atom_numbers, 3], dtype=np.float32), mstype.float32),
                                 requires_grad=False)
        self.last_crd = Parameter(Tensor(np.zeros([self.atom_numbers, 3], dtype=np.float32), mstype.float32),
                                  requires_grad=False)
        self.mass = Tensor(self.md_info.h_mass, mstype.float32)
        self.mass_inverse = Tensor(self.md_info.h_mass_inverse, mstype.float32)
        self.res_start = Tensor(self.md_info.h_res_start, mstype.int32)
        self.res_end = Tensor(self.md_info.h_res_end, mstype.int32)
        self.velocity = Parameter(Tensor(self.md_info.velocity, mstype.float32), requires_grad=False)
        self.acc = Parameter(Tensor(np.zeros([self.atom_numbers, 3], np.float32), mstype.float32), requires_grad=False)
        self.bond_atom_a = Tensor(np.asarray(self.bond.h_atom_a, np.int32), mstype.int32)
        self.bond_atom_b = Tensor(np.asarray(self.bond.h_atom_b, np.int32), mstype.int32)
        self.bond_k = Tensor(np.asarray(self.bond.h_k, np.float32), mstype.float32)
        self.bond_r0 = Tensor(np.asarray(self.bond.h_r0, np.float32), mstype.float32)
        self.angle_atom_a = Tensor(np.asarray(self.angle.h_atom_a, np.int32), mstype.int32)
        self.angle_atom_b = Tensor(np.asarray(self.angle.h_atom_b, np.int32), mstype.int32)
        self.angle_atom_c = Tensor(np.asarray(self.angle.h_atom_c, np.int32), mstype.int32)
        self.angle_k = Tensor(np.asarray(self.angle.h_angle_k, np.float32), mstype.float32)
        self.angle_theta0 = Tensor(np.asarray(self.angle.h_angle_theta0, np.float32), mstype.float32)
        self.dihedral_atom_a = Tensor(np.asarray(self.dihedral.h_atom_a, np.int32), mstype.int32)
        self.dihedral_atom_b = Tensor(np.asarray(self.dihedral.h_atom_b, np.int32), mstype.int32)
        self.dihedral_atom_c = Tensor(np.asarray(self.dihedral.h_atom_c, np.int32), mstype.int32)
        self.dihedral_atom_d = Tensor(np.asarray(self.dihedral.h_atom_d, np.int32), mstype.int32)
        self.pk = Tensor(np.asarray(self.dihedral.h_pk, np.float32), mstype.float32)
        self.gamc = Tensor(np.asarray(self.dihedral.h_gamc, np.float32), mstype.float32)
        self.gams = Tensor(np.asarray(self.dihedral.h_gams, np.float32), mstype.float32)
        self.pn = Tensor(np.asarray(self.dihedral.h_pn, np.float32), mstype.float32)
        self.ipn = Tensor(np.asarray(self.dihedral.h_ipn, np.int32), mstype.int32)
        self.nb14_atom_a = Tensor(np.asarray(self.nb14.h_atom_a, np.int32), mstype.int32)
        self.nb14_atom_b = Tensor(np.asarray(self.nb14.h_atom_b, np.int32), mstype.int32)
        self.lj_scale_factor = Tensor(np.asarray(self.nb14.h_lj_scale_factor, np.float32), mstype.float32)
        self.cf_scale_factor = Tensor(np.asarray(self.nb14.h_cf_scale_factor, np.float32), mstype.float32)
        self.grid_n = Tensor(self.nb_info.grid_N, mstype.int32)
        self.grid_length_inverse = Parameter(Tensor(self.nb_info.grid_length_inverse, mstype.float32),
                                             requires_grad=False)
        self.bucket = Parameter(Tensor(
            np.asarray(self.nb_info.bucket, np.int32).reshape([self.grid_numbers, self.max_atom_in_grid_numbers]),
            mstype.int32), requires_grad=False)
        self.atom_numbers_in_grid_bucket = Parameter(Tensor(self.nb_info.atom_numbers_in_grid_bucket, mstype.int32),
                                                     requires_grad=False)
        self.atom_in_grid_serial = Parameter(Tensor(np.zeros([self.nb_info.atom_numbers,], np.int32), mstype.int32),
                                             requires_grad=False)
        self.pointer = Parameter(
            Tensor(np.asarray(self.nb_info.pointer, np.int32).reshape([self.grid_numbers, 125]), mstype.int32),
            requires_grad=False)
        self.nl_atom_numbers = Parameter(Tensor(np.zeros([self.atom_numbers,], np.int32), mstype.int32),
                                         requires_grad=False)
        self.nl_atom_serial = Parameter(
            Tensor(np.zeros([self.atom_numbers, self.max_neighbor_numbers], np.int32), mstype.int32),
            requires_grad=False)
        self.excluded_list_start = Tensor(np.asarray(self.md_info.nb.h_excluded_list_start, np.int32), mstype.int32)
        self.excluded_list = Tensor(np.asarray(self.md_info.nb.h_excluded_list, np.int32), mstype.int32)
        self.excluded_numbers = Tensor(np.asarray(self.md_info.nb.h_excluded_numbers, np.int32), mstype.int32)

        self.need_refresh_flag = Tensor(np.asarray([0], np.int32), mstype.int32)
        self.atom_lj_type = Tensor(self.lj_info.atom_LJ_type, mstype.int32)
        self.lj_a = Tensor(self.lj_info.h_LJ_A, mstype.float32)
        self.lj_b = Tensor(self.lj_info.h_LJ_B, mstype.float32)
        self.sqrt_mass = Tensor(self.liujian_info.h_sqrt_mass, mstype.float32)
        self.rand_state = Parameter(Tensor(self.liujian_info.rand_state, mstype.float32))
        self.zero_fp_tensor = Tensor(np.asarray([0,], np.float32))

    def op_define(self):
        '''op define'''
        self.crd_to_uint_crd = P.CrdToUintCrd(self.atom_numbers)
        self.mdtemp = P.MDTemperature(self.residue_numbers, self.atom_numbers)
        self.setup_random_state = P.MDIterationSetupRandState(self.atom_numbers, self.random_seed)
        self.bond_force_with_atom_energy = P.BondForceWithAtomEnergy(bond_numbers=self.bond_numbers,
                                                                     atom_numbers=self.atom_numbers)
        self.angle_force_with_atom_energy = P.AngleForceWithAtomEnergy(angle_numbers=self.angle_numbers)
        self.dihedral_force_with_atom_energy = P.DihedralForceWithAtomEnergy(dihedral_numbers=self.dihedral_numbers)
        self.nb14_force_with_atom_energy = P.Dihedral14LJCFForceWithAtomEnergy(nb14_numbers=self.nb14_numbers,
                                                                               atom_numbers=self.atom_numbers)
        self.lj_force_pme_direct_force = P.LJForceWithPMEDirectForce(self.atom_numbers, self.cutoff, self.beta)
        self.pme_excluded_force = P.PMEExcludedForce(atom_numbers=self.atom_numbers,
                                                     excluded_numbers=self.excluded_atom_numbers, beta=self.beta)
        self.pme_reciprocal_force = P.PMEReciprocalForce(self.atom_numbers, self.beta, self.fftx, self.ffty, self.fftz,
                                                         self.md_info.box_length[0], self.md_info.box_length[1],
                                                         self.md_info.box_length[2])
        self.bond_energy = P.BondEnergy(self.bond_numbers, self.atom_numbers)
        self.angle_energy = P.AngleEnergy(self.angle_numbers)
        self.dihedral_energy = P.DihedralEnergy(self.dihedral_numbers)
        self.nb14_lj_energy = P.Dihedral14LJEnergy(self.nb14_numbers, self.atom_numbers)
        self.nb14_cf_energy = P.Dihedral14CFEnergy(self.nb14_numbers, self.atom_numbers)
        self.lj_energy = P.LJEnergy(self.atom_numbers, self.cutoff_square)
        self.pme_energy = P.PMEEnergy(self.atom_numbers, self.excluded_atom_numbers, self.beta, self.fftx, self.ffty,
                                      self.fftz, self.md_info.box_length[0], self.md_info.box_length[1],
                                      self.md_info.box_length[2])
        self.md_iteration_leap_frog_liujian = P.MDIterationLeapFrogLiujian(self.atom_numbers, self.half_dt, self.dt,
                                                                           self.exp_gamma)

        self.neighbor_list_update_init = P.NeighborListUpdate(grid_numbers=self.grid_numbers,
                                                              atom_numbers=self.atom_numbers, not_first_time=0,
                                                              Nxy=self.nxy,
                                                              excluded_atom_numbers=self.excluded_atom_numbers,
                                                              cutoff_square=self.cutoff_square,
                                                              half_skin_square=self.half_skin_square,
                                                              cutoff_with_skin=self.cutoff_with_skin,
                                                              half_cutoff_with_skin=self.half_cutoff_with_skin,
                                                              cutoff_with_skin_square=self.cutoff_with_skin_square,
                                                              refresh_interval=self.refresh_interval,
                                                              cutoff=self.cutoff, skin=self.skin,
                                                              max_atom_in_grid_numbers=self.max_atom_in_grid_numbers,
                                                              max_neighbor_numbers=self.max_neighbor_numbers)

        self.neighbor_list_update = P.NeighborListUpdate(grid_numbers=self.grid_numbers, atom_numbers=self.atom_numbers,
                                                         not_first_time=1, Nxy=self.nxy,
                                                         excluded_atom_numbers=self.excluded_atom_numbers,
                                                         cutoff_square=self.cutoff_square,
                                                         half_skin_square=self.half_skin_square,
                                                         cutoff_with_skin=self.cutoff_with_skin,
                                                         half_cutoff_with_skin=self.half_cutoff_with_skin,
                                                         cutoff_with_skin_square=self.cutoff_with_skin_square,
                                                         refresh_interval=self.refresh_interval, cutoff=self.cutoff,
                                                         skin=self.skin,
                                                         max_atom_in_grid_numbers=self.max_atom_in_grid_numbers,
                                                         max_neighbor_numbers=self.max_neighbor_numbers)
        self.random_force = Tensor(np.zeros([self.atom_numbers, 3], np.float32), mstype.float32)

    def simulation_beforce_caculate_force(self):
        '''simulation before calculate force'''
        crd_to_uint_crd_cof = 0.5 * self.crd_to_uint_crd_cof
        uint_crd = self.crd_to_uint_crd(crd_to_uint_crd_cof, self.crd)
        return uint_crd

    def simulation_caculate_force(self, uint_crd, scaler, nl_atom_numbers, nl_atom_serial):
        '''simulation calculate force'''
        bond_force, _ = self.bond_force_with_atom_energy(uint_crd, scaler, self.bond_atom_a,
                                                         self.bond_atom_b, self.bond_k, self.bond_r0)

        angle_force, _ = self.angle_force_with_atom_energy(uint_crd, scaler, self.angle_atom_a,
                                                           self.angle_atom_b, self.angle_atom_c,
                                                           self.angle_k, self.angle_theta0)

        dihedral_force, _ = self.dihedral_force_with_atom_energy(uint_crd, scaler,
                                                                 self.dihedral_atom_a,
                                                                 self.dihedral_atom_b,
                                                                 self.dihedral_atom_c,
                                                                 self.dihedral_atom_d, self.ipn,
                                                                 self.pk, self.gamc, self.gams,
                                                                 self.pn)

        nb14_force, _ = self.nb14_force_with_atom_energy(uint_crd, self.atom_lj_type, self.charge,
                                                         scaler, self.nb14_atom_a, self.nb14_atom_b,
                                                         self.lj_scale_factor, self.cf_scale_factor,
                                                         self.lj_a, self.lj_b)

        lj_force = self.lj_force_pme_direct_force(uint_crd, self.atom_lj_type, self.charge, scaler, nl_atom_numbers,
                                                  nl_atom_serial, self.lj_a, self.lj_b)
        pme_excluded_force = self.pme_excluded_force(uint_crd, scaler, self.charge, self.excluded_list_start,
                                                     self.excluded_list, self.excluded_numbers)
        pme_reciprocal_force = self.pme_reciprocal_force(uint_crd, self.charge)
        force = P.AddN()(
            [bond_force, angle_force, dihedral_force, nb14_force, lj_force, pme_excluded_force, pme_reciprocal_force])
        return force

    def simulation_caculate_energy(self, uint_crd, uint_dr_to_dr_cof):
        '''simulation calculate energy'''
        bond_energy = self.bond_energy(uint_crd, uint_dr_to_dr_cof, self.bond_atom_a, self.bond_atom_b, self.bond_k,
                                       self.bond_r0)
        bond_energy_sum = P.ReduceSum(True)(bond_energy)

        angle_energy = self.angle_energy(uint_crd, uint_dr_to_dr_cof, self.angle_atom_a, self.angle_atom_b,
                                         self.angle_atom_c, self.angle_k, self.angle_theta0)
        angle_energy_sum = P.ReduceSum(True)(angle_energy)

        dihedral_energy = self.dihedral_energy(uint_crd, uint_dr_to_dr_cof, self.dihedral_atom_a, self.dihedral_atom_b,
                                               self.dihedral_atom_c, self.dihedral_atom_d, self.ipn, self.pk, self.gamc,
                                               self.gams, self.pn)
        dihedral_energy_sum = P.ReduceSum(True)(dihedral_energy)

        nb14_lj_energy = self.nb14_lj_energy(uint_crd, self.atom_lj_type, self.charge, uint_dr_to_dr_cof,
                                             self.nb14_atom_a, self.nb14_atom_b, self.lj_scale_factor, self.lj_a,
                                             self.lj_b)
        nb14_cf_energy = self.nb14_cf_energy(uint_crd, self.atom_lj_type, self.charge, uint_dr_to_dr_cof,
                                             self.nb14_atom_a, self.nb14_atom_b, self.cf_scale_factor)
        nb14_lj_energy_sum = P.ReduceSum(True)(nb14_lj_energy)
        nb14_cf_energy_sum = P.ReduceSum(True)(nb14_cf_energy)

        lj_energy = self.lj_energy(uint_crd, self.atom_lj_type, self.charge, uint_dr_to_dr_cof, self.nl_atom_numbers,
                                   self.nl_atom_serial, self.lj_a, self.lj_b)
        lj_energy_sum = P.ReduceSum(True)(lj_energy)

        reciprocal_energy, self_energy, direct_energy, correction_energy = self.pme_energy(uint_crd, self.charge,
                                                                                           self.nl_atom_numbers,
                                                                                           self.nl_atom_serial,
                                                                                           uint_dr_to_dr_cof,
                                                                                           self.excluded_list_start,
                                                                                           self.excluded_list,
                                                                                           self.excluded_numbers)
        ee_ene = reciprocal_energy + self_energy + direct_energy + correction_energy
        total_energy = P.AddN()(
            [bond_energy_sum, angle_energy_sum, dihedral_energy_sum, nb14_lj_energy_sum, nb14_cf_energy_sum,
             lj_energy_sum, ee_ene])
        return bond_energy_sum, angle_energy_sum, dihedral_energy_sum, nb14_lj_energy_sum, nb14_cf_energy_sum, \
               lj_energy_sum, ee_ene, total_energy

    def simulation_temperature(self):
        '''caculate temperature'''
        res_ek_energy = self.mdtemp(self.res_start, self.res_end, self.velocity, self.mass)
        temperature = P.ReduceSum()(res_ek_energy)
        return temperature

    def simulation_md_iteration_leap_frog_liujian(self, inverse_mass, sqrt_mass_inverse, crd, frc, rand_state,
                                                  random_frc):
        '''simulation leap frog iteration liujian'''
        crd = self.md_iteration_leap_frog_liujian(inverse_mass, sqrt_mass_inverse, self.velocity, crd, frc, self.acc,
                                                  rand_state, random_frc)
        vel = F.depend(self.velocity, crd)
        acc = F.depend(self.acc, crd)
        return vel, crd, acc

    def main_print(self, *args):
        """compute the temperature"""
        steps, temperature, total_potential_energy, sigma_of_bond_ene, sigma_of_angle_ene, sigma_of_dihedral_ene, \
        nb14_lj_energy_sum, nb14_cf_energy_sum, lj_energy_sum, ee_ene = list(args)
        if steps == 0:
            print("_steps_ _TEMP_ _TOT_POT_ENE_ _BOND_ENE_ "
                  "_ANGLE_ENE_ _DIHEDRAL_ENE_ _14LJ_ENE_ _14CF_ENE_ _LJ_ENE_ _CF_PME_ENE_")

        temperature = temperature.asnumpy()
        total_potential_energy = total_potential_energy.asnumpy()
        print("{:>7.0f} {:>7.3f} {:>11.3f}".format(steps, float(temperature), float(total_potential_energy)),
              end=" ")
        if self.bond.bond_numbers > 0:
            sigma_of_bond_ene = sigma_of_bond_ene.asnumpy()
            print("{:>10.3f}".format(float(sigma_of_bond_ene)), end=" ")
        if self.angle.angle_numbers > 0:
            sigma_of_angle_ene = sigma_of_angle_ene.asnumpy()
            print("{:>11.3f}".format(float(sigma_of_angle_ene)), end=" ")
        if self.dihedral.dihedral_numbers > 0:
            sigma_of_dihedral_ene = sigma_of_dihedral_ene.asnumpy()
            print("{:>14.3f}".format(float(sigma_of_dihedral_ene)), end=" ")
        if self.nb14.nb14_numbers > 0:
            nb14_lj_energy_sum = nb14_lj_energy_sum.asnumpy()
            nb14_cf_energy_sum = nb14_cf_energy_sum.asnumpy()
            print("{:>10.3f} {:>10.3f}".format(float(nb14_lj_energy_sum), float(nb14_cf_energy_sum)), end=" ")
        lj_energy_sum = lj_energy_sum.asnumpy()
        ee_ene = ee_ene.asnumpy()
        print("{:>7.3f}".format(float(lj_energy_sum)), end=" ")
        print("{:>12.3f}".format(float(ee_ene)))
        if self.file is not None:
            self.file.write("{:>7.0f} {:>7.3f} {:>11.3f} {:>10.3f} {:>11.3f} {:>14.3f} {:>10.3f} {:>10.3f} {:>7.3f}"
                            " {:>12.3f}\n".format(steps, float(temperature), float(total_potential_energy),
                                                  float(sigma_of_bond_ene), float(sigma_of_angle_ene),
                                                  float(sigma_of_dihedral_ene), float(nb14_lj_energy_sum),
                                                  float(nb14_cf_energy_sum), float(lj_energy_sum), float(ee_ene)))
        if self.datfile is not None:
            self.datfile.write(self.crd.asnumpy())

    def main_initial(self):
        """main initial"""
        if self.control.mdout:
            self.file = open(self.control.mdout, 'w')
            self.file.write("_steps_ _TEMP_ _TOT_POT_ENE_ _BOND_ENE_ "
                            "_ANGLE_ENE_ _DIHEDRAL_ENE_ _14LJ_ENE_ _14CF_ENE_ _LJ_ENE_ _CF_PME_ENE_\n")
        if self.control.mdcrd:
            self.datfile = open(self.control.mdcrd, 'wb')

    def main_destroy(self):
        """main destroy"""
        if self.file is not None:
            self.file.close()
            print("Save .out file successfully!")
        if self.datfile is not None:
            self.datfile.close()
            print("Save .dat file successfully!")

    def construct(self, step, print_step):
        '''construct'''
        self.last_crd = self.crd
        if step == 0:
            res = self.neighbor_list_update_init(self.atom_numbers_in_grid_bucket, self.bucket, self.crd,
                                                 self.box_length, self.grid_n, self.grid_length_inverse,
                                                 self.atom_in_grid_serial, self.old_crd, self.crd_to_uint_crd_cof,
                                                 self.uint_crd, self.pointer, self.nl_atom_numbers, self.nl_atom_serial,
                                                 self.uint_dr_to_dr_cof, self.excluded_list_start, self.excluded_list,
                                                 self.excluded_numbers, self.need_refresh_flag, self.refresh_count)
            self.nl_atom_numbers = F.depend(self.nl_atom_numbers, res)
            self.nl_atom_serial = F.depend(self.nl_atom_serial, res)
            self.uint_dr_to_dr_cof = F.depend(self.uint_dr_to_dr_cof, res)
            self.old_crd = F.depend(self.old_crd, res)
            self.atom_numbers_in_grid_bucket = F.depend(self.atom_numbers_in_grid_bucket, res)
            self.bucket = F.depend(self.bucket, res)
            self.atom_in_grid_serial = F.depend(self.atom_in_grid_serial, res)
            self.pointer = F.depend(self.pointer, res)
            uint_crd = F.depend(self.uint_crd, res)

            force = self.simulation_caculate_force(uint_crd, self.uint_dr_to_dr_cof, self.nl_atom_numbers,
                                                   self.nl_atom_serial)
            bond_energy_sum, angle_energy_sum, dihedral_energy_sum, nb14_lj_energy_sum, nb14_cf_energy_sum, \
            lj_energy_sum, ee_ene, total_energy = self.simulation_caculate_energy(uint_crd, self.uint_dr_to_dr_cof)
            temperature = self.simulation_temperature()
            self.rand_state = self.setup_random_state()
            self.velocity, self.crd, _ = self.simulation_md_iteration_leap_frog_liujian(self.mass_inverse,
                                                                                        self.sqrt_mass, self.crd, force,
                                                                                        self.rand_state,
                                                                                        self.random_force)

            res = self.neighbor_list_update(self.atom_numbers_in_grid_bucket,
                                            self.bucket,
                                            self.crd,
                                            self.box_length,
                                            self.grid_n,
                                            self.grid_length_inverse,
                                            self.atom_in_grid_serial,
                                            self.old_crd,
                                            self.crd_to_uint_crd_cof,
                                            self.uint_crd,
                                            self.pointer,
                                            self.nl_atom_numbers,
                                            self.nl_atom_serial,
                                            self.uint_dr_to_dr_cof,
                                            self.excluded_list_start,
                                            self.excluded_list,
                                            self.excluded_numbers,
                                            self.need_refresh_flag,
                                            self.refresh_count)
            self.nl_atom_numbers = F.depend(self.nl_atom_numbers, res)
            self.nl_atom_serial = F.depend(self.nl_atom_serial, res)
        else:
            uint_crd = self.simulation_beforce_caculate_force()
            force = self.simulation_caculate_force(uint_crd, self.uint_dr_to_dr_cof, self.nl_atom_numbers,
                                                   self.nl_atom_serial)
            if print_step == 0:
                bond_energy_sum, angle_energy_sum, dihedral_energy_sum, nb14_lj_energy_sum, nb14_cf_energy_sum, \
                lj_energy_sum, ee_ene, total_energy = self.simulation_caculate_energy(
                    uint_crd, self.uint_dr_to_dr_cof)
            else:
                bond_energy_sum = self.zero_fp_tensor
                angle_energy_sum = self.zero_fp_tensor
                dihedral_energy_sum = self.zero_fp_tensor
                nb14_lj_energy_sum = self.zero_fp_tensor
                nb14_cf_energy_sum = self.zero_fp_tensor
                lj_energy_sum = self.zero_fp_tensor
                ee_ene = self.zero_fp_tensor
                total_energy = self.zero_fp_tensor
            temperature = self.simulation_temperature()
            self.velocity, self.crd, _ = self.simulation_md_iteration_leap_frog_liujian(self.mass_inverse,
                                                                                        self.sqrt_mass, self.crd, force,
                                                                                        self.rand_state,
                                                                                        self.random_force)
            res = self.neighbor_list_update(self.atom_numbers_in_grid_bucket,
                                            self.bucket,
                                            self.crd,
                                            self.box_length,
                                            self.grid_n,
                                            self.grid_length_inverse,
                                            self.atom_in_grid_serial,
                                            self.old_crd,
                                            self.crd_to_uint_crd_cof,
                                            self.uint_crd,
                                            self.pointer,
                                            self.nl_atom_numbers,
                                            self.nl_atom_serial,
                                            self.uint_dr_to_dr_cof,
                                            self.excluded_list_start,
                                            self.excluded_list,
                                            self.excluded_numbers,
                                            self.need_refresh_flag,
                                            self.refresh_count)
            self.nl_atom_numbers = F.depend(self.nl_atom_numbers, res)
            self.nl_atom_serial = F.depend(self.nl_atom_serial, res)
        return temperature, total_energy, bond_energy_sum, angle_energy_sum, dihedral_energy_sum, nb14_lj_energy_sum, \
               nb14_cf_energy_sum, lj_energy_sum, ee_ene, res
