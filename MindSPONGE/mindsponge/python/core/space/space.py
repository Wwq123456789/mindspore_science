import numpy as np
import math
import mindspore.common.dtype as mstype
from mindspore import nn
from mindspore import Tensor, Parameter
from ...common.constants import TIME_UNIT, KB

class Space(nn.Cell):
    def __init__(self, system=None):
        super(Space, self).__init__()
        if system:
            self._init_system(system)

    def _init_system(self, system):
        _type_list = [int, float, str, np.int32, np.float32]
        for key in vars(system):
            if type(vars(system)[key]) is np.ndarray:
                if "coordinates" in key or "velocities" in key:
                    setattr(self, key, Parameter(Tensor(vars(system)[key])))
                else:
                    setattr(self, key, Tensor(vars(system)[key]))
            if type(vars(system)[key]) in _type_list:
                setattr(self, key, vars(system)[key])


        # self.dt = config.dt
        # self.mode = config.mode
        # self.target_temperature = config.target_temperature
        # self.gamma_ln = 1.0
        # if 'gamma_ln' in vars(config):
        #     self.gamma_ln = config.gamma_ln
        # elif 'langevin_gamma' in vars(config):
        #     self.gamma_ln = config.langevin_gamma
        # self.gamma_ln = self.gamma_ln / TIME_UNIT
        # self.exp_gamma = math.exp(-1 * self.gamma_ln * self.dt)
        # self.sqrt_gamma = math.sqrt((1. - self.exp_gamma * self.exp_gamma) * self.target_temperature * KB)
        # if system is not None:
        #     self.angle = np.array(system.angle)
        #     self.angle_atom_a = Tensor(self.angle[:,0], mstype.int32)
        #     self.angle_atom_b = Tensor(self.angle[:,1], mstype.int32)
        #     self.angle_atom_c = Tensor(self.angle[:,2], mstype.int32)
        #     self.angle_k = Tensor(self.angle[:,3], mstype.float32)
        #     self.angle_theta0 = Tensor(self.angle[:,4], mstype.float32)

        #     self.bond = np.array(system.bond)
        #     self.bond_atom_a = Tensor(self.bond[:,0], mstype.int32)
        #     self.bond_atom_b = Tensor(self.bond[:,1], mstype.int32)
        #     self.bond_k = Tensor(self.bond[:,2], mstype.float32)
        #     self.bond_r0 = Tensor(self.bond[:,3], mstype.float32)

        #     self.dihedral = np.array(system.dihedral)
        #     self.dihedral_atom_a = Tensor(self.dihedral[:,0], mstype.int32)
        #     self.dihedral_atom_b = Tensor(self.dihedral[:,1], mstype.int32)
        #     self.dihedral_atom_c = Tensor(self.dihedral[:,2], mstype.int32)
        #     self.dihedral_atom_d = Tensor(self.dihedral[:,3], mstype.int32)
        #     self.dihedral_ipn = Tensor(self.dihedral[:,4], mstype.float32)
        #     self.dihedral_pn = Tensor(self.dihedral[:,4], mstype.float32)
        #     self.dihedral_pk = Tensor(self.dihedral[:,5], mstype.float32)
        #     self.dihedral_gamc = Tensor(np.cos(self.dihedral[:,6]) * self.dihedral[:,5], mstype.float32)
        #     self.dihedral_gams = Tensor(np.sin(self.dihedral[:,6]) * self.dihedral[:,5], mstype.float32)

        #     # lennard jones
        #     self.lj_a = Tensor(np.array(system.LJ_A) * 12.0, mstype.float32)
        #     self.lj_b = Tensor(np.array(system.LJ_B) * 6.0, mstype.float32)
        #     self.lj_type = Tensor(np.array(system.LJ_idx), mstype.int32)
        #     # nb14
        #     self.nb14_atom_a = Tensor(np.array(system.nb14)[:,0],mstype.int32)
        #     self.nb14_atom_b = Tensor(np.array(system.nb14)[:,1],mstype.int32)
        #     self.nb14_lj_scale_factor = Tensor(np.array(system.nb14)[:,2],mstype.float32)
        #     self.nb14_cf_scale_factor = Tensor(np.array(system.nb14)[:,3],mstype.float32)

        #     self.coordinates = Parameter(Tensor(system.coordinates, mstype.float32))
        #     self.box_len = Tensor(np.array(system.box[:3]), mstype.float32)
        #     self.charge = Tensor(np.array(system.charge), mstype.float32)
        #     if system.velocities:
        #         self.velocities = Parameter(Tensor(np.array(system.velocities), mstype.float32))
        #     else:
        #         self.velocities = Parameter(Tensor(np.zeros([system.atom_numbers, 3]), mstype.float32))
        #     self.mass = Tensor(np.array(system.mass), mstype.float32)
        #     self.inverse_mass = Tensor(1.0 / np.array(system.mass), mstype.float32)
        #     self.residue_numbers = system.residue_numbers
        #     self.atom_numbers = system.atom_numbers

        #     if config.constrain_mode == "simple_constrain":
        #         self.volume = system.box[0] * system.box[1] * system.box[2]

