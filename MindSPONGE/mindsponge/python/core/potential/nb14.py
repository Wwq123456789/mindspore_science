from .utils import get_periodic_displacement
# from ...common.constants import PI
import mindspore.numpy as mnp
import numpy as np
import mindspore.common.dtype as mstype
from mindspore import nn
from mindspore import Tensor

def nb14_lj_energy(crd, box_len, charge,
                    a_14, b_14, lj_scale_factor, LJ_type, LJ_type_A, LJ_type_B):
    r1_xyz = crd[a_14] # [uint_x, uint_y, uint_z] (M,3)
    r2_xyz = crd[b_14] # [uint_x, uint_y, uint_z] (M,3)
    dr_xyz = get_periodic_displacement(r2_xyz, r1_xyz, box_len)

    dr2 = dr_xyz * dr_xyz
    dr_2 = 1. / mnp.sum(dr2, 1)
    dr_4 = dr_2 * dr_2
    dr_6 = dr_4 * dr_2
    dr_12 = dr_6 * dr_6 # (M,3)

    r1_lj_type = LJ_type[a_14] # (M,)
    r2_lj_type = LJ_type[b_14] # (M,)

    y = mnp.abs(r2_lj_type - r1_lj_type) # (M,)
    x = r2_lj_type + r1_lj_type # (M,)

    r2_lj_type = mnp.divide(x + y, 2, dtype=mnp.int32)
    x = mnp.divide(x - y, 2, dtype=mnp.int32)
    atom_pair_LJ_type = mnp.divide(r2_lj_type * (r2_lj_type + 1), 2, dtype=mnp.int32) + x # (M,)

    ene_lin = 0.08333333 * LJ_type_A[atom_pair_LJ_type] * dr_12 - \
              0.1666666 * LJ_type_B[atom_pair_LJ_type] * dr_6
    ene = mnp.sum(ene_lin * lj_scale_factor)
    return ene

def nb14_cf_energy(crd, box_len, charge,nb14_atom_a, nb14_atom_b, cf_scale_factor,
                   LJ_type):
    r1_xyz = crd[nb14_atom_a] # [uint_x, uint_y, uint_z] (M,3)
    r2_xyz = crd[nb14_atom_b] # [uint_x, uint_y, uint_z] (M,3)
    dr_xyz = get_periodic_displacement(r2_xyz, r1_xyz, box_len)
    r_1 = 1. / mnp.norm(dr_xyz, axis=-1)

    r1_charge = charge[nb14_atom_a]
    r2_charge = charge[nb14_atom_b]
    ene_lin = r1_charge * r2_charge * r_1
    ene = mnp.sum(ene_lin * cf_scale_factor)
    return ene


class NonBond14Energy(nn.Cell):
    def __init__(self, space):
        super(NonBond14Energy, self).__init__()
        # lennard jones
        self.lj_a = space.LJ_A
        self.lj_b = space.LJ_B
        self.lj_type = space.LJ_type
        # nb14
        self.nb14_atom_a = space.nb14_atom_a
        self.nb14_atom_b = space.nb14_atom_b
        self.nb14_lj_scale_factor = space.nb14_lj_scale_factor
        self.nb14_cf_scale_factor = space.nb14_cf_scale_factor

    def construct(self, crd, box_len, charge):
        nb14_lj_ene = nb14_lj_energy(crd, box_len, charge, self.nb14_atom_a, self.nb14_atom_b, self.nb14_lj_scale_factor, self.lj_type, self.lj_a, self.lj_b)
        nb14_cf_ene = nb14_cf_energy(crd, box_len, charge, self.nb14_atom_a, self.nb14_atom_b, self.nb14_cf_scale_factor, self.lj_type)
        return nb14_lj_ene, nb14_cf_ene