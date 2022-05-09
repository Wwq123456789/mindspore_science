from .utils import get_periodic_displacement
from ...common.constants import PI
import mindspore.numpy as mnp
import numpy as np
import mindspore.common.dtype as mstype
from mindspore import nn
from mindspore import Tensor

def dihedral_energy(crd, box_len, atom_a, atom_b, atom_c, atom_d, ipn, pk, gamc, gams, pn):
    drij = get_periodic_displacement(crd[atom_a], crd[atom_b], box_len)
    drkj = get_periodic_displacement(crd[atom_c], crd[atom_b], box_len)
    drkl = get_periodic_displacement(crd[atom_c], crd[atom_d], box_len)

    r1 = mnp.cross(drij, drkj)
    r2 = mnp.cross(drkl, drkj)

    r1_1 = 1. / mnp.norm(r1, axis=-1)
    r2_1 = 1. / mnp.norm(r2, axis=-1)
    r1_1_r2_1 = r1_1 * r2_1

    phi = mnp.sum(r1 * r2, -1) * r1_1_r2_1
    phi = mnp.clip(phi, -0.999999, 0.999999)
    phi = mnp.arccos(phi)

    sign = mnp.sum(mnp.cross(r2, r1) * drkj, -1)
    phi = mnp.copysign(phi, sign)

    phi = PI - phi
    nphi = pn * phi

    cos_nphi = mnp.cos(nphi)
    sin_nphi = mnp.sin(nphi)
    ene = mnp.sum(pk + cos_nphi * gamc + sin_nphi * gams)
    return ene

class DihedralEnergy(nn.Cell):
    def __init__(self, space):
        super(DihedralEnergy, self).__init__()
        self.dihedral_atom_a = space.dihedral_atom_a
        self.dihedral_atom_b = space.dihedral_atom_b
        self.dihedral_atom_c = space.dihedral_atom_c
        self.dihedral_atom_d = space.dihedral_atom_d
        self.dihedral_ipn = space.dihedral_ipn
        self.dihedral_pn = space.dihedral_pn
        self.dihedral_pk = space.dihedral_pk
        self.dihedral_gamc = space.dihedral_gamc
        self.dihedral_gams = space.dihedral_gams
    def construct(self, crd, box_len):
        return dihedral_energy(crd, box_len, self.dihedral_atom_a, self.dihedral_atom_b, self.dihedral_atom_c, self.dihedral_atom_d, self.dihedral_ipn, self.dihedral_pk, self.dihedral_gamc, self.dihedral_gams, self.dihedral_pn)