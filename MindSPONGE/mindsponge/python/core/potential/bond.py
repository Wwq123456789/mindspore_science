from .utils import get_periodic_displacement
import mindspore.numpy as mnp
import numpy as np
import mindspore.common.dtype as mstype
from mindspore import nn
from mindspore import Tensor

def bond_energy(crd, box_len, bond_atom_a, bond_atom_b, bond_k, bond_r0):
    dr = get_periodic_displacement(crd[bond_atom_a], crd[bond_atom_a], box_len)
    r1 = mnp.norm(dr, axis=-1)
    temp = r1 - bond_r0
    bond_ene = temp * temp * bond_k
    ene = mnp.sum(bond_ene)
    return ene

class BondEnergy(nn.Cell):
    def __init__(self, space):
        super(BondEnergy, self).__init__()
        self.bond_atom_a = space.bond_atom_a
        self.bond_atom_b = space.bond_atom_b
        self.bond_k = space.bond_k
        self.bond_r0 = space.bond_r0
    def construct(self, crd, box_len):
        return bond_energy(crd, box_len, self.bond_atom_a, self.bond_atom_b, self.bond_k, self.bond_r0)