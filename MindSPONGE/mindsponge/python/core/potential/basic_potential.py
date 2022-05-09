import mindspore.numpy as mnp
import numpy as np
import mindspore.common.dtype as mstype
from mindspore import nn
from mindspore import Tensor,Parameter

from .angle import AngleEnergy
from .bond import BondEnergy
from .dihedral import DihedralEnergy
from .nb14 import NonBond14Energy

class BasicEnergy(nn.Cell):
    def __init__(self, space):
        super(BasicEnergy, self).__init__()
        self.coordinates = space.coordinates
        self.angle_energy = AngleEnergy(space)
        self.bond_energy = BondEnergy(space)
        self.dihedral_energy = DihedralEnergy(space)
        self.nb14_energy = NonBond14Energy(space)

    def construct(self, box_len, charge):
        angle_ene = self.angle_energy(self.coordinates, box_len)
        bond_ene = self.bond_energy(self.coordinates, box_len)
        dihedral_ene = self.dihedral_energy(self.coordinates, box_len)
        nb14_lj_ene, nb14_cf_ene = self.nb14_energy(self.coordinates, box_len, charge)
        total_ene = angle_ene + bond_ene + dihedral_ene + nb14_lj_ene + nb14_cf_ene
        return angle_ene
