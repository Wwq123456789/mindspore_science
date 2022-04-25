from .utils import get_periodic_displacement
import mindspore.numpy as mnp
import numpy as np
import mindspore.common.dtype as mstype
from mindspore import nn
from mindspore import Tensor

def angle_energy(crd, box_len, angle_atom_a, angle_atom_b, angle_atom_c, angle_atom_k, angle_theta0):
    drij = get_periodic_displacement(crd[angle_atom_a], crd[angle_atom_b], box_len)
    drkj = get_periodic_displacement(crd[angle_atom_c], crd[angle_atom_b], box_len)
    rij_2 = 1 / mnp.sum(drij ** 2, -1)
    rkj_2 = 1 / mnp.sum(drkj ** 2, -1)
    rij_1_rkj_1 = mnp.sqrt(rij_2 * rkj_2)
    costheta = mnp.sum(drij * drkj, -1) * rij_1_rkj_1
    costheta = mnp.clip(costheta, -0.999999, 0.999999)
    theta = mnp.arccos(costheta)
    dtheta = theta - angle_theta0
    ene = mnp.sum(angle_atom_k * dtheta * dtheta)
    return ene

class AngleEnergy(nn.Cell):
    def __init__(self, space):
        super(AngleEnergy, self).__init__()
        self.angle_atom_a = space.angle_atom_a
        self.angle_atom_b = space.angle_atom_b
        self.angle_atom_c = space.angle_atom_c
        self.angle_k = space.angle_k
        self.angle_theta0 = space.angle_theta0
    def construct(self, crd, box_len):
        return angle_energy(crd, box_len, self.angle_atom_a, self.angle_atom_b, self.angle_atom_c, self.angle_k, self.angle_theta0)