import mindspore.numpy as mnp
from ...common.constants import KB

def temperature(mass, vel):
    residue_numbers, dim = vel.shape
    mass = mnp.expand_dims(mass, -1)
    ek = 0.5 * mnp.square(mass * vel) / mass * 2. / dim / KB / residue_numbers
    temperature = mnp.sum(ek)
    return temperature

def pressure(self, vel, mass, virial, volume, is_download):
    ek = 0.5 * mass * P.ReduceSum(True)(vel * vel, 0)  # 可以优化
    sum_of_atom_ek = P.ReduceSum(True)(ek)
    atom_virial = P.ReduceSum(True)(virial)
    v_inverse = 1 / volume
    pressure = (sum_of_atom_ek + atom_virial) / 3 * v_inverse
    if is_download:
        return pressure
    return 0
