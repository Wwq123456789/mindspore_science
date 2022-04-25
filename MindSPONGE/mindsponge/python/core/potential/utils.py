import mindspore.numpy as mnp

def get_periodic_displacement(vec_a, vec_b, box_len):
    dr = vec_a - vec_b
    dr = dr - mnp.floor(dr / box_len + 0.5) * box_len
    return dr
