import numpy as np
import mindspore.ops as ops
import mindspore.common.dtype as mstype
from mindspore import nn
from mindspore import Tensor, ParameterTuple
from mindspore.ops import functional as F

class Force(nn.Cell):
    '''Autograd for force caculate'''
    def __init__(self, energy_fn, space):
        super(Force, self).__init__()
        self.box_len = space.box_len
        self.charge = space.charge

        self.force_op = ops.GradOperation(get_by_list=True)
        self._energy_fn = energy_fn
        _crd = []
        for val in space.trainable_params():
            if "coordinates" in val.name:
                _crd.append(val)
        self._crd  = ParameterTuple(_crd)

    def construct(self):
        force = self.force_op(self._energy_fn, self._crd)(self.box_len, self.charge)
        return force[0]