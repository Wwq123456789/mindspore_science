# Copyright 2021 The AIMM Group at Shenzhen Bay Laboratory & Peking University & Huawei Technologies Co., Ltd
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
"""
Optimizer used to get the minimum value of a given function.
"""
import mindspore as ms
from mindspore import nn, Parameter, Tensor
from mindspore import numpy as msnp


class GradientDescent(nn.Optimizer):
    """The gradient descent optimizer with growing learning rate.
    Args:
        crd(tuple): Usually a tuple of parameters is given and the first element is coordinates.
        learning_rate(float): A factor of each optimize step size.
        factor(float): A growing factor of learning rate.
        nonh_mask(Tensor): The mask of atoms which are not Hydrogen.
        max_shift(float): The max step size each atom can move.
    """

    def __init__(self, crd, learning_rate=1e-03, factor=1.001, nonh_mask=None, max_shift=1.0):
        super(GradientDescent, self).__init__(learning_rate, crd)
        self.crd = crd[0]
        self.learning_rate = Parameter(Tensor(learning_rate, ms.float32))
        self.factor = Parameter(Tensor(factor, ms.float32))
        if nonh_mask is not None:
            self.nonh_mask = nonh_mask
        else:
            self.nonh_mask = msnp.ones((1, self.crd.shape[-2], 1))
        self.max_shift = Parameter(Tensor(max_shift, ms.float32))

    def construct(self, gradients):
        shift = self.learning_rate * gradients[0] * self.nonh_mask
        shift = msnp.where(shift > self.max_shift, self.max_shift, shift)
        shift = msnp.where(shift < -self.max_shift, -self.max_shift, shift)
        self.crd -= shift
        self.learning_rate *= self.factor
        return self.crd
