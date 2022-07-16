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
"""run one steo"""
from mindspore import ops
from mindspore import nn
import mindspore.numpy as mnp
from mindspore import Tensor
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.nn import Cell
from mindspore.parallel._utils import (_get_device_num, _get_gradients_mean, _get_parallel_mode)
from mindspore.context import ParallelMode
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.nn.optim import Optimizer


class RunOneStepCell(Cell):
    '''run one step cell'''

    def __init__(
            self,
            network: Cell,
            integrator: Optimizer,
            sens: float = 1.0,
    ):
        super().__init__(auto_prefix=False)

        self.network = network
        self.network.set_grad()
        self.integrator = integrator

        self.coordinates = self.network.coordinates
        self.pbc_box = self.network.pbc_box

        self.weights = self.integrator.parameters
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = F.identity
        self.mul = ops.Mul()
        self.parallel_mode = _get_parallel_mode()
        self.reducer_flag = self.parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL)
        if self.reducer_flag:
            self.mean = _get_gradients_mean()
            self.degree = _get_device_num()
            self.grad_reducer = DistributedGradReducer(self.weights, self.mean, self.degree)

    def construct(self, *inputs):
        energy = self.network(*inputs)

        sens = F.fill(energy.dtype, energy.shape, self.sens)
        grads = self.grad(self.network, self.weights)(*inputs, sens)
        forces = -1 * grads[0]

        grads = (self.grad_reducer(grads)[0],)

        energy = F.depend(energy, self.integrator(grads))

        return energy, forces


grad_scale = C.MultitypeFuncGraph("grad_scale")


@grad_scale.register("Tensor", "Tensor")
def _grad_scale(scale, grad):
    return grad * ops.Reciprocal()(scale)


mask_grad = C.MultitypeFuncGraph("mask_grad")


@mask_grad.register("Tensor", "Tensor")
def _mask_grad(mask, grad):
    mask = F.cast(mask, grad.dtype)
    new_grad = grad * mask
    return new_grad


clip_grad = C.MultitypeFuncGraph("clip_grad")


@clip_grad.register("Number", "Tensor")
def _clip_grad(clip_value, grad):
    """
    Clip gradients.

    Inputs:
        clip_value (float): Specifies how much to clip.
        grad (tuple[Tensor]): Gradients.

    Outputs:
        tuple[Tensor], clipped gradients.
    """
    dt = F.dtype(grad)
    new_grad = C.clip_by_value(grad, F.cast(-clip_value, dt), F.cast(clip_value, dt))
    return new_grad


class ClippedRunOneStepCell(nn.TrainOneStepCell):
    """clipped run one step cell
    Inherit from TrainOneStepCell, and add clip_by_value(True or False) options

    """
    def __init__(self, network, optimizer, loss_scale, include_mask=None, grad_clip_value=None, clip_by_value=True):
        super(ClippedRunOneStepCell, self).__init__(network, optimizer)
        self.cast = P.Cast()
        self.integrator = optimizer
        self.hyper_map = C.HyperMap()

        self.sens = loss_scale
        self.loss_scale = Tensor(loss_scale, mnp.float32)

        self.include_mask = include_mask
        self.grad_clip_value = grad_clip_value
        self.clip_by_value = clip_by_value

    def construct(self, *inputs):
        energy = self.network(*inputs)
        sens = F.fill(energy.dtype, energy.shape, self.sens)
        grads = self.grad(self.network, self.weights)(*inputs, sens)

        # 1. Rescale Grads:
        if self.sens > 1. + 1e-5:
            grads = self.hyper_map(F.partial(grad_scale, self.loss_scale), grads)

        forces = -1 * grads[0]
        # 2. Zero-out masked Grads:
        if self.include_mask is not None:
            grads = self.hyper_map(F.partial(mask_grad, self.include_mask), grads)

        # 3. Clip Grads:
        if self.grad_clip_value is not None:
            if self.clip_by_value:
                grads = self.hyper_map(F.partial(clip_grad, self.grad_clip_value), grads)
            else:  # Clip by norm:
                grads = C.clip_by_global_norm(grads, self.grad_clip_value)

        energy = F.depend(energy, self.integrator(grads))
        return energy, forces
