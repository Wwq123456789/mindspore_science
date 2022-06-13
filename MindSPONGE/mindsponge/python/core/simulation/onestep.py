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
from mindspore.ops import functional as F
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
