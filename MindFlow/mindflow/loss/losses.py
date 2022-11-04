# Copyright 2021 Huawei Technologies Co., Ltd
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
# ==============================================================================
"""
add metric function
"""

from __future__ import absolute_import

import numpy as np

import mindspore.ops as ops
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore import Parameter
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P

from ..common.math import unpatchify
from ..utils.check_func import check_param_type, check_param_type_value


_loss_metric = {
    'l1_loss': nn.L1Loss,
    'l1': nn.L1Loss,
    'l2_loss': nn.MSELoss,
    'l2': nn.MSELoss,
    'mse_loss': nn.MSELoss,
    'mse': nn.MSELoss,
    'rmse_loss': nn.RMSELoss,
    'rmse': nn.RMSELoss,
    'mae_loss': nn.MAELoss,
    'mae': nn.MAELoss,
    'smooth_l1_loss': nn.SmoothL1Loss,
    'smooth_l1': nn.SmoothL1Loss,
}


def get_loss_metric(name):
    """
    Gets the loss function.

    Args:
        name (str): The name of the loss function.

    Returns:
        Function, the loss function.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindflow.loss import get_loss_metric
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> l1_loss = get_loss_metric('l1_loss')
        >>> logits = Tensor(np.array([1, 2, 3]), mindspore.float32)
        >>> labels = Tensor(np.array([[1, 1, 1], [1, 2, 2]]), mindspore.float32)
        >>> output = l1_loss(logits, labels)
        >>> print(output)
        0.6666667
    """
    if not isinstance(name, str):
        raise TypeError("the type of name should be str but got {}".format(type(name)))

    if name not in _loss_metric:
        raise ValueError("Unknown loss function type: {}".format(name))
    return _loss_metric.get(name)()


class RegularLossCell(nn.Cell):
    r"""
    L1/L2 regularized loss.

    Args:
        reg_params (Parameter): Parameter type tensor used for regularization.
        reg_mode (str): type to compute the regularized loss function. Only ["l1", "l2"] are supported. Default: "l2".

    Inputs:
        None.

    Outputs:
        Tensor. a scalar tensor with shape :math:`()`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindflow.loss import RegularLossCell
        >>> from mindspore import Parameter, Tensor
        >>> import mindspore.common.dtype as ms_type
        >>> latent_init = np.ones((2, 3))
        >>> latent_vector = Parameter(Tensor(latent_init, ms_type.float32), requires_grad=True)
        >>> net = RegularLossCell(latent_vector)
        >>> output = net()
        >>> print(output)
        1.0
    """

    def __init__(self, reg_params, reg_factor=0.01, reg_mode="l2"):
        super(RegularLossCell, self).__init__()
        check_param_type(reg_params, "reg_params", data_type=Parameter)
        check_param_type_value(reg_mode, "reg_mode", data_type=str, valid_value=["l1", "l2"])
        check_param_type(reg_factor, "reg_factor", data_type=float)
        if reg_factor < 0.0:
            raise ValueError("The reg_factor must be a non-negtive value, but got {}".format(reg_factor))
        self.reg_params = reg_params
        self.reg_mode = reg_mode
        self.reg_factor = reg_factor
        self.reduce_mean = ops.ReduceMean()
        self.pow = ops.Pow()
        self.abs = ops.Abs()

    def construct(self):
        """get regularized loss"""
        loss = 0.0
        if self.reg_mode == "l1":
            loss = self.reduce_mean(self.abs(self.reg_params))
        elif self.reg_mode == "l2":
            loss = self.reduce_mean(self.pow(self.reg_params, 2))
        loss = self.reg_factor * loss
        return loss


class WeightedLossCell(nn.Cell):
    r"""
    Base class of weighting multi-task losses automatically based on the multitasks learning strategy .
    """

    def __init__(self):
        super(WeightedLossCell, self).__init__()
        self.type = type(self).__name__
        self.use_grads = False

    def construct(self, losses):
        """
        Defines the computation to be performed. This method must be overridden by all subclasses.

        Returns:
            Tensor, returns the computed result.
        """
        return losses


class MTLWeightedLossCell(WeightedLossCell):
    r"""
    Compute the MTL strategy weighted multi-task losses automatically.
    For more information, please refer to `MTL weighted losses <https://arxiv.org/pdf/1805.06334.pdf>`_ .

    Args:
        num_losses (int): The number of multi-task losses, should be positive integer.

    Inputs:
        - **input** - tuple of Tensors.

    Outputs:
        Tensor. losses for MTL weighted strategy.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindelec.architecture import MTLWeightedLossCell
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> net = MTLWeightedLossCell(num_losses=2)
        >>> input1 = Tensor(1.0, mindspore.float32)
        >>> input2 = Tensor(0.8, mindspore.float32)
        >>> output = net((input1, input2))
        >>> print(output)
        2.2862945
    """

    def __init__(self, num_losses, bound_param=0.0):
        super(MTLWeightedLossCell, self).__init__()
        check_param_type(num_losses, "num_losses", data_type=int, exclude_type=bool)
        if num_losses <= 0:
            raise ValueError("the value of num_losses should be positive, but got {}".format(num_losses))
        self.num_losses = num_losses
        check_param_type(bound_param, "bound_param", data_type=float)
        self.bounded = bound_param > 1.0e-6
        self.bound_param = bound_param ** 2
        self.params = Parameter(Tensor(np.ones(num_losses), mstype.float32), requires_grad=True)
        self.concat = ops.Concat(axis=0)
        self.pow = ops.Pow()
        self.log = ops.Log()
        self.div = ops.RealDiv()

    def construct(self, losses):
        """get weighted total loss"""
        loss_sum = 0
        params = self.pow(self.params, 2)
        for i in range(self.num_losses):
            if self.bounded:
                weight = params[i] + self.bound_param
                reg = params[i] + self.bound_param
            else:
                weight = params[i]
                reg = params[i] + 1.0
            weighted_loss = 0.5 * self.div(losses[i], weight) + self.log(reg)
            loss_sum = loss_sum + weighted_loss
        return loss_sum


class WaveletTransformLoss(nn.LossBase):
    r"""
    The multi-level wavelet transformation losses.

    Args:
        wave_level (int): The number of the wavelet transformation levels, should be positive integer.
        regroup (bool): The regroup error combination form of the wavelet transformation losses. Default: "False".

    Inputs:
        - **input** - tuple of Tensors. Tensor of shape :math:`(B H*W/(P*P) P*P*C)`, where B denotes the batch size.
        H, W denotes the height and the width of the image, respectively.
        P denotes the patch size. C denots the feature channels.

    Outputs:
        Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindflow.loss import WaveletTransformLoss
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> net = WaveletTransformLoss(wave_level=2)
        >>> input1 = Tensor(np.ones((32, 288, 768)), mstype.float32)
        >>> input2 = Tensor(np.ones((32, 288, 768)), mstype.float32)
        >>> output = net((input1, input2))
        >>> print(output)
        2.0794415
    """

    def __init__(self, wave_level=2, regroup=False):
        super(WaveletTransformLoss, self).__init__()
        self.abs = P.Abs()
        self.wave_level = wave_level
        self.regroup = regroup
        self.print = P.Print()
        if self.regroup:
            self.mtl = MTLWeightedLossCell(num_losses=3)
        else:
            if self.wave_level == 1:
                self.mtl = MTLWeightedLossCell(num_losses=4)
            else:
                self.mtl = MTLWeightedLossCell(num_losses=self.wave_level + 1)

    def construct(self, logit, label):
        l1_loss = P.ReduceMean()(self.abs(logit - label))
        logit = unpatchify(logit)
        label = unpatchify(label)

        if self.wave_level == 1:
            _, x_hl, x_lh, x_hh = self.dwt_split(logit)
            _, y_hl, y_lh, y_hh = self.dwt_split(label)
            hl_loss = P.ReduceMean()(self.abs(x_hl - y_hl))
            lh_loss = P.ReduceMean()(self.abs(x_lh - y_lh))
            hh_loss = P.ReduceMean()(self.abs(x_hh - y_hh))
            l_total = self.mtl((l1_loss, hl_loss, lh_loss, hh_loss))
        else:
            wave_losses = []
            for _ in range(self.wave_level):
                _, x_hl, x_lh, x_hh = self.dwt_split(logit)
                _, y_hl, y_lh, y_hh = self.dwt_split(label)
                hl_loss = P.ReduceMean()(self.abs(x_hl - y_hl))
                lh_loss = P.ReduceMean()(self.abs(x_lh - y_lh))
                hh_loss = P.ReduceMean()(self.abs(x_hh - y_hh))
                wave_loss_cur = hl_loss + lh_loss + hh_loss
                wave_losses.append(wave_loss_cur)
            wave_losses.append(l1_loss)
            l_total = self.mtl(wave_losses)
        return l_total

    @staticmethod
    def _split_data(data, axis=1):
        data_shape = data.shape
        data_re = []
        if axis == 1:
            data_re = P.Reshape()(data, (data_shape[0],
                                         data_shape[1] // 2,
                                         2,
                                         data_shape[2],
                                         data_shape[3]))
            data_re = P.Transpose()(data_re, (0, 2, 1, 3, 4))
        if axis == 2:
            data_re = P.Reshape()(data, (data_shape[0],
                                         data_shape[1],
                                         data_shape[2] // 2,
                                         2,
                                         data_shape[3]))
            data_re = P.Transpose()(data_re, (0, 1, 3, 2, 4))

        split_op = P.Split(axis, 2)
        data_split = split_op(data_re)
        data_01 = P.Squeeze()(data_split[0])
        data_02 = P.Squeeze()(data_split[1])
        return data_01, data_02

    def dwt_split(self, data):
        x01, x02 = self._split_data(data, axis=1)
        x1, x3 = self._split_data(x01 / 2, axis=2)
        x2, x4 = self._split_data(x02 / 2, axis=2)
        x_ll = x1 + x2 + x3 + x4
        x_hl = -x1 - x2 + x3 + x4
        x_lh = -x1 + x2 - x3 + x4
        x_hh = x1 - x2 - x3 + x4
        return x_ll, x_hl, x_lh, x_hh