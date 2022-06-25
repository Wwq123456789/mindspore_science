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
"""Triangle"""
import numpy as np
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore import Parameter
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
from .basic import Attention


class TriangleAttention(nn.Cell):
    '''triangle attention'''

    def __init__(self, orientation, num_head, key_dim, value_dim, gating, layer_norm_dim, batch_size=None, slice_num=0,
                 mixed_precision=True):
        super(TriangleAttention, self).__init__()
        self.num_head = num_head
        self.orientation = orientation
        self.orientation_is_per_column = (self.orientation == 'per_column')
        self.init_factor = Tensor(1. / np.sqrt(layer_norm_dim), mstype.float32)
        self.query_norm = P.LayerNorm(begin_norm_axis=-1, begin_params_axis=-1, epsilon=1e-5)
        self.matmul = P.MatMul(transpose_b=True)
        self.attn_mod = Attention(num_head, key_dim, value_dim, gating, layer_norm_dim, layer_norm_dim, layer_norm_dim,
                                  batch_size, mixed_precision)
        self.batch_size = batch_size
        self.slice_num = slice_num
        self.layer_norm_dim = layer_norm_dim
        self.idx = Tensor(0, mstype.int32)
        if mixed_precision:
            self._type = self._type
        else:
            self._type = mstype.float32
        self._init_parameter()

    def construct(self, pair_act, pair_mask, index):
        '''construct'''
        if self.batch_size:
            query_norm_gamma = P.Gather()(self.query_norm_gammas, index, 0)
            query_norm_beta = P.Gather()(self.query_norm_betas, index, 0)
            feat_2d_weight = P.Cast()(P.Gather()(self.feat_2d_weights, index, 0), self._type)
        else:
            query_norm_gamma = self.query_norm_gammas
            query_norm_beta = self.query_norm_betas
            feat_2d_weight = P.Cast()(self.feat_2d_weights, self._type)
        if self.orientation_is_per_column:
            pair_act = P.Transpose()(pair_act, (1, 0, 2))
            pair_mask = P.Transpose()(pair_mask, (1, 0))

        pair_mask = P.Cast()(pair_mask, mstype.float32)
        pair_mask = 1e9 * (pair_mask - 1.)
        bias = P.ExpandDims()(P.ExpandDims()(pair_mask, 1), 2)

        pair_act = P.Cast()(pair_act, mstype.float32)
        pair_act, _, _ = self.query_norm(pair_act,
                                         query_norm_gamma,
                                         query_norm_beta)
        pair_act = P.Cast()(pair_act, self._type)

        q, k, _ = pair_act.shape
        nonbatched_bias = self.matmul(P.Reshape()(pair_act, (-1, pair_act.shape[-1])), feat_2d_weight)
        nonbatched_bias = P.Transpose()(P.Reshape()(nonbatched_bias, (q, k, -1)), (2, 0, 1))
        if self.slice_num:
            pair_act_ori_shape = P.Shape()(pair_act)
            slice_shape = (self.slice_num, -1) + pair_act_ori_shape[1:]
            pair_act = P.Reshape()(pair_act, slice_shape).astype(self._type)
            bias_shape = P.Shape()(bias)
            bias = P.Reshape()(bias, slice_shape[:2] + bias_shape[1:])

            slice_idx = 0
            slice_idx_tensor = self.idx
            pair_act_tuple = ()

            pair_act_slice = P.Gather()(pair_act, slice_idx_tensor, 0)
            bias_slice = P.Gather()(bias, slice_idx_tensor, 0)
            pair_act_slice = self.attn_mod(pair_act_slice, pair_act_slice, bias_slice, index, nonbatched_bias)
            pair_act_slice = P.Reshape()(pair_act_slice, ((1,) + P.Shape()(pair_act_slice)))
            pair_act_tuple = pair_act_tuple + (pair_act_slice,)
            slice_idx += 1
            slice_idx_tensor += 1

            while slice_idx < self.slice_num:
                pair_act_slice = P.Gather()(pair_act, slice_idx_tensor, 0)
                pair_act_slice = F.depend(pair_act_slice, pair_act_tuple[-1])
                bias_slice = P.Gather()(bias, slice_idx_tensor, 0)
                pair_act_slice = self.attn_mod(pair_act_slice, pair_act_slice, bias_slice, index, nonbatched_bias)
                pair_act_slice = P.Reshape()(pair_act_slice, ((1,) + P.Shape()(pair_act_slice)))
                pair_act_tuple = pair_act_tuple + (pair_act_slice,)
                slice_idx += 1
                slice_idx_tensor += 1
            pair_act = P.Concat()(pair_act_tuple)
            pair_act = P.Reshape()(pair_act, pair_act_ori_shape)

            if self.orientation_is_per_column:
                pair_act = mnp.swapaxes(pair_act, -2, -3)
            return pair_act

        pair_act = self.attn_mod(pair_act, pair_act, bias, index, nonbatched_bias)
        if self.orientation_is_per_column:
            pair_act = P.Transpose()(pair_act, (1, 0, 2))
        return pair_act

    def _init_parameter(self):
        '''init parameter'''
        if self.batch_size:
            self.query_norm_gammas = Parameter(Tensor(np.zeros((self.batch_size, self.layer_norm_dim)), mstype.float32))
            self.query_norm_betas = Parameter(Tensor(np.zeros((self.batch_size, self.layer_norm_dim)), mstype.float32))
            self.feat_2d_weights = Parameter(
                Tensor(np.zeros((self.batch_size, self.num_head, self.layer_norm_dim)), mstype.float32))
        else:
            self.query_norm_gammas = Parameter(Tensor(np.ones((self.layer_norm_dim)), mstype.float32))
            self.query_norm_betas = Parameter(Tensor(np.zeros((self.layer_norm_dim)), mstype.float32))
            self.feat_2d_weights = Parameter(Tensor(
                np.random.normal(scale=1 / np.sqrt(self.layer_norm_dim), size=(self.num_head, self.layer_norm_dim)),
                mstype.float32))


class TriangleMultiplication(nn.Cell):
    '''triangle multiplication'''

    def __init__(self, num_intermediate_channel, equation, layer_norm_dim, batch_size, mixed_precision=True):
        super(TriangleMultiplication, self).__init__()
        self.num_intermediate_channel = num_intermediate_channel
        self.equation = equation
        self.layer_norm = P.LayerNorm(begin_norm_axis=-1, begin_params_axis=-1, epsilon=1e-5)
        self.matmul = P.MatMul(transpose_b=True)
        self.sigmoid = nn.Sigmoid()
        self.batch_matmul_trans_b = P.BatchMatMul(transpose_b=True)
        if self.equation not in ["ikc,jkc->ijc", "kjc,kic->ijc"]:
            print("TriangleMultiplication Not Suppl")
        if self.equation == "ikc,jkc->ijc":
            self.equation = True
        elif self.equation == "kjc,kic->ijc":
            self.equation = False
        else:
            self.equation = None
        self.batch_size = batch_size
        self.layer_norm_dim = layer_norm_dim
        if mixed_precision:
            self._type = self._type
        else:
            self._type = mstype.float32
        self._init_parameter()

    def construct(self, act, mask, index):
        '''construct'''
        if self.batch_size:
            layer_norm_input_gamma = P.Gather()(self.layer_norm_input_gammas, index, 0)
            layer_norm_input_beta = P.Gather()(self.layer_norm_input_betas, index, 0)
            left_projection_weight = P.Cast()(P.Gather()(self.left_projection_weights, index, 0), self._type)
            left_projection_bias = P.Cast()(P.Gather()(self.left_projection_biases, index, 0), self._type)
            right_projection_weight = P.Cast()(P.Gather()(self.right_projection_weights, index, 0), self._type)
            right_projection_bias = P.Cast()(P.Gather()(self.right_projection_biases, index, 0), self._type)
            left_gate_weight = P.Cast()(P.Gather()(self.left_gate_weights, index, 0), self._type)
            left_gate_bias = P.Cast()(P.Gather()(self.left_gate_biases, index, 0), self._type)
            right_gate_weight = P.Cast()(P.Gather()(self.right_gate_weights, index, 0), self._type)
            right_gate_bias = P.Cast()(P.Gather()(self.right_gate_biases, index, 0), self._type)
            center_layer_norm_gamma = P.Cast()(P.Gather()(self.center_layer_norm_gammas, index, 0), self._type)
            center_layer_norm_beta = P.Cast()(P.Gather()(self.center_layer_norm_betas, index, 0), self._type)
            output_projection_weight = P.Cast()(P.Gather()(self.output_projection_weights, index, 0), self._type)
            output_projection_bias = P.Cast()(P.Gather()(self.output_projection_biases, index, 0), self._type)
            gating_linear_weight = P.Cast()(P.Gather()(self.gating_linear_weights, index, 0), self._type)
            gating_linear_bias = P.Cast()(P.Gather()(self.gating_linear_biases, index, 0), self._type)
        else:
            layer_norm_input_gamma = self.layer_norm_input_gammas
            layer_norm_input_beta = self.layer_norm_input_betas
            left_projection_weight = P.Cast()(self.left_projection_weights, self._type)
            left_projection_bias = P.Cast()(self.left_projection_biases, self._type)
            right_projection_weight = P.Cast()(self.right_projection_weights, self._type)
            right_projection_bias = P.Cast()(self.right_projection_biases, self._type)
            left_gate_weight = P.Cast()(self.left_gate_weights, self._type)
            left_gate_bias = P.Cast()(self.left_gate_biases, self._type)
            right_gate_weight = P.Cast()(self.right_gate_weights, self._type)
            right_gate_bias = P.Cast()(self.right_gate_biases, self._type)
            center_layer_norm_gamma = P.Cast()(self.center_layer_norm_gammas, self._type)
            center_layer_norm_beta = P.Cast()(self.center_layer_norm_betas, self._type)
            output_projection_weight = P.Cast()(self.output_projection_weights, self._type)
            output_projection_bias = P.Cast()(self.output_projection_biases, self._type)
            gating_linear_weight = P.Cast()(self.gating_linear_weights, self._type)
            gating_linear_bias = P.Cast()(self.gating_linear_biases, self._type)

        mask = P.Cast()(mask, self._type)
        mask = P.ExpandDims()(mask, -1)
        act = P.Cast()(act, mstype.float32)
        act, _, _ = self.layer_norm(act,
                                    layer_norm_input_gamma,
                                    layer_norm_input_beta)
        act = P.Cast()(act, self._type)

        input_act = act
        act_shape = P.Shape()(act)
        if len(act_shape) != 2:
            act = P.Reshape()(act, (-1, act_shape[-1]))
        out_shape = act_shape[:-1] + (-1,)
        left_projection = P.Reshape()(P.BiasAdd()(self.matmul(act, left_projection_weight), left_projection_bias),
                                      out_shape)
        act = F.depend(act, left_projection)

        left_gate_values = P.Reshape()(P.BiasAdd()(self.matmul(act, left_gate_weight), left_gate_bias), out_shape)
        left_gate_values = P.Cast()(left_gate_values, mstype.float32)
        left_gate_values = self.sigmoid(left_gate_values)
        left_gate_values = P.Cast()(left_gate_values, self._type)

        left_proj_act = left_projection * left_gate_values
        act = F.depend(act, left_proj_act)
        right_projection = mask * P.Reshape()(
            P.BiasAdd()(self.matmul(act, right_projection_weight), right_projection_bias), out_shape)

        act = F.depend(act, right_projection)

        right_gate_values = P.Reshape()(P.BiasAdd()(self.matmul(act, right_gate_weight), right_gate_bias), out_shape)
        right_gate_values = P.Cast()(right_gate_values, mstype.float32)
        right_gate_values = self.sigmoid(right_gate_values)
        right_gate_values = P.Cast()(right_gate_values, self._type)

        right_proj_act = right_projection * right_gate_values
        left_proj_act = F.depend(left_proj_act, right_proj_act)

        if self.equation is not None:
            if self.equation:
                left_proj_act_tmp = P.Transpose()(left_proj_act, (2, 0, 1))
                right_proj_act_tmp = P.Transpose()(right_proj_act, (2, 0, 1))
                act = self.batch_matmul_trans_b(left_proj_act_tmp, right_proj_act_tmp)
                act = P.Transpose()(act, (1, 2, 0))
            else:
                left_proj_act_tmp = P.Transpose()(left_proj_act, (2, 1, 0))
                right_proj_act_tmp = P.Transpose()(right_proj_act, (2, 1, 0))
                act = self.batch_matmul_trans_b(left_proj_act_tmp, right_proj_act_tmp)
                act = P.Transpose()(act, (2, 1, 0))

        act = P.Cast()(act, mstype.float32)
        center_layer_norm_gamma = P.Cast()(center_layer_norm_gamma, mstype.float32)
        center_layer_norm_beta = P.Cast()(center_layer_norm_beta, mstype.float32)
        act, _, _ = self.layer_norm(act,
                                    center_layer_norm_gamma,
                                    center_layer_norm_beta)
        act = P.Cast()(act, self._type)
        act_shape = P.Shape()(act)

        if len(act_shape) != 2:
            act = P.Reshape()(act, (-1, act_shape[-1]))
        out_shape = act_shape[:-1] + (-1,)
        act = P.Reshape()(P.BiasAdd()(self.matmul(act, output_projection_weight), output_projection_bias), out_shape)
        input_act_shape = P.Shape()(input_act)
        if len(input_act_shape) != 2:
            input_act = P.Reshape()(input_act, (-1, input_act_shape[-1]))
        out_shape = input_act_shape[:-1] + (-1,)
        gate_values = P.Reshape()(P.BiasAdd()(self.matmul(input_act, gating_linear_weight), gating_linear_bias),
                                  out_shape)
        gate_values = P.Cast()(gate_values, mstype.float32)
        gate_values = self.sigmoid(gate_values)
        gate_values = P.Cast()(gate_values, self._type)

        act = act * gate_values
        return act

    def _init_parameter(self):
        '''init parameter'''
        if self.batch_size:
            self.layer_norm_input_gammas = Parameter(
                Tensor(np.zeros((self.batch_size, self.layer_norm_dim)), mstype.float32))
            self.layer_norm_input_betas = Parameter(
                Tensor(np.zeros((self.batch_size, self.layer_norm_dim)), mstype.float32))
            self.left_projection_weights = Parameter(
                Tensor(np.zeros((self.batch_size, self.num_intermediate_channel, self.layer_norm_dim)),
                       mstype.float32))
            self.left_projection_biases = Parameter(
                Tensor(np.zeros((self.batch_size, self.num_intermediate_channel)), mstype.float32))
            self.right_projection_weights = Parameter(
                Tensor(np.zeros((self.batch_size, self.num_intermediate_channel, self.layer_norm_dim)),
                       mstype.float32))
            self.right_projection_biases = Parameter(
                Tensor(np.zeros((self.batch_size, self.num_intermediate_channel)), mstype.float32))
            self.left_gate_weights = Parameter(
                Tensor(np.zeros((self.batch_size, self.num_intermediate_channel, self.layer_norm_dim)),
                       mstype.float32))
            self.left_gate_biases = Parameter(
                Tensor(np.zeros((self.batch_size, self.num_intermediate_channel)), mstype.float32))
            self.right_gate_weights = Parameter(
                Tensor(np.zeros((self.batch_size, self.num_intermediate_channel, self.layer_norm_dim)),
                       mstype.float32))
            self.right_gate_biases = Parameter(
                Tensor(np.zeros((self.batch_size, self.num_intermediate_channel)), mstype.float32))
            self.center_layer_norm_gammas = Parameter(
                Tensor(np.zeros((self.batch_size, self.layer_norm_dim)), mstype.float32))
            self.center_layer_norm_betas = Parameter(
                Tensor(np.zeros((self.batch_size, self.layer_norm_dim)), mstype.float32))
            self.output_projection_weights = Parameter(
                Tensor(np.zeros((self.batch_size, self.layer_norm_dim, self.layer_norm_dim)), mstype.float32))
            self.output_projection_biases = Parameter(
                Tensor(np.zeros((self.batch_size, self.layer_norm_dim)), mstype.float32))
            self.gating_linear_weights = Parameter(
                Tensor(np.zeros((self.batch_size, self.layer_norm_dim, self.layer_norm_dim)), mstype.float32))
            self.gating_linear_biases = Parameter(
                Tensor(np.zeros((self.batch_size, self.layer_norm_dim)), mstype.float32))
        else:
            self.layer_norm_input_gammas = Parameter(Tensor(np.ones((self.layer_norm_dim)), mstype.float32))
            self.layer_norm_input_betas = Parameter(Tensor(np.zeros((self.layer_norm_dim)), mstype.float32))
            self.left_projection_weights = Parameter(initializer(lecun_init(self.num_intermediate_channel),
                                                                 [self.num_intermediate_channel,
                                                                  self.layer_norm_dim]))
            self.left_projection_biases = Parameter(
                Tensor(np.zeros((self.num_intermediate_channel)), mstype.float32))
            self.right_projection_weights = Parameter(initializer(lecun_init(self.num_intermediate_channel),
                                                                  [self.num_intermediate_channel,
                                                                   self.layer_norm_dim]))
            self.right_projection_biases = Parameter(
                Tensor(np.zeros((self.num_intermediate_channel)), mstype.float32))
            self.left_gate_weights = Parameter(
                Tensor(np.zeros((self.num_intermediate_channel, self.layer_norm_dim)), mstype.float32))
            self.left_gate_biases = Parameter(Tensor(np.ones((self.num_intermediate_channel)), mstype.float32))
            self.right_gate_weights = Parameter(
                Tensor(np.zeros((self.num_intermediate_channel, self.layer_norm_dim)), mstype.float32))
            self.right_gate_biases = Parameter(Tensor(np.ones((self.num_intermediate_channel)), mstype.float32))
            self.center_layer_norm_gammas = Parameter(Tensor(np.ones((self.layer_norm_dim)), mstype.float32))
            self.center_layer_norm_betas = Parameter(Tensor(np.zeros((self.layer_norm_dim)), mstype.float32))
            self.output_projection_weights = Parameter(
                Tensor(np.zeros((self.layer_norm_dim, self.layer_norm_dim)), mstype.float32))
            self.output_projection_biases = Parameter(Tensor(np.zeros((self.layer_norm_dim)), mstype.float32))
            self.gating_linear_weights = Parameter(
                Tensor(np.zeros((self.layer_norm_dim, self.layer_norm_dim)), mstype.float32))
            self.gating_linear_biases = Parameter(Tensor(np.ones((self.layer_norm_dim)), mstype.float32))


class OuterProductMean(nn.Cell):
    '''outerproduct mean'''

    def __init__(self, num_outer_channel, act_dim, num_output_channel, batch_size=None, slice_num=0,
                 mixed_precision=True):
        super(OuterProductMean, self).__init__()
        self.num_output_channel = num_output_channel
        self.num_outer_channel = num_outer_channel
        self.layer_norm_input = P.LayerNorm(begin_norm_axis=-1, begin_params_axis=-1, epsilon=1e-5)
        self.matmul_trans_b = P.MatMul(transpose_b=True)
        self.matmul = P.MatMul()
        self.batch_matmul_trans_b = P.BatchMatMul(transpose_b=True)
        self.act_dim = act_dim
        self.batch_size = batch_size
        self.slice_num = slice_num
        self.idx = Tensor(0, mstype.int32)
        if mixed_precision:
            self._type = mstype.float16
        else:
            self._type = mstype.float32
        self._init_parameter()

    def construct(self, act, mask, extra_msa_norm, index):
        '''construct'''
        if self.batch_size:
            layer_norm_input_gamma = P.Gather()(self.layer_norm_input_gammas, index, 0)
            layer_norm_input_beta = P.Gather()(self.layer_norm_input_betas, index, 0)
            left_projection_weight = P.Cast()(P.Gather()(self.left_projection_weights, index, 0), self._type)
            left_projection_bias = P.Cast()(P.Gather()(self.left_projection_biases, index, 0), self._type)
            right_projection_weight = P.Cast()(P.Gather()(self.right_projection_weights, index, 0), self._type)
            right_projection_bias = P.Cast()(P.Gather()(self.right_projection_biases, index, 0), self._type)
            linear_output_weight = P.Cast()(P.Gather()(self.linear_output_weights, index, 0), self._type)
            linear_output_bias = P.Cast()(P.Gather()(self.o_biases, index, 0), self._type)
        else:
            layer_norm_input_gamma = self.layer_norm_input_gammas
            layer_norm_input_beta = self.layer_norm_input_betas
            left_projection_weight = P.Cast()(self.left_projection_weights, self._type)
            left_projection_bias = P.Cast()(self.left_projection_biases, self._type)
            right_projection_weight = P.Cast()(self.right_projection_weights, self._type)
            right_projection_bias = P.Cast()(self.right_projection_biases, self._type)
            linear_output_weight = P.Cast()(self.linear_output_weights, self._type)
            linear_output_bias = P.Cast()(self.o_biases, self._type)

        mask = P.Cast()(mask, self._type)
        mask = P.ExpandDims()(mask, -1)
        act = P.Cast()(act, mstype.float32)
        layer_norm_input_gamma = P.Cast()(layer_norm_input_gamma, mstype.float32)
        layer_norm_input_beta = P.Cast()(layer_norm_input_beta, mstype.float32)
        act, _, _ = self.layer_norm_input(act, layer_norm_input_gamma, layer_norm_input_beta)
        act = P.Cast()(act, self._type)
        act_shape = P.Shape()(act)
        if len(act_shape) != 2:
            act = P.Reshape()(act, (-1, act_shape[-1]))
        out_shape = act_shape[:-1] + (-1,)
        left_act = mask * P.Reshape()(
            P.BiasAdd()(self.matmul_trans_b(act, left_projection_weight), left_projection_bias), out_shape)
        right_act = mask * P.Reshape()(
            P.BiasAdd()(self.matmul_trans_b(act, right_projection_weight), right_projection_bias), out_shape)
        _, d, e = right_act.shape
        if self.slice_num:
            left_act_shape = P.Shape()(left_act)
            slice_shape = (left_act_shape[0],) + (self.slice_num, -1) + (left_act_shape[-1],)
            left_act = P.Reshape()(left_act, slice_shape)
            slice_idx = 0
            slice_idx_tensor = self.idx
            act_tuple = ()
            left_act_slice = P.Gather()(left_act, slice_idx_tensor, 1)
            a, b, c = left_act_slice.shape
            left_act_slice = P.Reshape()(mnp.transpose(left_act_slice, [2, 1, 0]), (-1, a))
            right_act = P.Reshape()(right_act, (a, -1))
            act_slice = P.Reshape()(P.Transpose()(P.Reshape()(self.matmul(left_act_slice, right_act),
                                                              (c, b, d, e)), (2, 1, 0, 3)), (d, b, c * e))
            act_slice_shape = P.Shape()(act_slice)
            if len(act_shape) != 2:
                act_slice = P.Reshape()(act_slice, (-1, act_slice_shape[-1]))
            act_slice = P.Reshape()(P.BiasAdd()(self.matmul_trans_b(act_slice, linear_output_weight),
                                                linear_output_bias), (d, b, -1))
            act_slice = mnp.transpose(act_slice, [1, 0, 2])
            act_slice = P.Reshape()(act_slice, ((1,) + P.Shape()(act_slice)))
            act_tuple = act_tuple + (act_slice,)
            slice_idx += 1
            slice_idx_tensor += 1
            while slice_idx < self.slice_num:
                left_act_slice = P.Gather()(left_act, slice_idx_tensor, 1)
                left_act_slice = F.depend(left_act_slice, act_tuple[-1])
                a, b, c = left_act_slice.shape
                left_act_slice = P.Reshape()(mnp.transpose(left_act_slice, [2, 1, 0]), (-1, a))
                right_act = P.Reshape()(right_act, (a, -1))
                act_slice = P.Reshape()(P.Transpose()(P.Reshape()(self.matmul(left_act_slice, right_act),
                                                                  (c, b, d, e)), (2, 1, 0, 3)), (d, b, c * e))
                act_slice_shape = P.Shape()(act_slice)
                if len(act_shape) != 2:
                    act_slice = P.Reshape()(act_slice, (-1, act_slice_shape[-1]))
                act_slice = P.Reshape()(P.BiasAdd()(self.matmul_trans_b(act_slice, linear_output_weight),
                                                    linear_output_bias), (d, b, -1))
                act_slice = mnp.transpose(act_slice, [1, 0, 2])
                act_slice = P.Reshape()(act_slice, ((1,) + P.Shape()(act_slice)))
                act_tuple = act_tuple + (act_slice,)
                slice_idx += 1
                slice_idx_tensor += 1
            act = P.Concat()(act_tuple)
            act_shape = P.Shape()(act)
            act = P.Reshape()(act, (-1, act_shape[-2], act_shape[-1]))
            epsilon = 1e-3
            tmp_mask = P.Transpose()(mask, (2, 1, 0))
            norm = P.Transpose()(self.batch_matmul_trans_b(tmp_mask, tmp_mask), (1, 2, 0)).astype(mstype.float32)
            act = P.RealDiv()(act, epsilon + norm)
            return act

        a, b, c = left_act.shape
        left_act = P.Reshape()(P.Transpose()(left_act, (2, 1, 0)), (-1, a))
        right_act = P.Reshape()(right_act, (a, -1))
        act = P.Reshape()(P.Transpose()(P.Reshape()(self.matmul(left_act, right_act),
                                                    (c, b, d, e)), (2, 1, 0, 3)), (d, b, c * e))
        act_shape = P.Shape()(act)
        if len(act_shape) != 2:
            act = P.Reshape()(act, (-1, act_shape[-1]))
        act = P.Reshape()(P.BiasAdd()(self.matmul_trans_b(act, linear_output_weight),
                                      linear_output_bias), (d, b, -1))
        act = P.Transpose()(act, (1, 0, 2))
        epsilon = 1e-3
        act = P.RealDiv()(act, epsilon + extra_msa_norm)
        return act

    def _init_parameter(self):
        '''init parameter'''
        if self.batch_size:
            self.layer_norm_input_gammas = Parameter(Tensor(np.zeros((self.batch_size, self.act_dim)), mstype.float32))
            self.layer_norm_input_betas = Parameter(Tensor(np.zeros((self.batch_size, self.act_dim)), mstype.float32))
            self.left_projection_weights = Parameter(
                Tensor(np.zeros((self.batch_size, self.num_outer_channel, self.act_dim)), mstype.float32))
            self.left_projection_biases = Parameter(
                Tensor(np.zeros((self.batch_size, self.num_outer_channel)), mstype.float32))
            self.right_projection_weights = Parameter(
                Tensor(np.zeros((self.batch_size, self.num_outer_channel, self.act_dim)), mstype.float32))
            self.right_projection_biases = Parameter(
                Tensor(np.zeros((self.batch_size, self.num_outer_channel)), mstype.float32))
            self.linear_output_weights = Parameter(Tensor(np.zeros(
                (self.batch_size, self.num_output_channel, self.num_outer_channel *
                 self.num_outer_channel)), mstype.float32))
            self.o_biases = Parameter(Tensor(np.zeros((self.batch_size, self.num_output_channel)), mstype.float32))
        else:
            self.layer_norm_input_gammas = Parameter(Tensor(np.ones((self.act_dim)), mstype.float32))
            self.layer_norm_input_betas = Parameter(Tensor(np.zeros((self.act_dim)), mstype.float32))
            self.left_projection_weights = Parameter(
                initializer(lecun_init(self.act_dim), [self.num_outer_channel, self.act_dim]))
            self.left_projection_biases = Parameter(Tensor(np.zeros((self.num_outer_channel)), mstype.float32))
            self.right_projection_weights = Parameter(
                initializer(lecun_init(self.act_dim), [self.num_outer_channel, self.act_dim]))
            self.right_projection_biases = Parameter(Tensor(np.zeros((self.num_outer_channel)), mstype.float32))
            self.linear_output_weights = Parameter(
                Tensor(np.zeros((self.num_output_channel, self.num_outer_channel * self.num_outer_channel)),
                       mstype.float32))
            self.o_biases = Parameter(Tensor(np.zeros((self.num_output_channel)), mstype.float32))
