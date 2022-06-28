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
"""basic"""
import numpy as np
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore import Parameter
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
from .initializer import glorot_uniform


class Attention(nn.Cell):
    '''attention module'''

    def __init__(self, num_head, key_dim, value_dim, gating, q_data_dim, m_data_dim, output_dim, batch_size=None,
                 mixed_precision=True):
        super(Attention, self).__init__()
        self.q_data_dim = q_data_dim
        self.m_data_dim = m_data_dim
        self.output_dim = output_dim
        self.num_head = num_head
        self.gating = gating
        self.key_dim = key_dim if key_dim else int(q_data_dim)
        self.value_dim = value_dim if value_dim else int(m_data_dim)
        self.key_dim = self.key_dim // self.num_head
        self.value_dim = self.value_dim // self.num_head
        self.batch_size = batch_size
        self.matmul = P.MatMul(transpose_b=True)
        self.batch_matmul_trans_b = P.BatchMatMul(transpose_b=True)
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.batch_size = batch_size
        if mixed_precision:
            self._type = mstype.float16
        else:
            self._type = mstype.float32
        self._init_parameter()

    def construct(self, q_data, m_data, bias, index=None, nonbatched_bias=None):
        '''construct'''

        q_data = P.Cast()(q_data, self._type)
        m_data = P.Cast()(m_data, self._type)

        if self.batch_size:
            linear_q_weight = P.Cast()(P.Gather()(self.linear_q_weights, index, 0), self._type)
            linear_k_weight = P.Cast()(P.Gather()(self.linear_k_weights, index, 0), self._type)
            linear_v_weight = P.Cast()(P.Gather()(self.linear_v_weights, index, 0), self._type)
            linear_output_weight = P.Cast()(P.Gather()(self.linear_output_weights, index, 0), self._type)
            o_bias = P.Cast()(P.Gather()(self.o_biases, index, 0), self._type)
            linear_gating_weight = 0
            gating_bias = 0
            if self.gating:
                linear_gating_weight = P.Cast()(P.Gather()(self.linear_gating_weights, index, 0), self._type)
                gating_bias = P.Cast()(P.Gather()(self.gating_biases, index, 0), self._type)
        else:
            linear_q_weight = P.Cast()(self.linear_q_weights, self._type)
            linear_k_weight = P.Cast()(self.linear_k_weights, self._type)
            linear_v_weight = P.Cast()(self.linear_v_weights, self._type)
            linear_output_weight = P.Cast()(self.linear_output_weights, self._type)
            o_bias = P.Cast()(self.o_biases, self._type)
            linear_gating_weight = 0
            gating_bias = 0
            if self.gating:
                linear_gating_weight = P.Cast()(self.linear_gating_weights, self._type)
                gating_bias = P.Cast()(self.gating_biases, self._type)

        dim_b, dim_q, dim_a = q_data.shape
        _, dim_k, dim_c = m_data.shape
        dim_h = self.num_head

        q_data = P.Reshape()(q_data, (-1, dim_a))
        m_data = P.Reshape()(m_data, (-1, dim_c))

        q = self.matmul(q_data, linear_q_weight) * self.key_dim ** (-0.5)
        k = self.matmul(m_data, linear_k_weight)
        v = self.matmul(m_data, linear_v_weight)

        q = P.Reshape()(q, (dim_b, dim_q, dim_h, -1))
        k = P.Reshape()(k, (dim_b, dim_k, dim_h, -1))
        v = P.Reshape()(v, (dim_b, dim_k, dim_h, -1))

        tmp_q = P.Reshape()(P.Transpose()(q.astype(self._type), (0, 2, 1, 3)), (dim_b * dim_h, dim_q, -1))
        tmp_k = P.Reshape()(P.Transpose()(k.astype(self._type), (0, 2, 1, 3)), (dim_b * dim_h, dim_k, -1))
        bias = P.Cast()(bias, mstype.float32)
        logits = P.Add()(P.Cast()(P.Reshape()(self.batch_matmul_trans_b(tmp_q, tmp_k), (dim_b, dim_h, dim_q, dim_k)),
                                  mstype.float32), bias)

        if nonbatched_bias is not None:
            bias = P.Cast()(P.ExpandDims()(nonbatched_bias, 0), mstype.float32)
            logits = P.Add()(logits, bias)
        weights = self.softmax(logits)
        weights = P.Cast()(weights, self._type)
        tmp_v = P.Reshape()(P.Transpose()(v, (0, 2, 3, 1)), (dim_b * dim_h, -1, dim_k))
        tmp_weights = P.Reshape()(weights, (dim_b * dim_h, dim_q, -1))
        weighted_avg = P.Transpose()(
            P.Reshape()(self.batch_matmul_trans_b(tmp_weights, tmp_v), (dim_b, dim_h, dim_q, -1)), (0, 2, 1, 3))

        if self.gating:
            gating_bias = P.ExpandDims()(P.ExpandDims()(gating_bias, 0), 0)
            gate_values = P.Add()(P.Reshape()(self.matmul(q_data, linear_gating_weight), (dim_b, dim_q, dim_h, -1)),
                                  gating_bias)
            gate_values = P.Cast()(gate_values, mstype.float32)
            gate_values = self.sigmoid(gate_values)
            gate_values = P.Cast()(gate_values, self._type)
            weighted_avg = P.Reshape()(weighted_avg * gate_values, (dim_b * dim_q, -1))

        weighted_avg = P.Reshape()(weighted_avg, (dim_b * dim_q, -1))
        output = P.Add()(P.Reshape()(self.matmul(weighted_avg, linear_output_weight), (dim_b, dim_q, -1)),
                         P.ExpandDims()(o_bias, 0))
        return output

    def _init_parameter(self):
        '''init parameter'''
        if self.batch_size:
            self.linear_q_weights = Parameter(Tensor(np.zeros([self.batch_size, self.num_head * self.key_dim,
                                                               self.q_data_dim]), mstype.float32))
            self.linear_k_weights = Parameter(Tensor(np.zeros([self.batch_size, self.num_head * self.key_dim,
                                                               self.m_data_dim]), mstype.float32))
            self.linear_v_weights = Parameter(Tensor(np.zeros([self.batch_size, self.num_head * self.value_dim,
                                                               self.m_data_dim]), mstype.float32))
            self.linear_output_weights = Parameter(Tensor(np.zeros([self.batch_size, self.output_dim,
                                                                    self.num_head * self.value_dim]), mstype.float32))
            self.o_biases = Parameter(Tensor(np.zeros([self.batch_size, self.output_dim]), mstype.float32))
            if self.gating:
                self.linear_gating_weights = Parameter(Tensor(np.zeros([self.batch_size, self.num_head * self.value_dim,
                                                                        self.q_data_dim]), mstype.float32))
                self.gating_biases = Parameter(Tensor(np.zeros((self.batch_size, self.num_head, self.value_dim)),
                                                      mstype.float32), name="gating_b")
        else:
            self.linear_q_weights = Parameter(Tensor(
                glorot_uniform(self.num_head * self.q_data_dim, self.key_dim * self.q_data_dim,
                               [self.num_head * self.key_dim, self.q_data_dim]), mstype.float32))
            self.linear_k_weights = Parameter(Tensor(
                glorot_uniform(self.num_head * self.m_data_dim, self.key_dim * self.m_data_dim,
                               [self.num_head * self.key_dim, self.m_data_dim]), mstype.float32))
            self.linear_v_weights = Parameter(Tensor(
                glorot_uniform(self.num_head * self.m_data_dim, self.value_dim * self.m_data_dim,
                               [self.num_head * self.value_dim, self.m_data_dim]), mstype.float32))
            self.linear_output_weights = Parameter(
                Tensor(np.zeros([self.output_dim, self.num_head * self.value_dim]), mstype.float32))
            self.o_biases = Parameter(Tensor(np.zeros([self.output_dim]), mstype.float32))
            if self.gating:
                self.linear_gating_weights = Parameter(
                    Tensor(np.zeros([self.num_head * self.value_dim, self.q_data_dim]), mstype.float32))
                self.gating_biases = Parameter(Tensor(np.ones((self.num_head, self.value_dim)), mstype.float32),
                                               name="gating_b")


class GlobalAttention(nn.Cell):
    '''global attention'''

    def __init__(self, num_head, gating, key_dim, value_dim, output_dim, batch_size=None, mixed_precision=True):
        super(GlobalAttention, self).__init__()
        self.key_dim = key_dim
        self.ori_key_dim = key_dim
        self.value_dim = value_dim
        self.ori_value_dim = value_dim
        self.num_head = num_head
        self.key_dim = self.key_dim // self.num_head
        self.value_dim = self.value_dim // self.num_head
        self.output_dim = output_dim
        self.matmul_trans_b = P.MatMul(transpose_b=True)
        self.batch_matmul = P.BatchMatMul()
        self.batch_matmul_trans_b = P.BatchMatMul(transpose_b=True)
        self.matmul = P.MatMul()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.gating = gating
        if mixed_precision:
            self._type = mstype.float16
        else:
            self._type = mstype.float32
        self.batch_size = batch_size
        self._init_parameter()

    def construct(self, q_data, m_data, q_mask, bias, index):
        '''construct'''
        if self.batch_size:
            q_data = P.Cast()(q_data, self._type)
            q_weights = P.Cast()(P.Gather()(self.linear_q_weights, index, 0), self._type)
            k_weights = P.Cast()(P.Gather()(self.linear_k_weights, index, 0), self._type)
            v_weights = P.Cast()(P.Gather()(self.linear_v_weights, index, 0), self._type)
            output_weights = P.Cast()(P.Gather()(self.linear_output_weights, index, 0), self._type)
            output_bias = P.Cast()(P.Gather()(self.o_biases, index, 0), self._type)
            gating_weights = 0
            gating_bias = 0
            if self.gating:
                gating_weights = P.Gather()(self.linear_gating_weights, index, 0)
                gating_weights = P.Cast()(gating_weights, self._type)
                gating_bias = P.Cast()(P.Gather()(self.gating_biases, index, 0), self._type)
        else:
            q_mask = P.Cast()(q_mask, self._type)
            q_data = P.Cast()(q_data, self._type)
            q_weights = P.Cast()(self.linear_q_weights, self._type)
            k_weights = P.Cast()(self.linear_k_weights, self._type)
            v_weights = P.Cast()(self.linear_v_weights, self._type)
            output_weights = P.Cast()(self.linear_output_weights, self._type)
            output_bias = P.Cast()(self.o_biases, self._type)
            gating_weights = 0
            gating_bias = 0
            if self.gating:
                gating_weights = self.linear_gating_weights
                gating_weights = P.Cast()(gating_weights, self._type)
                gating_bias = P.Cast()(self.gating_biases, self._type)

        b, _, _ = m_data.shape

        v_weights = P.ExpandDims()(v_weights, 0)
        v_weights = P.BroadcastTo((b, self.value_dim * self.num_head, self.value_dim))(v_weights)
        v = self.batch_matmul(m_data, v_weights)
        q_mask = P.Cast()(q_mask, mstype.float32)
        q_data = P.Cast()(q_data, mstype.float32)

        mask_shape = q_mask.shape
        value_shape = q_data.shape
        broadcast_factor = 1.
        value_size = value_shape[1]
        mask_size = mask_shape[1]
        if mask_size == 1:
            broadcast_factor = broadcast_factor * value_size
        qa = P.ReduceSum()(q_mask * q_data, 1)
        qb = P.ReduceSum()(q_mask, 1) * broadcast_factor + 1e-10
        q_avg = P.Cast()(P.RealDiv()(qa, qb), self._type)

        q_data = P.Cast()(q_data, self._type)
        q_weights = P.Reshape()(q_weights, (-1, self.num_head * self.key_dim))
        q = P.Reshape()(self.matmul(q_avg, q_weights),
                        (-1, self.num_head, self.key_dim)) * (self.key_dim ** (-0.5))

        k_weights = P.ExpandDims()(k_weights, 0)
        k_weights = P.BroadcastTo((b, self.value_dim * self.num_head, self.key_dim))(k_weights)
        k = self.batch_matmul(m_data, k_weights)

        bias = 1e9 * (P.Transpose()(q_mask, (0, 2, 1)) - 1.0)
        logits = P.Add()(P.Cast()(self.batch_matmul_trans_b(q, k), mstype.float32), bias)

        weights = self.softmax(logits)
        weights = P.Cast()(weights, self._type)
        weighted_avg = self.batch_matmul(weights, v)

        if self.gating:
            q_data_shape = P.Shape()(q_data)
            if len(q_data_shape) != 2:
                q_data = P.Reshape()(q_data, (-1, q_data_shape[-1]))
            out_shape = q_data_shape[:-1] + (-1,)
            gate_values = P.Reshape()(self.matmul_trans_b(q_data, gating_weights) + gating_bias, out_shape)

            gate_values = P.Cast()(gate_values, mstype.float32)
            gate_values = P.Reshape()(self.sigmoid(gate_values), (b, -1, self.num_head, self.value_dim))
            gate_values = P.Cast()(gate_values, self._type)
            weighted_avg = P.Reshape()(P.ExpandDims()(weighted_avg, 1) * gate_values,
                                       (-1, self.num_head * self.value_dim))
            weighted_avg_shape = P.Shape()(weighted_avg)
            if len(weighted_avg_shape) != 2:
                weighted_avg = P.Reshape()(weighted_avg, (-1, weighted_avg_shape[-1]))
            output = P.Reshape()(P.Add()(self.matmul_trans_b(weighted_avg,
                                                             output_weights), output_bias),
                                 (b, -1, self.output_dim))
        else:
            weighted_avg = P.Reshape()(weighted_avg, (-1, self.num_head * self.value_dim))
            weighted_avg_shape = P.Shape()(weighted_avg)
            if len(weighted_avg_shape) != 2:
                weighted_avg = P.Reshape()(weighted_avg, (-1, weighted_avg_shape[-1]))
            out_shape = weighted_avg_shape[:-1] + (-1,)
            output = P.Reshape()(P.Add()(self.matmul_trans_b(weighted_avg,
                                                             output_weights), output_bias), out_shape)
            output = P.ExpandDims()(output, -1)
        return output

    def _init_parameter(self):
        '''init parameter'''
        if self.batch_size:
            self.linear_q_weights = Parameter(
                Tensor(np.zeros((self.batch_size, self.ori_key_dim, self.num_head, self.key_dim)), mstype.float32))
            self.linear_k_weights = Parameter(
                Tensor(np.zeros((self.batch_size, self.ori_value_dim, self.key_dim)), mstype.float32))
            self.linear_v_weights = Parameter(
                Tensor(np.zeros((self.batch_size, self.ori_value_dim, self.value_dim)), mstype.float32))
            self.linear_output_weights = Parameter(
                Tensor(np.zeros((self.batch_size, self.output_dim, self.num_head * self.value_dim)), mstype.float32))
            self.o_biases = Parameter(Tensor(np.zeros((self.batch_size, self.output_dim)), mstype.float32))
            if self.gating:
                self.linear_gating_weights = Parameter(
                    Tensor(np.zeros((self.batch_size, self.num_head * self.value_dim, self.ori_key_dim)),
                           mstype.float32))
                self.gating_biases = Parameter(Tensor(np.zeros((self.batch_size, self.ori_key_dim)), mstype.float32))
        else:
            self.linear_q_weights = Parameter(Tensor(
                glorot_uniform(self.num_head * self.ori_key_dim, self.key_dim * self.ori_key_dim,
                               (self.ori_key_dim, self.num_head, self.key_dim)), mstype.float32))
            self.linear_k_weights = Parameter(
                Tensor(glorot_uniform(self.ori_value_dim, self.key_dim, (self.ori_value_dim, self.key_dim)),
                       mstype.float32))
            self.linear_v_weights = Parameter(
                Tensor(glorot_uniform(self.ori_value_dim, self.value_dim, (self.ori_value_dim, self.value_dim)),
                       mstype.float32))
            self.linear_output_weights = Parameter(
                Tensor(np.zeros((self.output_dim, self.num_head * self.value_dim)), mstype.float32))
            self.o_biases = Parameter(Tensor(np.zeros((self.output_dim)), mstype.float32))
            if self.gating:
                self.linear_gating_weights = Parameter(
                    Tensor(np.zeros((self.num_head * self.value_dim, self.ori_key_dim)), mstype.float32))
                self.gating_biases = Parameter(Tensor(np.ones((self.ori_key_dim)), mstype.float32))
