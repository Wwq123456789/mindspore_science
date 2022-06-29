# Copyright 2022 Huawei Technologies Co., Ltd & CPL YiQin GAO Research Group
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
"""model"""
import numpy as np
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.numpy as mnp
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
from mindspore import Parameter
from mindsponge.common.utils import pseudo_beta_fn, dgram_from_positions, atom37_to_torsion_angles, get_chi_atom_indices
from mindsponge.core.layer.initializer import lecun_init
import mindsponge.common.residue_constants as residue_constants
from template_embedding import TemplateEmbedding
from evoformer import Evoformer
from structure import StructureModule


def caculate_constant_array(seq_length):
    '''constant array'''
    chi_atom_indices = np.array(get_chi_atom_indices()).astype(np.int32)
    chi_angles_mask = list(residue_constants.chi_angles_mask)
    chi_angles_mask.append([0.0, 0.0, 0.0, 0.0])
    chi_angles_mask = np.array(chi_angles_mask).astype(np.float32)
    mirror_psi_mask = np.float32(np.asarray([1., 1., -1., 1., 1., 1., 1.])[None, None, :, None])
    chi_pi_periodic = np.float32(np.array(residue_constants.chi_pi_periodic))

    indices0 = np.arange(4).reshape((-1, 1, 1, 1, 1)).astype("int32")  # 4 batch
    indices0 = indices0.repeat(seq_length, axis=1)  # seq_length sequence length
    indices0 = indices0.repeat(4, axis=2)  # 4 chis
    indices0 = indices0.repeat(4, axis=3)  # 4 atoms

    indices1 = np.arange(seq_length).reshape((1, -1, 1, 1, 1)).astype("int32")
    indices1 = indices1.repeat(4, axis=0)
    indices1 = indices1.repeat(4, axis=2)
    indices1 = indices1.repeat(4, axis=3)

    constant_array = [chi_atom_indices, chi_angles_mask, mirror_psi_mask, chi_pi_periodic, indices0, indices1]
    constant_array = [Tensor(val) for val in constant_array]
    return constant_array


class MFold(nn.Cell):
    """Mfold"""

    def __init__(self, config, mixed_precision=True):
        super(MFold, self).__init__()

        self.cfg = config

        if mixed_precision:
            self._type = mstype.float16
        else:
            self._type = mstype.float32
        self.is_training = self.cfg.is_training
        self.recycle_pos = self.cfg.recycle_pos
        self.recycle_features = self.cfg.recycle_features
        self.max_relative_feature = self.cfg.max_relative_feature
        self.num_bins = self.cfg.prev_pos.num_bins
        self.min_bin = self.cfg.prev_pos.min_bin
        self.max_bin = self.cfg.prev_pos.max_bin
        self.template_enabled = self.cfg.template.enabled
        self.template_embed_torsion_angles = self.cfg.template.embed_torsion_angles
        self.extra_msa_stack_num = self.cfg.evoformer.extra_msa_stack_num
        self.msa_stack_num = self.cfg.evoformer.msa_stack_num
        self.chi_atom_indices, self.chi_angles_mask, self.mirror_psi_mask, self.chi_pi_periodic, \
        self.indices0, self.indices1 = caculate_constant_array(self.cfg.seq_length)

        self.preprocess_1d = nn.Dense(self.cfg.common.target_feat_dim, self.cfg.msa_channel,
                                      weight_init=lecun_init(self.cfg.common.target_feat_dim))
        self.preprocess_msa = nn.Dense(self.cfg.common.msa_feat_dim, self.cfg.msa_channel,
                                       weight_init=lecun_init(self.cfg.common.msa_feat_dim))
        self.left_single = nn.Dense(self.cfg.common.target_feat_dim, self.cfg.pair_channel,
                                    weight_init=lecun_init(self.cfg.common.target_feat_dim))
        self.right_single = nn.Dense(self.cfg.common.target_feat_dim, self.cfg.pair_channel,
                                     weight_init=lecun_init(self.cfg.common.target_feat_dim))
        self.prev_pos_linear = nn.Dense(self.cfg.common.dgram_dim, self.cfg.pair_channel,
                                        weight_init=lecun_init(self.cfg.common.dgram_dim))
        self.pair_activations = nn.Dense(self.cfg.common.pair_in_dim, self.cfg.pair_channel,
                                         weight_init=lecun_init(self.cfg.common.pair_in_dim))
        self.extra_msa_one_hot = nn.OneHot(depth=23, axis=-1)
        self.template_aatype_one_hot = nn.OneHot(depth=22, axis=-1)
        self.prev_msa_first_row_norm = nn.LayerNorm([256,], epsilon=1e-5)
        self.prev_pair_norm = nn.LayerNorm([128,], epsilon=1e-5)
        self.one_hot = nn.OneHot(depth=self.cfg.max_relative_feature * 2 + 1, axis=-1)
        self.extra_msa_activations = nn.Dense(25, self.cfg.extra_msa_channel, weight_init=lecun_init(25))
        self.template_embedding = TemplateEmbedding(self.cfg.template, self.cfg.seq_length, mixed_precision)

        self.matmul_trans_b = P.MatMul(transpose_b=True)
        self.batch_matmul_trans_b = P.BatchMatMul(transpose_b=True)
        self.template_single_embedding = nn.Dense(57, self.cfg.msa_channel,
                                                  weight_init=
                                                  lecun_init(57, initializer_name='relu'))
        self.template_projection = nn.Dense(self.cfg.msa_channel, self.cfg.msa_channel,
                                            weight_init=lecun_init(self.cfg.msa_channel,
                                                                   initializer_name='relu'))
        self.relu = nn.ReLU()
        self.single_activations = nn.Dense(self.cfg.msa_channel, self.cfg.seq_channel,
                                           weight_init=lecun_init(self.cfg.msa_channel))
        extra_msa_stack = nn.CellList()
        for _ in range(self.extra_msa_stack_num):
            extra_msa_block = Evoformer(self.cfg,
                                        msa_act_dim=64,
                                        pair_act_dim=128,
                                        is_extra_msa=True,
                                        batch_size=None,
                                        mixed_precision=mixed_precision)
            extra_msa_stack.append(extra_msa_block)
        self.extra_msa_stack = extra_msa_stack
        if self.is_training:
            msa_stack = nn.CellList()
            for _ in range(self.msa_stack_num):
                msa_block = Evoformer(self.cfg,
                                      msa_act_dim=256,
                                      pair_act_dim=128,
                                      is_extra_msa=False,
                                      batch_size=None,
                                      mixed_precision=mixed_precision)
                msa_stack.append(msa_block)
            self.msa_stack = msa_stack
        else:
            self.msa_stack = Evoformer(self.cfg,
                                       msa_act_dim=256,
                                       pair_act_dim=128,
                                       is_extra_msa=False,
                                       batch_size=self.msa_stack_num,
                                       mixed_precision=mixed_precision)
        self.idx_evoformer_block = Parameter(Tensor(0, mstype.int32), requires_grad=False)
        self.evoformer_num_block_eval = Tensor(self.msa_stack_num, mstype.int32)

        self.structure_module = StructureModule(self.cfg,
                                                self.cfg.seq_channel,
                                                self.cfg.pair_channel,
                                                mixed_precision)

    def construct(self, target_feat, msa_feat, msa_mask, seq_mask, aatype,
                  template_aatype, template_all_atom_masks, template_all_atom_positions,
                  template_mask, template_pseudo_beta_mask, template_pseudo_beta, extra_msa, extra_has_deletion,
                  extra_deletion_value, extra_msa_mask,
                  residx_atom37_to_atom14, atom37_atom_exists, residue_index,
                  prev_pos, prev_msa_first_row, prev_pair):
        """construct"""

        preprocess_1d = self.preprocess_1d(target_feat)
        preprocess_msa = self.preprocess_msa(msa_feat)
        msa_activations = mnp.expand_dims(preprocess_1d, axis=0) + preprocess_msa
        left_single = self.left_single(target_feat)
        right_single = self.right_single(target_feat)
        pair_activations = P.ExpandDims()(left_single, 1) + P.ExpandDims()(right_single, 0)
        mask_2d = P.ExpandDims()(seq_mask, 1) * P.ExpandDims()(seq_mask, 0)
        if self.recycle_pos:
            prev_pseudo_beta = pseudo_beta_fn(aatype, prev_pos, None)
            dgram = dgram_from_positions(prev_pseudo_beta, self.num_bins, self.min_bin, self.max_bin)
            pair_activations += self.prev_pos_linear(dgram)

        if self.recycle_features:
            prev_msa_first_row = self.prev_msa_first_row_norm(prev_msa_first_row)
            msa_activations = mnp.concatenate(
                (mnp.expand_dims(prev_msa_first_row + msa_activations[0, ...], 0), msa_activations[1:, ...]), 0)
            pair_activations += self.prev_pair_norm(prev_pair)

        if self.max_relative_feature:
            offset = P.ExpandDims()(residue_index, 1) - P.ExpandDims()(residue_index, 0)
            rel_pos = self.one_hot(mnp.clip(offset + self.max_relative_feature, 0, 2 * self.max_relative_feature))
            pair_activations += self.pair_activations(rel_pos)

        template_pair_representation = 0
        if self.template_enabled:
            template_pair_representation = self.template_embedding(pair_activations, template_aatype,
                                                                   template_all_atom_masks, template_all_atom_positions,
                                                                   template_mask, template_pseudo_beta_mask,
                                                                   template_pseudo_beta, mask_2d)
            pair_activations += template_pair_representation
        msa_1hot = self.extra_msa_one_hot(extra_msa)
        extra_msa_feat = mnp.concatenate((msa_1hot, extra_has_deletion[..., None], extra_deletion_value[..., None]),
                                         axis=-1)
        extra_msa_activations = self.extra_msa_activations(extra_msa_feat)
        extra_msa_mask_tmp = P.Transpose()(P.ExpandDims()(extra_msa_mask, -1), (2, 1, 0))
        extra_msa_norm = P.Transpose()(self.batch_matmul_trans_b(extra_msa_mask_tmp, extra_msa_mask_tmp), (1, 2, 0))
        for i in range(self.extra_msa_stack_num):
            extra_msa_activations, pair_activations = \
                self.extra_msa_stack[i](extra_msa_activations, pair_activations, extra_msa_mask, extra_msa_norm,
                                        mask_2d)

        if self.template_enabled and self.template_embed_torsion_angles:
            num_templ, num_res = template_aatype.shape
            aatype_one_hot = self.template_aatype_one_hot(template_aatype)
            torsion_angles_sin_cos, alt_torsion_angles_sin_cos, torsion_angles_mask = atom37_to_torsion_angles(
                template_aatype, template_all_atom_positions, template_all_atom_masks, self.chi_atom_indices,
                self.chi_angles_mask, self.mirror_psi_mask, self.chi_pi_periodic, self.indices0, self.indices1)
            template_features = mnp.concatenate([aatype_one_hot,
                                                 mnp.reshape(torsion_angles_sin_cos, [num_templ, num_res, 14]),
                                                 mnp.reshape(alt_torsion_angles_sin_cos, [num_templ, num_res, 14]),
                                                 torsion_angles_mask], axis=-1)
            template_activations = self.template_single_embedding(template_features)
            template_activations = self.relu(template_activations)
            template_activations = self.template_projection(template_activations)
            msa_activations = mnp.concatenate([msa_activations, template_activations], axis=0)
            torsion_angle_mask = torsion_angles_mask[:, :, 2]
            msa_mask = mnp.concatenate([msa_mask, torsion_angle_mask], axis=0)

        msa_mask_tmp = P.Transpose()(P.ExpandDims()(msa_mask, -1), (2, 1, 0))
        msa_mask_norm = P.Transpose()(self.batch_matmul_trans_b(msa_mask_tmp, msa_mask_tmp), (1, 2, 0))
        if self.is_training:
            for i in range(self.msa_stack_num):
                msa_activations, pair_activations = self.msa_stack[i](msa_activations, pair_activations, msa_mask,
                                                                      msa_mask_norm, mask_2d)
        else:
            self.idx_evoformer_block = self.idx_evoformer_block * 0
            while self.idx_evoformer_block < self.evoformer_num_block_eval:
                msa_activations, pair_activations = self.msa_stack(msa_activations,
                                                                   pair_activations,
                                                                   msa_mask,
                                                                   msa_mask_norm,
                                                                   mask_2d,
                                                                   self.idx_evoformer_block)
                self.idx_evoformer_block += 1
        single_activations = self.single_activations(msa_activations[0])
        num_sequences = msa_feat.shape[0]
        msa = msa_activations[:num_sequences, :, :]
        msa_first_row = msa_activations[0]

        final_atom_positions, final_atom_mask, rp_structure_module, atom14_pred_positions, final_affines, \
        angles_sin_cos_new, um_angles_sin_cos_new, sidechain_frames, sidechain_atom_pos, structure_traj = \
            self.structure_module(single_activations,
                                  pair_activations,
                                  seq_mask,
                                  aatype,
                                  residx_atom37_to_atom14,
                                  atom37_atom_exists)
        prev_pos = final_atom_positions
        prev_msa_first_row = msa_first_row
        prev_pair = pair_activations
        res = (final_atom_positions, final_atom_mask, prev_pos, prev_msa_first_row, prev_pair, \
               msa, rp_structure_module, atom14_pred_positions, final_affines, angles_sin_cos_new, \
               um_angles_sin_cos_new, sidechain_frames, sidechain_atom_pos, structure_traj)
        return res
