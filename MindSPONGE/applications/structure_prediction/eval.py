# Copyright 2022 Huawei Technologies Co., Ltd
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
"""eval script"""
import argparse
import pickle
import os
import time
import mindspore.context as context
import mindspore.common.dtype as mstype
from mindspore import Tensor
from mindsponge.cell.initializer import do_keep_cell_fp32
from mindsponge.common.config_load import load_config
from mindsponge.common.protein import to_pdb, from_prediction
from data import Feature
from model import MegaFold


parser = argparse.ArgumentParser(description='Inputs for eval.py')
parser.add_argument('--data_config', help='data process config')
parser.add_argument('--model_config', help='model config')
parser.add_argument('--pkl_path', help='processed raw feature path')
parser.add_argument('--device_id', default=1, type=int, help='DEVICE_ID')
parser.add_argument('--mixed_precision', default=1, type=int, help='whether to use mixed precision')
parser.add_argument('--run_platform', default='Ascend', type=str, help='which platform to use, Ascend or GPU')
args = parser.parse_args()


def load_pkl(pickle_path):
    f = open(pickle_path, "rb")
    data = pickle.load(f)
    f.close()
    return data


if __name__ == "__main__":
    if args.run_platform == 'Ascend':
        context.set_context(mode=context.GRAPH_MODE,
                            device_target="Ascend",
                            max_device_memory="31GB",
                            device_id=args.device_id)
    elif args.run_platform == 'GPU':
        context.set_context(mode=context.GRAPH_MODE,
                            device_target="GPU",
                            max_device_memory="31GB",
                            device_id=args.device_id,
                            enable_graph_kernel=True,
                            graph_kernel_flags="--enable_expand_ops_only=Softmax --enable_cluster_ops_only=Add")
    else:
        raise Exception("Only support GPU or Ascend")
    data_cfg = load_config(args.data_config)
    model_cfg = load_config(args.model_config)
    model_cfg.seq_length = data_cfg.eval.crop_size
    SLICE_KEY = "seq_" + str(model_cfg.seq_length)
    slice_val = vars(model_cfg.slice)[SLICE_KEY]
    model_cfg.slice = slice_val

    megafold = MegaFold(model_cfg, mixed_precision=args.mixed_precision)
    if args.mixed_precision:
        megafold.to_float(mstype.float16)
        do_keep_cell_fp32(megafold)
    else:
        megafold.to_float(mstype.float32)

    seq_files = os.listdir(args.pkl_path)
    for seq_file in seq_files:
        seq_name = seq_file.split('.')[0]
        raw_feature = load_pkl(args.pkl_path + seq_file)
        ori_res_length = raw_feature['msa'].shape[1]
        processed_feature = Feature(data_cfg, raw_feature)
        feat, prev_pos, prev_msa_first_row, prev_pair = processed_feature.pipeline(data_cfg,
                                                                                   mixed_precision=args.mixed_precision)
        t1 = time.time()
        for i in range(data_cfg.common.num_recycle):
            feat_i = [Tensor(x[i]) for x in feat]
            prev_pos, prev_msa_first_row, prev_pair = megafold(*feat_i, prev_pos, prev_msa_first_row, prev_pair)
        t2 = time.time()
        final_atom_positions = prev_pos.asnumpy()[:ori_res_length]
        final_atom_mask = feat[16][0][:ori_res_length]
        unrelaxed_protein = from_prediction(final_atom_positions, final_atom_mask,
                                            feat[4][0][:ori_res_length], feat[17][0][:ori_res_length])
        pdb_file = to_pdb(unrelaxed_protein)
        os.makedirs(f'./result/seq_{seq_name}_{model_cfg.seq_length}', exist_ok=True)
        with open(os.path.join(f'./result/seq_{seq_name}_{model_cfg.seq_length}',
                               f'unrelaxed_model_{seq_name}.pdb'), 'w') as file:
            file.write(pdb_file)
