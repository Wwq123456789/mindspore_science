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
import mindspore.context as context
from mindspore import Tensor, Model, nn
from mindsponge.common.config_load import load_config
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
    raw_feature = load_pkl(args.pkl_path)
    data_cfg = load_config(args.data_config)
    model_cfg = load_config(args.model_config)
    model_cfg.seq_length = data_cfg.eval.crop_size
    SLICE_KEY = "seq_" + str(model_cfg.seq_length)
    slice_val = vars(model_cfg.slice)[SLICE_KEY]
    model_cfg.slice = slice_val
    processed_feature = Feature(data_cfg, raw_feature)
    feat, prev_pos, prev_msa_first_row, prev_pair = processed_feature.pipeline(data_cfg,
                                                                               mixed_precision=args.mixed_precision)
    structure_prediction = MegaFold(model_cfg, mixed_precision=args.mixed_precision)
    opt = nn.Adam(params=structure_prediction.trainable_params(), learning_rate=1e-4, eps=1e-6)
    if args.mixed_precision:
        AMP_LEVEL = 'O3'
    else:
        AMP_LEVEL = 'O0'
    megafold = Model(structure_prediction, optimizer=opt, amp_level=AMP_LEVEL)
    for i in range(1):
        feat_i = [Tensor(x[i]) for x in feat]
        result = megafold.predict(*feat_i, prev_pos, prev_msa_first_row, prev_pair)
        for val in result:
            print(val.shape, val.dtype)
