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
import time
import os
import json
import argparse
import numpy as np

# import mindspore.context as context
# from mindspore.common.tensor import Tensor
# from mindspore import load_checkpoint

# from data.feature.feature_extraction import process_features
# from data.tools.data_process import data_process
# from commons.generate_pdb import to_pdb, from_prediction
# from commons.utils import compute_confidence
# from model import AlphaFold
from config import config, global_config
parser = argparse.ArgumentParser(description='Inputs for eval.py')
# parser = argparse.ArgumentParser(description='Inputs for run.py')
parser.add_argument('--seq_length', help='padding sequence length')
# parser.add_argument('--input_fasta_path', help='Path of FASTA files folder directory to be predicted.')
# parser.add_argument('--msa_result_path', help='Path to save msa result.')
# parser.add_argument('--database_dir', help='Path of data to generate msa.')
# parser.add_argument('--database_envdb_dir', help='Path of expandable data to generate msa.')
# parser.add_argument('--hhsearch_binary_path', help='Path of hhsearch executable.')
# parser.add_argument('--pdb70_database_path', help='Path to pdb70.')
# parser.add_argument('--template_mmcif_dir', help='Path of template mmcif.')
# parser.add_argument('--max_template_date', help='Maximum template release date.')
# parser.add_argument('--kalign_binary_path', help='Path to kalign executable.')
# parser.add_argument('--obsolete_pdbs_path', help='Path to obsolete pdbs path.')
# parser.add_argument('--checkpoint_path', help='Path of the checkpoint.')
# parser.add_argument('--device_id', default=0, type=int, help='Device id to be used.')
args = parser.parse_args()
from mindsponge.common import load_config

if __name__ == "__main__":
    model_name = "model_1"
    config = load_config("./config/base.yaml")
    # model_config = config.model_config(model_name)
    # num_recycle = model_config.model.num_recycle
    # global_config = global_config.global_config(256)
    # extra_msa_length = global_config.extra_msa_length
    print(config.model.embeddings_and_evoformer.evoformer_num_block)
    print(config.model.embeddings_and_evoformer.extra_msa_stack_num_block)
    # print(global_config)