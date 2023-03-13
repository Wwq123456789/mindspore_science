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
"""
The GROVER models for pretraining, finetuning and fingerprint generating.
"""
import mindspore as ms
from mindspore import nn, ops
from .src.data.molgraph import get_atom_fdim, get_bond_fdim
from .src.model.layers import Readout, GTransEncoder
from .src.util.nn_utils import get_activation_function


class GROVEREmbedding(nn.Cell):
    """
    The GROVER Embedding class. It contains the GTransEncoder.
    This GTransEncoder can be replaced by any validate encoders.
    """

    def __init__(self, args):
        """
        Initialize the GROVEREmbedding class.
        :param args:
        """
        super(GROVEREmbedding, self).__init__()
        self.embedding_output_type = args.embedding_output_type
        edge_dim = get_bond_fdim() + get_atom_fdim()
        node_dim = get_atom_fdim()
        if not hasattr(args, "backbone"):
            print("No backbone specified in args, use gtrans backbone.")
            args.backbone = "gtrans"
        if args.backbone == "gtrans" or args.backbone == "dualtrans":
            # dualtrans is the old name.
            self.encoders = GTransEncoder(args=args,
                                          hidden_size=args.hidden_size,
                                          edge_fdim=edge_dim,
                                          node_fdim=node_dim,
                                          dropout=args.dropout,
                                          activation=args.activation,
                                          num_mt_block=args.num_mt_block,
                                          num_attn_head=args.num_attn_head,
                                          atom_emb_output=self.embedding_output_type,
                                          bias=args.bias)

    def construct(self, graph_batch):
        """
        The forward function takes graph_batch as input and output a dict. The content of the dict is decided by
        self.embedding_output_type.

        :param graph_batch: the input graph batch generated by MolCollator.
        :return: a dict containing the embedding results.
        """
        output = self.encoders(graph_batch)
        # index: "atom_from_atom", "atom_from_bond", "bond_from_atom", "bond_from_bond" 0, 1, 2, 3
        preds = None
        if self.embedding_output_type == 'atom':
            preds = (output[0][0], output[0][1], None, None)
        elif self.embedding_output_type == 'bond':
            preds = (None, None, output[1][0], output[1][1])
        elif self.embedding_output_type == "both":
            preds = (output[0][0], output[0][1], output[1][0], output[1][1])

        return preds


class AtomVocabPrediction(nn.Cell):
    """
    The atom-wise vocabulary prediction task. The atom vocabulary is constructed by the context.
    """

    def __init__(self, args, vocab_size, hidden_size=None):
        """
        :param args: the argument.
        :param vocab_size: the size of atom vocabulary.
        """
        super(AtomVocabPrediction, self).__init__()
        if not hidden_size:
            hidden_size = args.hidden_size
        self.embedding_output_type = args.embedding_output_type
        self.linear = nn.Dense(hidden_size, vocab_size)
        self.logsoftmax = nn.LogSoftmax(axis=1)

    def construct(self, embeddings):
        """
        If embeddings is None: do not go through forward pass.
        :param embeddings: the atom embeddings, num_atom X fea_dim.
        :return: the prediction for each atom, num_atom X vocab_size.
        """
        if self.embedding_output_type != "atom" and self.embedding_output_type != "both":
            return None
        linear_out = self.linear(embeddings)
        embeddings = self.logsoftmax(linear_out)

        return embeddings


class BondVocabPrediction(nn.Cell):
    """
    The bond-wise vocabulary prediction task. The bond vocabulary is constructed by the context.
    """

    def __init__(self, args, vocab_size, hidden_size=None):
        """
        Might need to use different architecture for bond vocab prediction.
        :param args:
        :param vocab_size: size of bond vocab.
        :param hidden_size: hidden size
        """
        super(BondVocabPrediction, self).__init__()
        if not hidden_size:
            hidden_size = args.hidden_size
        self.embedding_output_type = args.embedding_output_type
        self.linear = nn.Dense(hidden_size, vocab_size)

        # ad-hoc here
        # If TWO_FC_4_BOND_VOCAB, we will use two distinct fc layer to deal with the bond and rev bond.
        self.two_fc_for_bond_vocab = True
        if self.two_fc_for_bond_vocab:
            self.linear_rev = nn.Dense(hidden_size, vocab_size)
        self.logsoftmax = nn.LogSoftmax(axis=1)

    def construct(self, embeddings):
        """
        If embeddings is None: do not go through forward pass.
        :param embeddings: the atom embeddings, num_bond X fea_dim.
        :return: the prediction for each atom, num_bond X vocab_size.
        """
        if self.embedding_output_type != "bond" and self.embedding_output_type != "both":
            return None
        # The bond and rev bond have odd and even ids respectively. See definition in molgraph.
        embed1 = embeddings[::2]
        embed2 = ops.concat((embeddings[:1], embeddings[1::2]), 0)

        if self.two_fc_for_bond_vocab:
            logits = self.linear(embed1) + self.linear_rev(embed2)
        else:
            logits = self.linear(embed1 + embed2)

        logits = self.logsoftmax(logits)

        return logits


class FunctionalGroupPrediction(nn.Cell):
    """
    The functional group (semantic motifs) prediction task. This is a graph-level task.
    """

    def __init__(self, args, fg_size):
        """
        :param args: The arguments.
        :param fg_size: The size of semantic motifs.
        """
        super(FunctionalGroupPrediction, self).__init__()
        self.first_linear_dim = args.hidden_size
        self.hidden_size = args.hidden_size
        self.embedding_output_type = args.embedding_output_type
        # In order to retain maximal information in the encoder, we use a simple readout function here.
        self.readout = Readout(rtype="mean", hidden_size=self.hidden_size)

        self.linear_atom_from_atom = nn.Dense(self.first_linear_dim, fg_size)
        self.linear_atom_from_bond = nn.Dense(self.first_linear_dim, fg_size)
        self.linear_bond_from_atom = nn.Dense(self.first_linear_dim, fg_size)
        self.linear_bond_from_bond = nn.Dense(self.first_linear_dim, fg_size)

    def construct(self, embeddings, ascope, bscope):
        """
        The forward function of semantic motif prediction. It takes the node/bond embeddings, and the corresponding
        atom/bond scope as input and produce the prediction logits for different branches.
        :param embeddings: The input embeddings are organized as dict. The output of GROVEREmbedding.
        :param ascope: The scope for bonds. Please refer BatchMolGraph for more details.
        :param bscope: The scope for aotms. Please refer BatchMolGraph for more details.
        :return: a dict contains the predicted logits.
        """
        # index: "atom_from_atom", "atom_from_bond", "bond_from_atom", "bond_from_bond" 0, 1, 2, 3

        preds_atom_from_atom, preds_atom_from_bond, preds_bond_from_atom, preds_bond_from_bond = \
            None, None, None, None

        if self.embedding_output_type == "atom" or self.embedding_output_type == "both":
            preds_atom_from_atom = self.readout(embeddings[0], ascope)
            preds_atom_from_atom = self.linear_atom_from_atom(preds_atom_from_atom)
            preds_atom_from_bond = self.readout(embeddings[1], ascope)
            preds_atom_from_bond = self.linear_atom_from_bond(preds_atom_from_bond)

        if self.embedding_output_type == "bond" or self.embedding_output_type == "both":
            preds_bond_from_atom = self.readout(embeddings[2], bscope)
            preds_bond_from_atom = self.linear_bond_from_atom(preds_bond_from_atom)
            preds_bond_from_bond = self.readout(embeddings[3], bscope)
            preds_bond_from_bond = self.linear_bond_from_bond(preds_bond_from_bond)

        preds = (preds_atom_from_atom, preds_atom_from_bond, preds_bond_from_atom, preds_bond_from_bond)

        return preds


class GroverPretrainLossBlock(nn.Cell):
    """Pretrain loss block."""

    def __init__(self, args):
        super(GroverPretrainLossBlock, self).__init__()
        self.embedding_output_type = args.embedding_output_type
        self.av_task_loss = ops.NLLLoss(reduction="mean")
        self.fg_task_loss = nn.BCEWithLogitsLoss(reduction="mean")
        self.av_task_dist_loss = nn.MSELoss(reduction="mean")
        self.fg_task_dist_loss = nn.MSELoss(reduction="mean")
        self.sigmoid = nn.Sigmoid()
        self.dist_coff = args.dist_coff

    def construct(self, preds, targets):
        """
        Compute the pretraining loss.
        """
        av_atom_loss, av_bond_loss, av_dist_loss = 0.0, 0.0, 0.0
        fg_atom_from_atom_loss, fg_atom_from_bond_loss, fg_atom_dist_loss = 0.0, 0.0, 0.0
        bv_atom_loss, bv_bond_loss, bv_dist_loss = 0.0, 0.0, 0.0
        fg_bond_from_atom_loss, fg_bond_from_bond_loss, fg_bond_dist_loss = 0.0, 0.0, 0.0

        # index: "atom_from_atom", "atom_from_bond", "bond_from_atom", "bond_from_bond" 0, 1, 2, 3
        # index: "av_task", "bv_task", "fg_task" 0, 1, 2

        if self.embedding_output_type == "atom" or self.embedding_output_type == "both":
            nllloss_weight = ops.ones(preds[0][0].shape[-1], ms.float32)
            av_atom_loss, _ = self.av_task_loss(preds[0][0], targets[0], nllloss_weight)
            fg_atom_from_atom_loss = self.fg_task_loss(preds[2][0], targets[2])

            nllloss_weight = ops.ones(preds[0][1].shape[-1], ms.float32)
            av_bond_loss, _ = self.av_task_loss(preds[0][1], targets[0], nllloss_weight)
            fg_atom_from_bond_loss = self.fg_task_loss(preds[2][1], targets[2])

            av_dist_loss = self.av_task_dist_loss(preds[0][0], preds[0][1])
            fg_atom_dist_loss = self.fg_task_dist_loss(self.sigmoid(preds[2][0]),
                                                       self.sigmoid(preds[2][1]))

        if self.embedding_output_type == "bond" or self.embedding_output_type == "both":
            nllloss_weight = ops.ones(preds[1][0].shape[-1], ms.float32)
            bv_atom_loss, _ = self.av_task_loss(preds[1][0], targets[1], nllloss_weight)
            fg_bond_from_atom_loss = self.fg_task_loss(preds[2][2], targets[2])

            nllloss_weight = ops.ones(preds[1][1].shape[-1], ms.float32)
            bv_bond_loss, _ = self.av_task_loss(preds[1][1], targets[1], nllloss_weight)
            fg_bond_from_bond_loss = self.fg_task_loss(preds[2][3], targets[2])

            bv_dist_loss = self.av_task_dist_loss(preds[1][0], preds[1][1])
            fg_bond_dist_loss = self.fg_task_dist_loss(self.sigmoid(preds[2][2]),
                                                       self.sigmoid(preds[2][3]))

        av_loss = av_atom_loss + av_bond_loss
        bv_loss = bv_atom_loss + bv_bond_loss
        fg_atom_loss = fg_atom_from_atom_loss + fg_atom_from_bond_loss
        fg_bond_loss = fg_bond_from_atom_loss + fg_bond_from_bond_loss

        fg_loss = fg_atom_loss + fg_bond_loss
        fg_dist_loss = fg_atom_dist_loss + fg_bond_dist_loss

        overall_loss = av_loss + bv_loss + fg_loss + self.dist_coff * av_dist_loss + \
                       self.dist_coff * bv_dist_loss + fg_dist_loss

        return overall_loss


class GroverPretrainTask(nn.Cell):
    """
    The pretrain task.
    """

    def __init__(self, args, grover, atom_vocab_size, bond_vocab_size, fg_size):
        super(GroverPretrainTask, self).__init__()
        self.grover = grover
        self.av_task_atom = AtomVocabPrediction(args, atom_vocab_size)
        self.av_task_bond = AtomVocabPrediction(args, atom_vocab_size)
        self.bv_task_atom = BondVocabPrediction(args, bond_vocab_size)
        self.bv_task_bond = BondVocabPrediction(args, bond_vocab_size)
        self.fg_task_all = FunctionalGroupPrediction(args, fg_size)
        self.embedding_output_type = args.embedding_output_type
        self.loss_block = GroverPretrainLossBlock(args)

    def construct(self, graph_batch, scope, targets):
        """
        Deal pretrain task.
        """
        a_scope = scope[0]
        b_scope = scope[1]

        embeddings = self.grover(graph_batch)

        # index: "atom_from_atom", "atom_from_bond", "bond_from_atom", "bond_from_bond" 0, 1, 2, 3

        av_task_pred_atom = self.av_task_atom(embeddings[0])  # if None: means not go through this forward
        av_task_pred_bond = self.av_task_bond(embeddings[1])

        bv_task_pred_atom = self.bv_task_atom(embeddings[2])
        bv_task_pred_bond = self.bv_task_bond(embeddings[3])

        fg_task_pred_all = self.fg_task_all(embeddings, a_scope, b_scope)

        preds = ((av_task_pred_atom, av_task_pred_bond), (bv_task_pred_atom, bv_task_pred_bond), fg_task_pred_all)

        loss = self.loss_block(preds, targets)

        return loss


class GroverFinetuneLossBlock(nn.Cell):
    """
    Finetune loss block.
    """

    def __init__(self, args):
        super(GroverFinetuneLossBlock, self).__init__()
        self.type = args.dataset_type
        self.dist_coff = args.dist_coff

        if self.type == 'classification':
            self.pred_loss = nn.BCEWithLogitsLoss(reduction='mean')
        elif self.type == 'regression':
            self.pred_loss = nn.MSELoss(reduction='mean')
        else:
            raise ValueError(f'Dataset type "{args.dataset_type}" not supported.')

        self.dist_loss = nn.MSELoss(reduction='mean')

        self.mode = None

    def construct(self, preds, targets):
        """
        Compute the finetuning loss.
        """
        dist = self.dist_loss(preds[0], preds[1])
        pred_loss1 = self.pred_loss(preds[0], targets)
        pred_loss2 = self.pred_loss(preds[1], targets)
        loss = pred_loss1 + pred_loss2 + self.dist_coff * dist

        return loss


class GroverFinetuneEvalBlock(nn.Cell):
    """
    The finetune eval task.
    """

    def __init__(self, args):
        super(GroverFinetuneEvalBlock, self).__init__()
        self.classification = args.dataset_type == 'classification'
        if self.classification:
            self.sigmoid = nn.Sigmoid()

    def construct(self, preds):
        """
        Compute the predicts.
        """
        atom_ffn_output = preds[0]
        bond_ffn_output = preds[1]
        if self.classification:
            atom_ffn_output = self.sigmoid(atom_ffn_output)
            bond_ffn_output = self.sigmoid(bond_ffn_output)
        preds = (atom_ffn_output + bond_ffn_output) / 2
        return preds


class GroverFinetuneTask(nn.Cell):
    """
    The finetune task.
    """

    def __init__(self, args, grover, is_training):
        super(GroverFinetuneTask, self).__init__()

        self.hidden_size = args.hidden_size
        self.grover = grover

        if args.self_attention:
            self.readout = Readout(rtype="self_attention", hidden_size=self.hidden_size,
                                   attn_hidden=args.attn_hidden,
                                   attn_out=args.attn_out)
        else:
            self.readout = Readout(rtype="mean", hidden_size=self.hidden_size)

        self.mol_atom_from_atom_ffn = self.create_ffn(args)
        self.mol_atom_from_bond_ffn = self.create_ffn(args)

        self.classification = args.dataset_type == 'classification'
        if self.classification:
            self.sigmoid = nn.Sigmoid()

        if is_training:
            self.loss_block = GroverFinetuneLossBlock(args)
        else:
            self.eval_block = GroverFinetuneEvalBlock(args)

    def create_ffn(self, args):
        """
        Creates the feed-forward network for the model.
        """
        # Note: args.features_dim is set according the real loaded features data
        if args.features_only:
            first_linear_dim = args.features_size + args.features_dim
        else:
            if args.self_attention:
                first_linear_dim = args.hidden_size * args.attn_out
                # if args.use_input_features:
                first_linear_dim += args.features_dim
            else:
                first_linear_dim = args.hidden_size + args.features_dim

        dropout = nn.Dropout(keep_prob=1 - args.dropout)
        activation = get_activation_function(args.activation)
        # Create FFN layers
        if args.ffn_num_layers == 1:
            ffn = [
                dropout,
                nn.Dense(first_linear_dim, args.output_size)
            ]
        else:
            ffn = [
                dropout,
                nn.Dense(first_linear_dim, args.ffn_hidden_size)
            ]
            for _ in range(args.ffn_num_layers - 2):
                ffn.extend([
                    activation,
                    dropout,
                    nn.Dense(args.ffn_hidden_size, args.ffn_hidden_size),
                ])
            ffn.extend([
                activation,
                dropout,
                nn.Dense(args.ffn_hidden_size, args.output_size),
            ])

        # Create FFN model
        return nn.SequentialCell(ffn)

    def construct(self, graph_batch, scope, features_batch, targets=None):
        """
        Deal the finetune task.
        """
        a_scope = scope[0]

        output = self.grover(graph_batch)
        # Share readout
        mol_atom_from_atom_output = self.readout(output[0], a_scope)
        mol_atom_from_bond_output = self.readout(output[1], a_scope)

        mol_atom_from_atom_output = ops.concat([mol_atom_from_atom_output, features_batch], 1)
        mol_atom_from_bond_output = ops.concat([mol_atom_from_bond_output, features_batch], 1)

        atom_ffn_output = self.mol_atom_from_atom_ffn(mol_atom_from_atom_output)
        bond_ffn_output = self.mol_atom_from_bond_ffn(mol_atom_from_bond_output)

        preds = (atom_ffn_output, bond_ffn_output)

        if self.training:
            result = self.loss_block(preds, targets)
        else:
            result = self.eval_block(preds)

        return result


class GroverFpGenerationTask(nn.Cell):
    """
    GroverFpGeneration class.
    It loads the pre-trained model and produce the fingerprints for input molecules.
    """

    def __init__(self, args, grover):
        super(GroverFpGenerationTask, self).__init__()

        self.fingerprint_source = args.fingerprint_source
        self.grover = grover
        self.readout = Readout(rtype="mean", hidden_size=args.hidden_size)

    def construct(self, graph_batch, scope, features_batch):
        """
        It takes graph batch and molecular feature batch as input and produce the fingerprints of this molecules.
        """
        a_scope = scope[0]
        b_scope = scope[1]

        output = self.grover(graph_batch)

        mol_atom_from_bond_output = self.readout(output[0], a_scope)
        mol_atom_from_atom_output = self.readout(output[1], a_scope)

        mol_bond_from_atom_output = None
        mol_bond_from_bond_output = None

        if self.fingerprint_source == "bond" or self.fingerprint_source == "both":
            mol_bond_from_atom_output = self.readout(output[2], b_scope)
            mol_bond_from_bond_output = self.readout(output[3], b_scope)

        if self.fingerprint_source == "atom":
            fp = ops.concat([mol_atom_from_atom_output, mol_atom_from_bond_output], 1)
        elif self.fingerprint_source == "bond":
            fp = ops.concat([mol_bond_from_atom_output, mol_bond_from_bond_output], 1)
        else:
            # the both case.
            fp = ops.concat([mol_atom_from_atom_output, mol_atom_from_bond_output,
                             mol_bond_from_atom_output, mol_bond_from_bond_output], 1)

        fp = ops.concat([fp, features_batch], 1)

        return fp


class GroverExportTask(nn.Cell):
    """
    The class for 310 infer.
    """
    def __init__(self, task):
        super(GroverExportTask, self).__init__()
        self.task = task

    def construct(self, f_atoms, f_bonds, a2b, b2a, b2revb, a2a, a_scope, b_scope, features_batch):
        """
        310 eval.
        """
        graph_inputs = (f_atoms, f_bonds, a2b, b2a, b2revb, a2a)
        scope = (a_scope, b_scope)
        targets = None
        preds = self.task(graph_inputs, scope, features_batch, targets)
        return preds
