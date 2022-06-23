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
"""data transform MSA TEMPLATE"""
import numpy as np
from ..common.residue_constants import MAP_HHBLITS_AATYPE_TO_OUR_AATYPE, restype_order, atom_order, restypes, \
    restype_name_to_atom14_names, restype_1to3, atom_types, residue_atoms

MS_MIN32 = -2147483648
MS_MAX32 = 2147483647


def one_hot(depth, indices):
    """one hot compute"""
    res = np.eye(depth)[indices.reshape(-1)]
    return res.reshape(list(indices.shape) + [depth])


def correct_msa_restypes(msa):
    """Correct MSA restype to have the same order as residue_constants."""
    new_order_list = MAP_HHBLITS_AATYPE_TO_OUR_AATYPE
    new_order = np.array(new_order_list, dtype=msa.dtype)
    msa = new_order[msa]
    return msa


def randomly_replace_msa_with_unknown(msa, aatype, replace_proportion):
    """Replace a proportion of the MSA with 'X'."""
    msa_mask = np.random.uniform(size=msa.shape, low=0, high=1) < replace_proportion
    x_idx = 20
    gap_idx = 21
    msa_mask = np.logical_and(msa_mask, msa != gap_idx)
    msa = np.where(msa_mask, np.ones_like(msa) * x_idx, msa)
    aatype_mask = np.random.uniform(size=aatype.shape, low=0, high=1) < replace_proportion
    aatype = np.where(aatype_mask, np.ones_like(aatype) * x_idx, aatype)
    return msa, aatype


def make_random_crop_to_size_seed(protein):
    """Random seed for cropping residues and templates."""
    protein['random_crop_to_size_seed'] = np.array(make_random_seed([2], seed_maker_t=seed_maker()), np.int32)
    return protein


def fix_templates_aatype(template_aatype):
    """Fixes aatype encoding of templates."""
    # Map one-hot to indices.
    template_aatype = np.argmax(template_aatype, axis=-1).astype(np.int32)
    # Map hhsearch-aatype to our aatype.
    new_order_list = MAP_HHBLITS_AATYPE_TO_OUR_AATYPE
    new_order = np.array(new_order_list, np.int32)
    template_aatype = new_order[template_aatype]
    return template_aatype


def pseudo_beta_fn(aatype, all_atom_positions, all_atom_masks):
    """Create pseudo beta features."""
    is_gly = np.equal(aatype, restype_order['G'])
    ca_idx = atom_order['CA']
    cb_idx = atom_order['CB']
    pseudo_beta = np.where(
        np.tile(is_gly[..., None].astype("int32"), [1,] * len(is_gly.shape) + [3,]).astype("bool"),
        all_atom_positions[..., ca_idx, :],
        all_atom_positions[..., cb_idx, :])
    if all_atom_masks is not None:
        pseudo_beta_mask = np.where(is_gly, all_atom_masks[..., ca_idx], all_atom_masks[..., cb_idx])
        pseudo_beta_mask = pseudo_beta_mask.astype(np.float32)
        return pseudo_beta, pseudo_beta_mask
    return pseudo_beta


def make_atom14_masks(aatype):
    """Construct denser atom positions (14 dimensions instead of 37)."""
    restype_atom14_to_atom37 = []  # mapping (restype, atom14) --> atom37
    restype_atom37_to_atom14 = []  # mapping (restype, atom37) --> atom14
    restype_atom14_mask = []

    for rt in restypes:
        atom_names = restype_name_to_atom14_names.get(restype_1to3.get(rt))

        restype_atom14_to_atom37.append([(atom_order[name] if name else 0) for name in atom_names])

        atom_name_to_idx14 = {name: i for i, name in enumerate(atom_names)}
        restype_atom37_to_atom14.append([(atom_name_to_idx14[name] if name in atom_name_to_idx14 else 0)
                                         for name in atom_types])

        restype_atom14_mask.append([(1. if name else 0.) for name in atom_names])

    # Add dummy mapping for restype 'UNK'
    restype_atom14_to_atom37.append([0] * 14)
    restype_atom37_to_atom14.append([0] * 37)
    restype_atom14_mask.append([0.] * 14)

    restype_atom14_to_atom37 = np.array(restype_atom14_to_atom37, np.int32)
    restype_atom37_to_atom14 = np.array(restype_atom37_to_atom14, np.int32)
    restype_atom14_mask = np.array(restype_atom14_mask, np.float32)

    # create the mapping for (residx, atom14) --> atom37, i.e. an array
    # with shape (num_res, 14) containing the atom37 indices for this protein
    residx_atom14_to_atom37 = restype_atom14_to_atom37[aatype]
    residx_atom14_mask = restype_atom14_mask[aatype]

    atom14_atom_exists = residx_atom14_mask
    residx_atom14_to_atom37 = residx_atom14_to_atom37

    # create the gather indices for mapping back
    residx_atom37_to_atom14 = restype_atom37_to_atom14[aatype]
    residx_atom37_to_atom14 = residx_atom37_to_atom14

    # create the corresponding mask
    restype_atom37_mask = np.zeros([21, 37], np.float32)
    for restype, restype_letter in enumerate(restypes):
        restype_name = restype_1to3.get(restype_letter)
        atom_names = residue_atoms.get(restype_name)
        for atom_name in atom_names:
            atom_type = atom_order[atom_name]
            restype_atom37_mask[restype, atom_type] = 1

    atom37_atom_exists = restype_atom37_mask[aatype]
    res = [atom14_atom_exists, residx_atom14_to_atom37, residx_atom37_to_atom14, atom37_atom_exists]
    return res


def block_delete_msa_indices(msa, msa_fraction_per_block, randomize_num_blocks, num_blocks):
    """Sample MSA by deleting contiguous blocks.

    Jumper et al. (2021) Suppl. Alg. 1 "MSABlockDeletion"

    Arguments:
    protein: batch dict containing the msa
    config: ConfigDict with parameters

    Returns:
    updated protein
    """

    num_seq = msa.shape[0]
    block_num_seq = np.floor(num_seq * msa_fraction_per_block).astype(np.int32)

    if randomize_num_blocks:
        nb = int(np.random.uniform(0, num_blocks + 1))
    else:
        nb = num_blocks
    del_block_starts = np.random.uniform(0, num_seq, nb).astype(np.int32)
    del_blocks = del_block_starts[:, None] + np.array([_ for _ in range(block_num_seq)]).astype(np.int32)
    del_blocks = np.clip(del_blocks, 0, num_seq - 1)
    del_indices = np.unique(np.sort(np.reshape(del_blocks, (-1,))))

    # Make sure we keep the original sequence
    keep_indices = np.setdiff1d(np.array([_ for _ in range(1, num_seq)]),
                                del_indices)
    keep_indices = np.concatenate([[0], keep_indices], axis=0)
    keep_indices = [int(x) for x in keep_indices]
    return keep_indices


def sample_msa(msa, max_seq):
    """Sample MSA randomly, remaining sequences are stored as `extra_*`."""
    num_seq = msa.shape[0]

    shuffled = list(range(1, num_seq))
    np.random.shuffle(shuffled)
    shuffled.insert(0, 0)
    index_order = np.array(shuffled, np.int32)
    num_sel = min(max_seq, num_seq)

    sel_seq = index_order[:num_sel]
    not_sel_seq = index_order[num_sel:]
    is_sel = num_seq - num_sel
    return is_sel, not_sel_seq, sel_seq


def make_masked_msa(msa, hhblits_profile, uniform_prob, profile_prob, same_prob, replace_fraction):
    """Create data for BERT on raw MSA."""
    # Add a random amino acid uniformly
    random_aa = np.array([0.05] * 20 + [0., 0.], dtype=np.float32)

    categorical_probs = uniform_prob * random_aa + profile_prob * hhblits_profile + \
                        same_prob * one_hot(22, msa)

    # Put all remaining probability on [MASK] which is a new column
    pad_shapes = [[0, 0] for _ in range(len(categorical_probs.shape))]
    pad_shapes[-1][1] = 1
    mask_prob = 1. - config.profile_prob - config.same_prob - config.uniform_prob
    assert mask_prob >= 0.
    categorical_probs = np.pad(categorical_probs, pad_shapes, constant_values=(mask_prob,))

    mask_position = np.random.uniform(size=msa.shape, low=0, high=1) < replace_fraction

    bert_msa = shaped_categorical(categorical_probs)
    bert_msa = np.where(mask_position, bert_msa, msa)

    # Mix real and masked MSA
    bert_mask = mask_position.astype(np.int32)
    true_msa = msa
    msa = bert_msa
    return bert_mask, true_msa, msa


def nearest_neighbor_clusters(msa_mask, msa, extra_msa_mask, extra_msa, gap_agreement_weight=0.):
    """Assign each extra MSA sequence to its nearest neighbor in sampled MSA."""

    # Determine how much weight we assign to each agreement.  In theory, we could
    # use a full blosum matrix here, but right now let's just down-weight gap
    # agreement because it could be spurious.
    # Never put weight on agreeing on BERT mask
    weights = np.concatenate([np.ones(21), gap_agreement_weight * np.ones(1), np.zeros(1)], 0)

    # Make agreement score as weighted Hamming distance
    sample_one_hot = msa_mask[:, :, None] * one_hot(23, msa)
    num_seq, num_res, _ = sample_one_hot.shape

    array_extra_msa_mask = extra_msa_mask
    if array_extra_msa_mask.any():
        extra_one_hot = extra_msa_mask[:, :, None] * one_hot(23, extra_msa)
        extra_num_seq, _, _ = extra_one_hot.shape

        agreement = np.matmul(
            np.reshape(extra_one_hot, [extra_num_seq, num_res * 23]),
            np.reshape(sample_one_hot * weights, [num_seq, num_res * 23]).T)
        # Assign each sequence in the extra sequences to the closest MSA sample
        extra_cluster_assignment = np.argmax(agreement, axis=1)
    else:
        extra_cluster_assignment = np.array([])
    return extra_cluster_assignment


def summarize_clusters(msa, msa_mask, extra_cluster_assignment, extra_msa_mask, extra_msa, extra_deletion_matrix,
                       deletion_matrix):
    """Produce profile and deletion_matrix_mean within each cluster."""
    num_seq = msa.shape[0]

    def csum(x):
        result = []
        for i in range(num_seq):
            result.append(np.sum(x[np.where(extra_cluster_assignment == i)], axis=0))
        return np.array(result)

    mask = extra_msa_mask
    mask_counts = 1e-6 + msa_mask + csum(mask)  # Include center

    msa_sum = csum(mask[:, :, None] * one_hot(23, extra_msa))
    msa_sum += one_hot(23, msa)  # Original sequence
    cluster_profile = msa_sum / mask_counts[:, :, None]

    del msa_sum

    del_sum = csum(mask * extra_deletion_matrix)
    del_sum += deletion_matrix  # Original sequence
    cluster_deletion_mean = del_sum / mask_counts
    del del_sum

    return cluster_profile, cluster_deletion_mean


def crop_extra_msa(extra_msa, max_extra_msa):
    """MSA features are cropped so only `max_extra_msa` sequences are kept."""
    if extra_msa.any():
        num_seq = extra_msa.shape[0]
        num_sel = np.minimum(max_extra_msa, num_seq)
        shuffled = list(range(num_seq))
        np.random.shuffle(shuffled)
        select_indices = shuffled[:num_sel]
        return select_indices
    return None


def make_msa_feat(between_segment_residues, aatype, msa, deletion_matrix, cluster_deletion_mean, cluster_profile,
                  extra_deletion_matrix):
    """Create and concatenate MSA features."""
    # Whether there is a domain break. Always zero for chains, but keeping
    # for compatibility with domain datasets.
    has_break = np.clip(between_segment_residues.astype(np.float32), np.array(0), np.array(1))
    aatype_1hot = one_hot(21, aatype)

    target_feat = [np.expand_dims(has_break, axis=-1), aatype_1hot]

    msa_1hot = one_hot(23, msa)
    has_deletion = np.clip(deletion_matrix, np.array(0), np.array(1))
    deletion_value = np.arctan(deletion_matrix / 3.) * (2. / np.pi)

    msa_feat = [msa_1hot, np.expand_dims(has_deletion, axis=-1), np.expand_dims(deletion_value, axis=-1)]

    if cluster_profile is not None:
        deletion_mean_value = (np.arctan(cluster_deletion_mean / 3.) * (2. / np.pi))
        msa_feat.extend([cluster_profile, np.expand_dims(deletion_mean_value, axis=-1)])
    extra_has_deletion = None
    extra_deletion_value = None
    if extra_deletion_matrix is not None:
        extra_has_deletion = np.clip(extra_deletion_matrix, np.array(0), np.array(1))
        extra_deletion_value = np.arctan(extra_deletion_matrix / 3.) * (2. / np.pi)

    msa_feat = np.concatenate(msa_feat, axis=-1)
    target_feat = np.concatenate(target_feat, axis=-1)
    res = [extra_has_deletion, extra_deletion_value, msa_feat, target_feat]
    return res


def make_random_seed(size, seed_maker_t, low=MS_MIN32, high=MS_MAX32):
    np.random.seed(seed_maker_t)
    return np.random.uniform(size=size, low=low, high=high)


def random_crop_to_size(seq_length, template_mask, crop_size, max_templates,
                        subsample_templates=False, seed=0):
    """Crop randomly to `crop_size`, or keep as is if shorter than that."""
    seq_length = seq_length
    seq_length_int = int(seq_length)
    if template_mask is not None:
        num_templates = np.array(template_mask.shape[0], np.int32)
    else:
        num_templates = np.array(0, np.int32)
    num_res_crop_size = np.minimum(seq_length, crop_size)
    num_res_crop_size_int = int(num_res_crop_size)

    # Ensures that the cropping of residues and templates happens in the same way
    # across ensembling iterations.
    # Do not use for randomness that should vary in ensembling.

    if subsample_templates:
        templates_crop_start = int(make_random_seed(size=(), seed_maker_t=seed, low=0, high=num_templates + 1))
    else:
        templates_crop_start = 0

    num_templates_crop_size = np.minimum(num_templates - templates_crop_start, max_templates)
    num_templates_crop_size_int = int(num_templates_crop_size)

    num_res_crop_start = int(make_random_seed(size=(), seed_maker_t=seed, low=0,
                                              high=seq_length_int - num_res_crop_size_int + 1))

    templates_select_indices = np.argsort(make_random_seed(size=[num_templates], seed_maker_t=seed))
    res = [num_res_crop_size, num_templates_crop_size_int, num_res_crop_start, num_res_crop_size_int, \
           templates_crop_start, templates_select_indices]
    return res
