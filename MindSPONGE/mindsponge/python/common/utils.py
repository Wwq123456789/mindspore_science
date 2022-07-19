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
"""utils module"""

import numpy as np
from Bio import Align
from Bio.Align import substitution_matrices
from mindspore import nn
import mindspore.numpy as mnp
from mindspore.ops import operations as P
from scipy.special import softmax

from . import r3
from . import residue_constants, protein


def pseudo_beta_fn(aatype, all_atom_positions, all_atom_masks):
    """Create pseudo beta features."""

    is_gly = mnp.equal(aatype, residue_constants.restype_order['G'])
    ca_idx = residue_constants.atom_order['CA']
    cb_idx = residue_constants.atom_order['CB']
    pseudo_beta = mnp.where(
        mnp.tile(is_gly[..., None], [1,] * len(is_gly.shape) + [3,]),
        all_atom_positions[..., ca_idx, :],
        all_atom_positions[..., cb_idx, :])
    if all_atom_masks is not None:
        pseudo_beta_mask = mnp.where(is_gly, all_atom_masks[..., ca_idx], all_atom_masks[..., cb_idx])
        pseudo_beta_mask = pseudo_beta_mask.astype(mnp.float32)
        return pseudo_beta, pseudo_beta_mask
    return pseudo_beta


def dgram_from_positions(positions, num_bins, min_bin, max_bin):
    """Compute distogram from amino acid positions.

    Arguments:
    positions: [N_res, 3] Position coordinates.
    num_bins: The number of bins in the distogram.
    min_bin: The left edge of the first bin.
    max_bin: The left edge of the final bin. The final bin catches
    everything larger than `max_bin`.

    Returns:
    Distogram with the specified number of bins.
    """

    def squared_difference(x, y):
        return mnp.square(x - y)

    lower_breaks = mnp.linspace(min_bin, max_bin, num_bins)
    lower_breaks = mnp.square(lower_breaks)
    upper_breaks = mnp.concatenate([lower_breaks[1:], mnp.array([1e8], dtype=mnp.float32)], axis=-1)
    dist2 = mnp.sum(squared_difference(mnp.expand_dims(positions, axis=-2),
                                       mnp.expand_dims(positions, axis=-3)), axis=-1, keepdims=True)
    dgram = ((dist2 > lower_breaks).astype(mnp.float32) * (dist2 < upper_breaks).astype(mnp.float32))
    return dgram


def mask_mean(mask, value, axis=0, drop_mask_channel=False, eps=1e-10):
    """Masked mean."""
    if drop_mask_channel:
        mask = mask[..., 0]
    mask_shape = mask.shape
    value_shape = value.shape
    broadcast_factor = 1.
    value_size = value_shape[axis]
    mask_size = mask_shape[axis]
    if mask_size == 1:
        broadcast_factor *= value_size
    return mnp.sum(mask * value, axis=axis) / (mnp.sum(mask, axis=axis) * broadcast_factor + eps)


def atom37_to_torsion_angles(
        aatype,  # (B, N)
        all_atom_pos,  # (B, N, 37, 3)
        all_atom_mask,  # (B, N, 37)
        chi_atom_indices,
        chi_angles_mask,
        mirror_psi_mask,
        chi_pi_periodic,
        indices0,
        indices1
):
    """Computes the 7 torsion angles (in sin, cos encoding) for each residue.

    The 7 torsion angles are in the order
    '[pre_omega, phi, psi, chi_1, chi_2, chi_3, chi_4]',
    here pre_omega denotes the omega torsion angle between the given amino acid
    and the previous amino acid.

    Args:
    aatype: Amino acid type, given as array with integers.
    all_atom_pos: atom37 representation of all atom coordinates.
    all_atom_mask: atom37 representation of mask on all atom coordinates.
    placeholder_for_undefined: flag denoting whether to set masked torsion
    angles to zero.
    Returns:
    Dict containing:
    * 'torsion_angles_sin_cos': Array with shape (B, N, 7, 2) where the final
    2 dimensions denote sin and cos respectively
    * 'alt_torsion_angles_sin_cos': same as 'torsion_angles_sin_cos', but
    with the angle shifted by pi for all chi angles affected by the naming
    ambiguities.
    * 'torsion_angles_mask': Mask for which chi angles are present.
    """

    # Map aatype > 20 to 'Unknown' (20).
    aatype = mnp.minimum(aatype, 20)

    # Compute the backbone angles.
    num_batch, num_res = aatype.shape

    pad = mnp.zeros([num_batch, 1, 37, 3], mnp.float32)
    prev_all_atom_pos = mnp.concatenate([pad, all_atom_pos[:, :-1, :, :]], axis=1)

    pad = mnp.zeros([num_batch, 1, 37], mnp.float32)
    prev_all_atom_mask = mnp.concatenate([pad, all_atom_mask[:, :-1, :]], axis=1)

    # For each torsion angle collect the 4 atom positions that define this angle.
    # shape (B, N, atoms=4, xyz=3)
    pre_omega_atom_pos = mnp.concatenate([prev_all_atom_pos[:, :, 1:3, :], all_atom_pos[:, :, 0:2, :]], axis=-2)
    phi_atom_pos = mnp.concatenate([prev_all_atom_pos[:, :, 2:3, :], all_atom_pos[:, :, 0:3, :]], axis=-2)
    psi_atom_pos = mnp.concatenate([all_atom_pos[:, :, 0:3, :], all_atom_pos[:, :, 4:5, :]], axis=-2)
    # # Collect the masks from these atoms.
    # # Shape [batch, num_res]
    # ERROR NO PROD
    pre_omega_mask = (P.ReduceProd()(prev_all_atom_mask[:, :, 1:3], -1)  # prev CA, C
                      * P.ReduceProd()(all_atom_mask[:, :, 0:2], -1))  # this N, CA
    phi_mask = (prev_all_atom_mask[:, :, 2]  # prev C
                * P.ReduceProd()(all_atom_mask[:, :, 0:3], -1))  # this N, CA, C
    psi_mask = (P.ReduceProd()(all_atom_mask[:, :, 0:3], -1) *  # this N, CA, C
                all_atom_mask[:, :, 4])  # this O
    # Collect the atoms for the chi-angles.
    # Compute the table of chi angle indices. Shape: [restypes, chis=4, atoms=4].
    # Select atoms to compute chis. Shape: [batch, num_res, chis=4, atoms=4].
    atom_indices = mnp.take(chi_atom_indices, aatype, axis=0)

    # # Gather atom positions Batch Gather. Shape: [batch, num_res, chis=4, atoms=4, xyz=3].

    # 4 seq_length 4 4  batch, sequence length, chis, atoms
    seq_length = all_atom_pos.shape[1]
    atom_indices = atom_indices.reshape((4, seq_length, 4, 4, 1)).astype("int32")
    new_indices = P.Concat(4)((indices0, indices1, atom_indices))  # 4, seq_length, 4, 4, 3
    chis_atom_pos = P.GatherNd()(all_atom_pos, new_indices)
    chis_mask = mnp.take(chi_angles_mask, aatype, axis=0)
    chi_angle_atoms_mask = P.GatherNd()(all_atom_mask, new_indices)

    # Check if all 4 chi angle atoms were set. Shape: [batch, num_res, chis=4].
    chi_angle_atoms_mask = P.ReduceProd()(chi_angle_atoms_mask, -1)
    chis_mask = chis_mask * (chi_angle_atoms_mask).astype(mnp.float32)

    # Stack all torsion angle atom positions.
    # Shape (B, N, torsions=7, atoms=4, xyz=3)ls
    torsions_atom_pos = mnp.concatenate([pre_omega_atom_pos[:, :, None, :, :],
                                         phi_atom_pos[:, :, None, :, :],
                                         psi_atom_pos[:, :, None, :, :],
                                         chis_atom_pos], axis=2)
    # Stack up masks for all torsion angles.
    # shape (B, N, torsions=7)
    torsion_angles_mask = mnp.concatenate([pre_omega_mask[:, :, None],
                                           phi_mask[:, :, None],
                                           psi_mask[:, :, None],
                                           chis_mask], axis=2)

    torsion_frames_rots, torsion_frames_trans = r3.rigids_from_3_points(
        torsions_atom_pos[:, :, :, 1, :],
        torsions_atom_pos[:, :, :, 2, :],
        torsions_atom_pos[:, :, :, 0, :])
    inv_torsion_rots, inv_torsion_trans = r3.invert_rigids(torsion_frames_rots, torsion_frames_trans)
    forth_atom_rel_pos = r3.rigids_mul_vecs(inv_torsion_rots, inv_torsion_trans, torsions_atom_pos[:, :, :, 3, :])

    # Compute the position of the forth atom in this frame (y and z coordinate
    torsion_angles_sin_cos = mnp.stack([forth_atom_rel_pos[..., 2], forth_atom_rel_pos[..., 1]], axis=-1)
    torsion_angles_sin_cos /= mnp.sqrt(mnp.sum(mnp.square(torsion_angles_sin_cos), axis=-1, keepdims=True) + 1e-8)
    # Mirror psi, because we computed it from the Oxygen-atom.
    torsion_angles_sin_cos *= mirror_psi_mask
    chi_is_ambiguous = mnp.take(chi_pi_periodic, aatype, axis=0)
    mirror_torsion_angles = mnp.concatenate([mnp.ones([num_batch, num_res, 3]), 1.0 - 2.0 * chi_is_ambiguous], axis=-1)
    alt_torsion_angles_sin_cos = (torsion_angles_sin_cos * mirror_torsion_angles[:, :, :, None])
    return torsion_angles_sin_cos, alt_torsion_angles_sin_cos, torsion_angles_mask


def get_chi_atom_indices():
    """Returns atom indices needed to compute chi angles for all residue types.

  Returns:
    A tensor of shape [residue_types=21, chis=4, atoms=4]. The residue types are
    in the order specified in residue_constants.restypes + unknown residue type
    at the end. For chi angles which are not defined on the residue, the
    positions indices are by default set to 0.
  """
    chi_atom_indices = []
    for residue_name in residue_constants.restypes:
        residue_name = residue_constants.restype_1to3[residue_name]
        residue_chi_angles = residue_constants.chi_angles_atoms[residue_name]
        atom_indices = []
        for chi_angle in residue_chi_angles:
            atom_indices.append(
                [residue_constants.atom_order[atom] for atom in chi_angle])
        for _ in range(4 - len(atom_indices)):
            atom_indices.append([0, 0, 0, 0])  # For chi angles not defined on the AA.
        chi_atom_indices.append(atom_indices)

    chi_atom_indices.append([[0, 0, 0, 0]] * 4)  # For UNKNOWN residue.

    return np.array(chi_atom_indices).astype(np.int32)


def rigids_from_tensor4x4(m):
    """Construct Rigids object from an 4x4 array.

    Here the 4x4 is representing the transformation in homogeneous coordinates.

    Args:
    m: Array representing transformations in homogeneous coordinates.
    Returns:
    Rigids object corresponding to transformations m
    """
    return m[..., 0, 0], m[..., 0, 1], m[..., 0, 2], m[..., 1, 0], m[..., 1, 1], m[..., 1, 2], m[..., 2, 0], \
           m[..., 2, 1], m[..., 2, 2], m[..., 0, 3], m[..., 1, 3], m[..., 2, 3]


def frames_and_literature_positions_to_atom14_pos(aatype, all_frames_to_global, restype_atom14_to_rigid_group,
                                                  restype_atom14_rigid_group_positions, restype_atom14_mask):  # (N, 14)
    """Put atom literature positions (atom14 encoding) in each rigid group.

    Jumper et al. (2021) Suppl. Alg. 24 "computeAllAtomCoordinates" line 11

    Args:
    aatype: aatype for each residue.
    all_frames_to_global: All per residue coordinate frames.
    Returns:
    Positions of all atom coordinates in global frame.
    """

    # Pick the appropriate transform for every atom.
    residx_to_group_idx = P.Gather()(restype_atom14_to_rigid_group, aatype, 0)
    group_mask = nn.OneHot(depth=8, axis=-1)(residx_to_group_idx)

    # # r3.Rigids with shape (N, 14)
    map_atoms_to_global = map_atoms_to_global_func(all_frames_to_global, group_mask)

    # Gather the literature atom positions for each residue.
    # r3.Vecs with shape (N, 14)
    lit_positions = vecs_from_tensor(P.Gather()(restype_atom14_rigid_group_positions, aatype, 0))

    # Transform each atom from its local frame to the global frame.
    # r3.Vecs with shape (N, 14)
    pred_positions = rigids_mul_vecs(map_atoms_to_global, lit_positions)

    # Mask out non-existing atoms.
    mask = P.Gather()(restype_atom14_mask, aatype, 0)

    pred_positions = pred_map_mul(pred_positions, mask)

    return pred_positions


def pred_map_mul(pred_positions, mask):
    return [pred_positions[0] * mask,
            pred_positions[1] * mask,
            pred_positions[2] * mask]


def rots_mul_vecs(m, v):
    """Apply rotations 'm' to vectors 'v'."""

    return [m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
            m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
            m[6] * v[0] + m[7] * v[1] + m[8] * v[2]]


def rigids_mul_vecs(r, v):
    """Apply rigid transforms 'r' to points 'v'."""

    rots = rots_mul_vecs(r, v)
    vecs_add_r = [rots[0] + r[9],
                  rots[1] + r[10],
                  rots[2] + r[11]]
    return vecs_add_r


def vecs_from_tensor(x):  # shape (...)
    """Converts from tensor of shape (3,) to Vecs."""
    return x[..., 0], x[..., 1], x[..., 2]


def get_exp_atom_pos(atom_pos):
    return [mnp.expand_dims(atom_pos[0], axis=0),
            mnp.expand_dims(atom_pos[1], axis=0),
            mnp.expand_dims(atom_pos[2], axis=0)
           ]


def quat_multiply_by_vec(quat, vec):
    """Multiply a quaternion by a pure-vector quaternion."""

    return mnp.sum(residue_constants.QUAT_MULTIPLY_BY_VEC * quat[..., :, None, None] * vec[..., None, :, None],
                   axis=(-3, -2))


def rigids_mul_rots(xx, xy, xz, yx, yy, yz, zx, zy, zz, ones, zeros, cos_angles, sin_angles):
    """Compose rigid transformations 'r' with rotations 'm'."""

    c00 = xx * ones + xy * zeros + xz * zeros
    c01 = yx * ones + yy * zeros + yz * zeros
    c02 = zx * ones + zy * zeros + zz * zeros
    c10 = xx * zeros + xy * cos_angles + xz * sin_angles
    c11 = yx * zeros + yy * cos_angles + yz * sin_angles
    c12 = zx * zeros + zy * cos_angles + zz * sin_angles
    c20 = xx * zeros + xy * (-sin_angles) + xz * cos_angles
    c21 = yx * zeros + yy * (-sin_angles) + yz * cos_angles
    c22 = zx * zeros + zy * (-sin_angles) + zz * cos_angles
    return c00, c10, c20, c01, c11, c21, c02, c12, c22


def rigids_mul_rigids(a, b):
    """Group composition of Rigids 'a' and 'b'."""

    c00 = a[0] * b[0] + a[1] * b[3] + a[2] * b[6]
    c01 = a[3] * b[0] + a[4] * b[3] + a[5] * b[6]
    c02 = a[6] * b[0] + a[7] * b[3] + a[8] * b[6]

    c10 = a[0] * b[1] + a[1] * b[4] + a[2] * b[7]
    c11 = a[3] * b[1] + a[4] * b[4] + a[5] * b[7]
    c12 = a[6] * b[1] + a[7] * b[4] + a[8] * b[7]

    c20 = a[0] * b[2] + a[1] * b[5] + a[2] * b[8]
    c21 = a[3] * b[2] + a[4] * b[5] + a[5] * b[8]
    c22 = a[6] * b[2] + a[7] * b[5] + a[8] * b[8]

    tr0 = a[0] * b[9] + a[1] * b[10] + a[2] * b[11]
    tr1 = a[3] * b[9] + a[4] * b[10] + a[5] * b[11]
    tr2 = a[6] * b[9] + a[7] * b[10] + a[8] * b[11]

    new_tr0 = a[9] + tr0
    new_tr1 = a[10] + tr1
    new_tr2 = a[11] + tr2

    return [c00, c10, c20, c01, c11, c21, c02, c12, c22, new_tr0, new_tr1, new_tr2]


def rigits_concate_all(xall, x5, x6, x7):
    return [mnp.concatenate([xall[0][:, 0:5], x5[0][:, None], x6[0][:, None], x7[0][:, None]], axis=-1),
            mnp.concatenate([xall[1][:, 0:5], x5[1][:, None], x6[1][:, None], x7[1][:, None]], axis=-1),
            mnp.concatenate([xall[2][:, 0:5], x5[2][:, None], x6[2][:, None], x7[2][:, None]], axis=-1),
            mnp.concatenate([xall[3][:, 0:5], x5[3][:, None], x6[3][:, None], x7[3][:, None]], axis=-1),
            mnp.concatenate([xall[4][:, 0:5], x5[4][:, None], x6[4][:, None], x7[4][:, None]], axis=-1),
            mnp.concatenate([xall[5][:, 0:5], x5[5][:, None], x6[5][:, None], x7[5][:, None]], axis=-1),
            mnp.concatenate([xall[6][:, 0:5], x5[6][:, None], x6[6][:, None], x7[6][:, None]], axis=-1),
            mnp.concatenate([xall[7][:, 0:5], x5[7][:, None], x6[7][:, None], x7[7][:, None]], axis=-1),
            mnp.concatenate([xall[8][:, 0:5], x5[8][:, None], x6[8][:, None], x7[8][:, None]], axis=-1),
            mnp.concatenate([xall[9][:, 0:5], x5[9][:, None], x6[9][:, None], x7[9][:, None]], axis=-1),
            mnp.concatenate([xall[10][:, 0:5], x5[10][:, None], x6[10][:, None], x7[10][:, None]], axis=-1),
            mnp.concatenate([xall[11][:, 0:5], x5[11][:, None], x6[11][:, None], x7[11][:, None]], axis=-1)
           ]


def reshape_back(backb):
    return [backb[0][:, None],
            backb[1][:, None],
            backb[2][:, None],
            backb[3][:, None],
            backb[4][:, None],
            backb[5][:, None],
            backb[6][:, None],
            backb[7][:, None],
            backb[8][:, None],
            backb[9][:, None],
            backb[10][:, None],
            backb[11][:, None]
           ]


def l2_normalize(x, axis=-1):
    return x / mnp.sqrt(mnp.sum(x ** 2, axis=axis, keepdims=True))


def torsion_angles_to_frames(aatype, backb_to_global, torsion_angles_sin_cos, restype_rigid_group_default_frame):
    """Compute rigid group frames from torsion angles."""

    # Gather the default frames for all rigid groups.
    m = P.Gather()(restype_rigid_group_default_frame, aatype, 0)

    xx1, xy1, xz1, yx1, yy1, yz1, zx1, zy1, zz1, x1, y1, z1 = rigids_from_tensor4x4(m)

    # Create the rotation matrices according to the given angles (each frame is
    # defined such that its rotation is around the x-axis).
    sin_angles = torsion_angles_sin_cos[..., 0]
    cos_angles = torsion_angles_sin_cos[..., 1]

    # insert zero rotation for backbone group.
    num_residues, = aatype.shape
    sin_angles = mnp.concatenate([mnp.zeros([num_residues, 1]), sin_angles], axis=-1)
    cos_angles = mnp.concatenate([mnp.ones([num_residues, 1]), cos_angles], axis=-1)
    zeros = mnp.zeros_like(sin_angles)
    ones = mnp.ones_like(sin_angles)
    # Apply rotations to the frames.
    xx2, xy2, xz2, yx2, yy2, yz2, zx2, zy2, zz2 = rigids_mul_rots(xx1, xy1, xz1, yx1, yy1, yz1, zx1, zy1, zz1,
                                                                  ones, zeros, cos_angles, sin_angles)
    all_frames = [xx2, xy2, xz2, yx2, yy2, yz2, zx2, zy2, zz2, x1, y1, z1]
    # chi2, chi3, and chi4 frames do not transform to the backbone frame but to
    # the previous frame. So chain them up accordingly.
    chi2_frame_to_frame = [xx2[:, 5], xy2[:, 5], xz2[:, 5], yx2[:, 5], yy2[:, 5], yz2[:, 5], zx2[:, 5], zy2[:, 5],
                           zz2[:, 5], x1[:, 5], y1[:, 5], z1[:, 5]]
    chi3_frame_to_frame = [xx2[:, 6], xy2[:, 6], xz2[:, 6], yx2[:, 6], yy2[:, 6], yz2[:, 6], zx2[:, 6], zy2[:, 6],
                           zz2[:, 6], x1[:, 6], y1[:, 6], z1[:, 6]]
    chi4_frame_to_frame = [xx2[:, 7], xy2[:, 7], xz2[:, 7], yx2[:, 7], yy2[:, 7], yz2[:, 7], zx2[:, 7], zy2[:, 7],
                           zz2[:, 7], x1[:, 7], y1[:, 7], z1[:, 7]]
    #
    chi1_frame_to_backb = [xx2[:, 4], xy2[:, 4], xz2[:, 4], yx2[:, 4], yy2[:, 4], yz2[:, 4], zx2[:, 4], zy2[:, 4],
                           zz2[:, 4], x1[:, 4], y1[:, 4], z1[:, 4]]

    chi2_frame_to_backb = rigids_mul_rigids(chi1_frame_to_backb, chi2_frame_to_frame)
    chi3_frame_to_backb = rigids_mul_rigids(chi2_frame_to_backb, chi3_frame_to_frame)
    chi4_frame_to_backb = rigids_mul_rigids(chi3_frame_to_backb, chi4_frame_to_frame)

    # Recombine them to a r3.Rigids with shape (N, 8).
    all_frames_to_backb = rigits_concate_all(all_frames, chi2_frame_to_backb,
                                             chi3_frame_to_backb, chi4_frame_to_backb)
    backb_to_global_new = reshape_back(backb_to_global)
    # Create the global frames.
    all_frames_to_global = rigids_mul_rigids(backb_to_global_new, all_frames_to_backb)
    return all_frames_to_global


def map_atoms_to_global_func(all_frames, group_mask):
    return [mnp.sum(all_frames[0][:, None, :] * group_mask, axis=-1),
            mnp.sum(all_frames[1][:, None, :] * group_mask, axis=-1),
            mnp.sum(all_frames[2][:, None, :] * group_mask, axis=-1),
            mnp.sum(all_frames[3][:, None, :] * group_mask, axis=-1),
            mnp.sum(all_frames[4][:, None, :] * group_mask, axis=-1),
            mnp.sum(all_frames[5][:, None, :] * group_mask, axis=-1),
            mnp.sum(all_frames[6][:, None, :] * group_mask, axis=-1),
            mnp.sum(all_frames[7][:, None, :] * group_mask, axis=-1),
            mnp.sum(all_frames[8][:, None, :] * group_mask, axis=-1),
            mnp.sum(all_frames[9][:, None, :] * group_mask, axis=-1),
            mnp.sum(all_frames[10][:, None, :] * group_mask, axis=-1),
            mnp.sum(all_frames[11][:, None, :] * group_mask, axis=-1)
           ]


def get_exp_frames(frames):
    return [mnp.expand_dims(frames[0], axis=0),
            mnp.expand_dims(frames[1], axis=0),
            mnp.expand_dims(frames[2], axis=0),
            mnp.expand_dims(frames[3], axis=0),
            mnp.expand_dims(frames[4], axis=0),
            mnp.expand_dims(frames[5], axis=0),
            mnp.expand_dims(frames[6], axis=0),
            mnp.expand_dims(frames[7], axis=0),
            mnp.expand_dims(frames[8], axis=0),
            mnp.expand_dims(frames[9], axis=0),
            mnp.expand_dims(frames[10], axis=0),
            mnp.expand_dims(frames[11], axis=0)
           ]


def vecs_to_tensor(v):
    """Converts 'v' to tensor with shape 3, inverse of 'vecs_from_tensor'."""

    return mnp.stack([v[0], v[1], v[2]], axis=-1)


def atom14_to_atom37(atom14_data, residx_atom37_to_atom14, atom37_atom_exists, indices0):
    """Convert atom14 to atom37 representation."""

    seq_length = atom14_data.shape[0]
    residx_atom37_to_atom14 = residx_atom37_to_atom14.reshape((seq_length, 37, 1))
    new_indices = P.Concat(2)((indices0, residx_atom37_to_atom14))

    atom37_data = P.GatherNd()(atom14_data, new_indices)

    if len(atom14_data.shape) == 2:
        atom37_data *= atom37_atom_exists
    elif len(atom14_data.shape) == 3:
        atom37_data *= atom37_atom_exists[:, :, None].astype(atom37_data.dtype)

    return atom37_data


def compute_confidence(predicted_lddt_logits):
    """compute confidence"""

    num_bins = predicted_lddt_logits.shape[-1]
    bin_width = 1 / num_bins
    start_n = bin_width / 2
    plddt = compute_plddt(predicted_lddt_logits, start_n, bin_width)
    confidence = np.mean(plddt)
    return confidence


def compute_plddt(logits, start_n, bin_width):
    """Computes per-residue pLDDT from logits.

    Args:
      logits: [num_res, num_bins] output from the PredictedLDDTHead.

    Returns:
      plddt: [num_res] per-residue pLDDT.
    """
    bin_centers = np.arange(start=start_n, stop=1.0, step=bin_width)
    probs = softmax(logits, axis=-1)
    predicted_lddt_ca = np.sum(probs * bin_centers[None, :], axis=-1)
    return predicted_lddt_ca * 100


def make_atom14_positions(aatype, all_atom_mask, all_atom_positions):
    """Constructs denser atom positions (14 dimensions instead of 37).
    Args:
        input with "aatype", "all_atom_positions" and "all_atom_mask"

    Returns:
        * 'atom14_atom_exists': atom14 position exists mask
        * 'atom14_gt_exists': ground truth atom14 position exists mask
        * 'atom14_gt_positions': ground truth atom14 positions
        * 'residx_atom14_to_atom37': mapping for (residx, atom14) --> atom37, i.e. an array
        * 'residx_atom37_to_atom14': gather indices for mapping back
        * 'atom37_atom_exists': atom37 exists mask
        * 'atom14_alt_gt_positions': apply transformation matrices for the given residue sequence to the ground
         truth positions
        * 'atom14_alt_gt_exists': the mask for the alternative ground truth
        * 'atom14_atom_is_ambiguous': create an ambiguous_mask for the given sequence

    """
    restype_atom14_to_atom37 = []  # mapping (restype, atom14) --> atom37
    restype_atom37_to_atom14 = []  # mapping (restype, atom37) --> atom14
    restype_atom14_mask = []

    for rt in residue_constants.restypes:
        atom_names = residue_constants.restype_name_to_atom14_names[
            residue_constants.restype_1to3[rt]]

        restype_atom14_to_atom37.append([
            (residue_constants.atom_order[name] if name else 0)
            for name in atom_names
        ])

        atom_name_to_idx14 = {name: i for i, name in enumerate(atom_names)}
        restype_atom37_to_atom14.append([
            (atom_name_to_idx14[name] if name in atom_name_to_idx14 else 0)
            for name in residue_constants.atom_types
        ])

        restype_atom14_mask.append([(1. if name else 0.) for name in atom_names])

    # Add dummy mapping for restype 'UNK'.
    restype_atom14_to_atom37.append([0] * 14)
    restype_atom37_to_atom14.append([0] * 37)
    restype_atom14_mask.append([0.] * 14)

    restype_atom14_to_atom37 = np.array(restype_atom14_to_atom37, dtype=np.int32)
    restype_atom37_to_atom14 = np.array(restype_atom37_to_atom14, dtype=np.int32)
    restype_atom14_mask = np.array(restype_atom14_mask, dtype=np.float32)

    # Create the mapping for (residx, atom14) --> atom37, i.e. an array
    # with shape (num_res, 14) containing the atom37 indices for this protein.
    residx_atom14_to_atom37 = restype_atom14_to_atom37[aatype]
    residx_atom14_mask = restype_atom14_mask[aatype]

    # Create a mask for known ground truth positions.
    residx_atom14_gt_mask = residx_atom14_mask * np.take_along_axis(
        all_atom_mask, residx_atom14_to_atom37, axis=1).astype(np.float32)

    # Gather the ground truth positions.
    residx_atom14_gt_positions = residx_atom14_gt_mask[:, :, None] * (
        np.take_along_axis(all_atom_positions, residx_atom14_to_atom37[..., None], axis=1))

    atom14_atom_exists = residx_atom14_mask
    atom14_gt_exists = residx_atom14_gt_mask
    atom14_gt_positions = residx_atom14_gt_positions

    residx_atom14_to_atom37 = residx_atom14_to_atom37

    # Create the gather indices for mapping back.
    residx_atom37_to_atom14 = restype_atom37_to_atom14[aatype]

    # Create the corresponding mask.
    restype_atom37_mask = np.zeros([21, 37], dtype=np.float32)
    for restype, restype_letter in enumerate(residue_constants.restypes):
        restype_name = residue_constants.restype_1to3[restype_letter]
        atom_names = residue_constants.residue_atoms[restype_name]
        for atom_name in atom_names:
            atom_type = residue_constants.atom_order[atom_name]
            restype_atom37_mask[restype, atom_type] = 1

    atom37_atom_exists = restype_atom37_mask[aatype]

    # As the atom naming is ambiguous for 7 of the 20 amino acids, provide
    # alternative ground truth coordinates where the naming is swapped
    restype_3 = [
        residue_constants.restype_1to3[res] for res in residue_constants.restypes
    ]
    restype_3 += ["UNK"]

    # Matrices for renaming ambiguous atoms.
    all_matrices = {res: np.eye(14, dtype=np.float32) for res in restype_3}
    for resname, swap in residue_constants.residue_atom_renaming_swaps.items():
        correspondences = np.arange(14)
        for source_atom_swap, target_atom_swap in swap.items():
            source_index = residue_constants.restype_name_to_atom14_names.get(resname).index(source_atom_swap)
            target_index = residue_constants.restype_name_to_atom14_names.get(resname).index(target_atom_swap)
            correspondences[source_index] = target_index
            correspondences[target_index] = source_index
            renaming_matrix = np.zeros((14, 14), dtype=np.float32)
            for index, correspondence in enumerate(correspondences):
                renaming_matrix[index, correspondence] = 1.
        all_matrices[resname] = renaming_matrix.astype(np.float32)
    renaming_matrices = np.stack([all_matrices[restype] for restype in restype_3])

    # Pick the transformation matrices for the given residue sequence
    # shape (num_res, 14, 14).
    renaming_transform = renaming_matrices[aatype]

    # Apply it to the ground truth positions. shape (num_res, 14, 3).
    alternative_gt_positions = np.einsum("rac,rab->rbc", residx_atom14_gt_positions, renaming_transform)
    atom14_alt_gt_positions = alternative_gt_positions

    # Create the mask for the alternative ground truth (differs from the
    # ground truth mask, if only one of the atoms in an ambiguous pair has a
    # ground truth position).
    alternative_gt_mask = np.einsum("ra,rab->rb", residx_atom14_gt_mask, renaming_transform)

    atom14_alt_gt_exists = alternative_gt_mask

    # Create an ambiguous atoms mask.  shape: (21, 14).
    restype_atom14_is_ambiguous = np.zeros((21, 14), dtype=np.float32)
    for resname, swap in residue_constants.residue_atom_renaming_swaps.items():
        for atom_name1, atom_name2 in swap.items():
            restype = residue_constants.restype_order[
                residue_constants.restype_3to1[resname]]
            atom_idx1 = residue_constants.restype_name_to_atom14_names.get(resname).index(atom_name1)
            atom_idx2 = residue_constants.restype_name_to_atom14_names.get(resname).index(atom_name2)
            restype_atom14_is_ambiguous[restype, atom_idx1] = 1
            restype_atom14_is_ambiguous[restype, atom_idx2] = 1

    # From this create an ambiguous_mask for the given sequence.
    atom14_atom_is_ambiguous = restype_atom14_is_ambiguous[aatype]
    return_pack = (atom14_atom_exists, atom14_gt_exists, atom14_gt_positions, residx_atom14_to_atom37,
                   residx_atom37_to_atom14, atom37_atom_exists, atom14_alt_gt_positions, atom14_alt_gt_exists,
                   atom14_atom_is_ambiguous)
    return return_pack


def get_pdb_info(pdb_path):
    """get atom positions, residue index etc. info from pdb file

    """
    with open(pdb_path, 'r', encoding="UTF-8") as f:
        prot_pdb = protein.from_pdb_string(f.read())
    aatype = prot_pdb.aatype
    atom37_positions = prot_pdb.atom_positions.astype(np.float32)
    atom37_mask = prot_pdb.atom_mask.astype(np.float32)

    # get ground truth of atom14
    features = {'aatype': aatype,
                'all_atom_positions': atom37_positions,
                'all_atom_mask': atom37_mask}
    atom14_atom_exists, atom14_gt_exists, atom14_gt_positions, residx_atom14_to_atom37, residx_atom37_to_atom14, \
    atom37_atom_exists, atom14_alt_gt_positions, atom14_alt_gt_exists, atom14_atom_is_ambiguous = \
        make_atom14_positions(aatype, atom37_mask, atom37_positions)
    features.update({"atom14_atom_exists": atom14_atom_exists,
                     "atom14_gt_exists": atom14_gt_exists,
                     "atom14_gt_positions": atom14_gt_positions,
                     "residx_atom14_to_atom37": residx_atom14_to_atom37,
                     "residx_atom37_to_atom14": residx_atom37_to_atom14,
                     "atom37_atom_exists": atom37_atom_exists,
                     "atom14_alt_gt_positions": atom14_alt_gt_positions,
                     "atom14_alt_gt_exists": atom14_alt_gt_exists,
                     "atom14_atom_is_ambiguous": atom14_atom_is_ambiguous})

    features["residue_index"] = prot_pdb.residue_index

    return features


def get_fasta_info(pdb_path):
    # get fasta info from pdb
    with open(pdb_path, 'r', encoding='UTF-8') as f:
        prot_pdb = protein.from_pdb_string(f.read())
    aatype = prot_pdb.aatype
    fasta = [residue_constants.order_restype.get(x) for x in aatype]

    return ''.join(fasta)


def get_aligned_seq(gt_seq, pr_seq):
    """align two protein fasta sequence"""
    aligner = Align.PairwiseAligner()
    substitution_matrices.load()
    matrix = substitution_matrices.load("BLOSUM62")
    for i in range(len(str(matrix.alphabet))):
        res = matrix.alphabet[i]
        matrix['X'][res] = 0
        matrix[res]['X'] = 0
    aligner.substitution_matrix = matrix
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -1
    # many align results, get only the one w/ highest score. gt_seq as reference
    alignments = aligner.align(gt_seq, pr_seq)
    align = alignments[0]
    align_str = str(align)
    align_str_len = len(align_str)
    point = []
    target = ''
    align_relationship = ''
    query = ''
    for i in range(align_str_len):
        if align_str[i] == '\n':
            point.append(i)
    for i in range(int(point[0])):
        target = target + align_str[i]
    for i in range(int(point[1])-int(point[0])-1):
        align_relationship = align_relationship + align_str[i + int(point[0])+1]
    for i in range(int(point[2])-int(point[1])-1):
        query = query + align_str[i + int(point[1])+1]
    return target, align_relationship, query
