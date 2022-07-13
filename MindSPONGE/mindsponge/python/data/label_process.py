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
"""label process"""
import numpy as np
from mindsponge.common.residue_constants import chi_angles_mask, chi_pi_periodic, restypes, \
 restype_1to3, chi_angles_atoms, atom_order, residue_atom_renaming_swaps, restype_order, restype_3to1
import mindsponge.common.r3_np as r3


def atom37_to_torsion_angles(
        aatype: np.ndarray,  # inputs1 shape (N,)
        all_atom_pos: np.ndarray,  # inputs2 shape (N, 37, 3)
        all_atom_mask: np.ndarray,  # inputs3 shape (N, 37)
        alt_torsions=False,
):
    """get the torsion angles of each residue"""

    true_aatype = np.minimum(aatype, 20)

    # get the number residue
    num_batch, num_res = true_aatype.shape

    paddings = np.zeros([num_batch, 1, 37, 3], np.float32)
    padding_atom_pos = np.concatenate([paddings, all_atom_pos[:, :-1, :, :]], axis=1)

    paddings = np.zeros([num_batch, 1, 37], np.float32)
    padding_atom_mask = np.concatenate([paddings, all_atom_mask[:, :-1, :]], axis=1)

    # compute padding atom position for omega, phi and psi
    omega_atom_pos_padding = np.concatenate(
        [padding_atom_pos[..., 1:3, :],
         all_atom_pos[..., 0:2, :]
         ], axis=-2)
    phi_atom_pos_padding = np.concatenate(
        [padding_atom_pos[..., 2:3, :],
         all_atom_pos[..., 0:3, :]
         ], axis=-2)
    psi_atom_pos_padding = np.concatenate(
        [all_atom_pos[..., 0:3, :],
         all_atom_pos[..., 4:5, :]
         ], axis=-2)

    # compute padding atom position mask for omega, phi and psi
    omega_mask_padding = (np.prod(padding_atom_mask[..., 1:3], axis=-1) *
                          np.prod(all_atom_mask[..., 0:2], axis=-1))
    phi_mask_padding = (padding_atom_mask[..., 2] * np.prod(all_atom_mask[..., 0:3], axis=-1))
    psi_mask_padding = (np.prod(all_atom_mask[..., 0:3], axis=-1) * all_atom_mask[..., 4])

    chi_atom_pos_indices = get_chi_atom_pos_indices()
    atom_pos_indices = np_gather_ops(chi_atom_pos_indices, true_aatype, 0, 0)
    chi_atom_pos = np_gather_ops(all_atom_pos, atom_pos_indices, -2, 2)

    angles_mask = list(chi_angles_mask)
    angles_mask.append([0.0, 0.0, 0.0, 0.0])
    angles_mask = np.array(angles_mask)

    chis_mask = np_gather_ops(angles_mask, true_aatype, 0, 0)

    chi_angle_atoms_mask = np_gather_ops(all_atom_mask, atom_pos_indices, -1, 2)

    chi_angle_atoms_mask = np.prod(chi_angle_atoms_mask, axis=-1)
    chis_mask = chis_mask * chi_angle_atoms_mask.astype(np.float32)

    torsions_atom_pos_padding = np.concatenate(
        [omega_atom_pos_padding[:, :, None, :, :],
         phi_atom_pos_padding[:, :, None, :, :],
         psi_atom_pos_padding[:, :, None, :, :],
         chi_atom_pos
         ], axis=2)

    torsion_angles_mask_padding = np.concatenate(
        [omega_mask_padding[:, :, None],
         phi_mask_padding[:, :, None],
         psi_mask_padding[:, :, None],
         chis_mask
         ], axis=2)

    torsion_frames = r3.rigids_from_3_points(
        point_on_neg_x_axis=r3.vecs_from_tensor(torsions_atom_pos_padding[:, :, :, 1, :]),
        origin=r3.vecs_from_tensor(torsions_atom_pos_padding[:, :, :, 2, :]),
        point_on_xy_plane=r3.vecs_from_tensor(torsions_atom_pos_padding[:, :, :, 0, :]))

    forth_atom_rel_pos = r3.rigids_mul_vecs(
        r3.invert_rigids(torsion_frames),
        r3.vecs_from_tensor(torsions_atom_pos_padding[:, :, :, 3, :]))

    torsion_angles_sin_cos = np.stack(
        [forth_atom_rel_pos.z, forth_atom_rel_pos.y], axis=-1)
    torsion_angles_sin_cos /= np.sqrt(
        np.sum(np.square(torsion_angles_sin_cos), axis=-1, keepdims=True)
        + 1e-8)

    torsion_angles_sin_cos *= np.array(
        [1., 1., -1., 1., 1., 1., 1.])[None, None, :, None]

    chi_is_ambiguous = np_gather_ops(
        np.array(chi_pi_periodic), true_aatype)
    mirror_torsion_angles = np.concatenate(
        [np.ones([num_batch, num_res, 3]),
         1.0 - 2.0 * chi_is_ambiguous], axis=-1)
    alt_torsion_angles_sin_cos = (torsion_angles_sin_cos * mirror_torsion_angles[:, :, :, None])

    if alt_torsions:
        fix_torsions = np.stack([np.ones(torsion_angles_sin_cos.shape[:-1]),
                                 np.zeros(torsion_angles_sin_cos.shape[:-1])], axis=-1)
        torsion_angles_sin_cos = torsion_angles_sin_cos * torsion_angles_mask_padding[
            ..., None] + fix_torsions * (1 - torsion_angles_mask_padding[..., None])
        alt_torsion_angles_sin_cos = alt_torsion_angles_sin_cos * torsion_angles_mask_padding[
            ..., None] + fix_torsions * (1 - torsion_angles_mask_padding[..., None])

    return {
        'torsion_angles_sin_cos': torsion_angles_sin_cos[0],  # (N, 7, 2)
        'alt_torsion_angles_sin_cos': alt_torsion_angles_sin_cos[0],  # (N, 7, 2)
        'torsion_angles_mask': torsion_angles_mask_padding[0]  # (N, 7)
    }


def atom37_to_frames(
        aatype,  # inputs1 shape (...)
        all_atom_positions,  # inputs2 shape (..., 37, 3)
        all_atom_mask,  # inputs3 shape (..., 37)
        is_affine=False
):
    """get the frames and affine for each residue"""
    aatype_shape = aatype.shape

    flat_aatype = np.reshape(aatype, [-1])
    all_atom_positions = np.reshape(all_atom_positions, [-1, 37, 3])
    all_atom_mask = np.reshape(all_atom_mask, [-1, 37])

    rigid_group_names_res = np.full([21, 8, 3], '', dtype=object)

    # group 0: backbone frame
    rigid_group_names_res[:, 0, :] = ['C', 'CA', 'N']

    # group 3: 'psi'
    rigid_group_names_res[:, 3, :] = ['CA', 'C', 'O']

    # group 4,5,6,7: 'chi1,2,3,4'
    for restype, letter in enumerate(restypes):
        restype_name = restype_1to3[letter]
        for chi_idx in range(4):
            if chi_angles_mask[restype][chi_idx]:
                atom_names = chi_angles_atoms[restype_name][chi_idx]
                rigid_group_names_res[restype, chi_idx + 4, :] = atom_names[1:]

    # create rigid group mask
    rigid_group_mask_res = np.zeros([21, 8], dtype=np.float32)
    rigid_group_mask_res[:, 0] = 1
    rigid_group_mask_res[:, 3] = 1
    rigid_group_mask_res[:20, 4:] = chi_angles_mask

    lookup_table = atom_order.copy()
    lookup_table[''] = 0
    rigid_group_atom37_idx_restype = np.vectorize(lambda x: lookup_table[x])(
        rigid_group_names_res)

    rigid_group_atom37_idx_residx = np_gather_ops(
        rigid_group_atom37_idx_restype, flat_aatype)

    base_atom_pos = np_gather_ops(
        all_atom_positions,
        rigid_group_atom37_idx_residx,
        batch_dims=1)

    gt_frames = r3.rigids_from_3_points(
        point_on_neg_x_axis=r3.vecs_from_tensor(base_atom_pos[:, :, 0, :]),
        origin=r3.vecs_from_tensor(base_atom_pos[:, :, 1, :]),
        point_on_xy_plane=r3.vecs_from_tensor(base_atom_pos[:, :, 2, :])
    )

    # get the group mask
    group_masks = np_gather_ops(rigid_group_mask_res, flat_aatype)

    # get the atom mask
    gt_atoms_exists = np_gather_ops(
        all_atom_mask.astype(np.float32),
        rigid_group_atom37_idx_residx,
        batch_dims=1)
    gt_masks = np.min(gt_atoms_exists, axis=-1) * group_masks

    rotations = np.tile(np.eye(3, dtype=np.float32), [8, 1, 1])
    rotations[0, 0, 0] = -1
    rotations[0, 2, 2] = -1
    gt_frames = r3.rigids_mul_rots(gt_frames, r3.rots_from_tensor3x3(rotations))

    rigid_group_is_ambiguous_res = np.zeros([21, 8], dtype=np.float32)
    rigid_group_rotations_res = np.tile(np.eye(3, dtype=np.float32), [21, 8, 1, 1])

    for restype_name, _ in residue_atom_renaming_swaps.items():
        restype = restype_order[restype_3to1[restype_name]]
        chi_idx = int(sum(chi_angles_mask[restype]) - 1)
        rigid_group_is_ambiguous_res[restype, chi_idx + 4] = 1
        rigid_group_rotations_res[restype, chi_idx + 4, 1, 1] = -1
        rigid_group_rotations_res[restype, chi_idx + 4, 2, 2] = -1

    # Gather the ambiguity information for each residue.
    rigid_group_is_ambiguous_res_index = np_gather_ops(
        rigid_group_is_ambiguous_res, flat_aatype)
    rigid_group_ambiguity_rotation_res_index = np_gather_ops(
        rigid_group_rotations_res, flat_aatype)

    # Create the alternative ground truth frames.
    alt_gt_frames = r3.rigids_mul_rots(
        gt_frames, r3.rots_from_tensor3x3(rigid_group_ambiguity_rotation_res_index))

    gt_frames_flat12 = r3.rigids_to_tensor_flat12(gt_frames)
    alt_gt_frames_flat12 = r3.rigids_to_tensor_flat12(alt_gt_frames)

    # reshape back to original residue layout
    gt_frames_flat12 = np.reshape(gt_frames_flat12, aatype_shape + (8, 12))
    gt_masks = np.reshape(gt_masks, aatype_shape + (8,))
    group_masks = np.reshape(group_masks, aatype_shape + (8,))
    gt_frames_flat12 = np.reshape(gt_frames_flat12, aatype_shape + (8, 12))
    rigid_group_is_ambiguous_res_index = np.reshape(rigid_group_is_ambiguous_res_index, aatype_shape + (8,))
    alt_gt_frames_flat12 = np.reshape(alt_gt_frames_flat12,
                                      aatype_shape + (8, 12,))
    if not is_affine:
        return {
            'rigidgroups_gt_frames': gt_frames_flat12,  # shape (..., 8, 12)
            'rigidgroups_gt_exists': gt_masks,  # shape (..., 8)
            'rigidgroups_group_exists': group_masks,  # shape (..., 8)
            'rigidgroups_group_is_ambiguous':
                rigid_group_is_ambiguous_res_index,  # shape (..., 8)
            'rigidgroups_alt_gt_frames': alt_gt_frames_flat12,  # shape (..., 8, 12)
        }

    rotation = [[gt_frames.rot.xx, gt_frames.rot.xy, gt_frames.rot.xz],
                [gt_frames.rot.yx, gt_frames.rot.yy, gt_frames.rot.yz],
                [gt_frames.rot.zx, gt_frames.rot.zy, gt_frames.rot.zz]],
    translation = [gt_frames.trans.x, gt_frames.trans.y, gt_frames.trans.z]
    backbone_affine_tensor = to_tensor(rotation[0], translation)[:, 0, :]
    return {
        'rigidgroups_gt_frames': gt_frames_flat12,  # shape (..., 8, 12)
        'rigidgroups_gt_exists': gt_masks,  # shape (..., 8)
        'rigidgroups_group_exists': group_masks,  # shape (..., 8)
        'rigidgroups_group_is_ambiguous': rigid_group_is_ambiguous_res_index,  # shape (..., 8)
        'rigidgroups_alt_gt_frames': alt_gt_frames_flat12,  # shape (..., 8, 12)
        'backbone_affine_tensor': backbone_affine_tensor,  # shape (..., 7)
    }


def get_chi_atom_pos_indices():
    """get the atom indices for computing chi angles for all residue types"""
    chi_atom_pos_indices = []
    for residue_name in restypes:
        residue_name = restype_1to3[residue_name]
        residue_chi_angles = chi_angles_atoms[residue_name]
        atom_pos_indices = []
        for chi_angle in residue_chi_angles:
            atom_pos_indices.append([atom_order[atom] for atom in chi_angle])
        for _ in range(4 - len(atom_pos_indices)):
            atom_pos_indices.append([0, 0, 0, 0])  # For chi angles not defined on the AA.
        chi_atom_pos_indices.append(atom_pos_indices)

    chi_atom_pos_indices.append([[0, 0, 0, 0]] * 4)  # For UNKNOWN residue.

    return np.array(chi_atom_pos_indices)


def gather(params, indices, axis=0):
    """gather operation"""
    func = lambda p, i: np.take(p, i, axis=axis)
    return func(params, indices)


def np_gather_ops(params, indices, axis=0, batch_dims=0):
    """np gather operation"""
    if batch_dims == 0:
        return gather(params, indices)
    result = []
    if batch_dims == 1:
        for p, i in zip(params, indices):
            axis = axis - batch_dims if axis - batch_dims > 0 else 0
            r = gather(p, i, axis=axis)
            result.append(r)
        return np.stack(result)
    for p, i in zip(params[0], indices[0]):
        r = gather(p, i, axis=axis)
        result.append(r)
    res = np.stack(result)
    return res.reshape((1,) + res.shape)


def rot_to_quat(rot, unstack_inputs=False):
    """transfer the rotation matrix to quaternion matrix"""
    if unstack_inputs:
        rot = [np.moveaxis(x, -1, 0) for x in np.moveaxis(rot, -2, 0)]
    [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]] = rot

    k = [[xx + yy + zz, zy - yz, xz - zx, yx - xy,],
         [zy - yz, xx - yy - zz, xy + yx, xz + zx,],
         [xz - zx, xy + yx, yy - xx - zz, yz + zy,],
         [yx - xy, xz + zx, yz + zy, zz - xx - yy,]]

    k = (1. / 3.) * np.stack([np.stack(x, axis=-1) for x in k],
                             axis=-2)

    # compute eigenvalues
    _, qs = np.linalg.eigh(k)
    return qs[..., -1]


def to_tensor(rotation, translation):
    """get affine based on rotation and translation"""
    quaternion = rot_to_quat(rotation)
    return np.concatenate(
        [quaternion] +
        [np.expand_dims(x, axis=-1) for x in translation],
        axis=-1)
