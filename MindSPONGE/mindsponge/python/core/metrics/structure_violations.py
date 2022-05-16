"""Modules and utilities for the structure module."""

from mindspore import Tensor
import mindspore.numpy as mnp
from mindspore.ops import operations as P
import mindspore as ms
from mindspore import nn

from ...common import residue_constants


def between_residue_bond(
        pred_atom_positions,  # (N, 37(14), 3)
        pred_atom_mask,  # (N, 37(14))
        residue_index,  # (N)
        aatype,  # (N)
        tolerance_factor_soft=12.0,
        tolerance_factor_hard=12.0
):
    """Flat-bottom loss to penalize structural violations between residues.

    This is a loss penalizing any violation of the geometry around the peptide
    bond between consecutive amino acids. This loss corresponds to
    Jumper et al. (2021) Suppl. Sec. 1.9.11, eq 44, 45.

    Args:
      pred_atom_positions: Atom positions in atom37/14 representation
      pred_atom_mask: Atom mask in atom37/14 representation
      residue_index: Residue index for given amino acid, this is assumed to be
        monotonically increasing.
      aatype: Amino acid type of given residue
      tolerance_factor_soft: soft tolerance factor measured in standard deviations
        of pdb distributions
      tolerance_factor_hard: hard tolerance factor measured in standard deviations
        of pdb distributions

    Returns:
        'c_n_loss_mean': Loss for peptide bond length violations
        'ca_c_n_loss_mean': Loss for violations of bond angle around C spanned by CA, C, N
        'c_n_ca_loss_mean': Loss for violations of bond angle around N spanned by C, N, CA
        'per_residue_loss_sum': sum of all losses for each residue
        'per_residue_violation_mask': mask denoting all residues with violation present.
    """

    # Get the positions of the relevant backbone atoms.
    this_ca_pos = pred_atom_positions[:-1, 1, :]  # (N - 1, 3)
    this_ca_mask = pred_atom_mask[:-1, 1]  # (N - 1)
    this_c_pos = pred_atom_positions[:-1, 2, :]  # (N - 1, 3)
    this_c_mask = pred_atom_mask[:-1, 2]  # (N - 1)
    next_n_pos = pred_atom_positions[1:, 0, :]  # (N - 1, 3)
    next_n_mask = pred_atom_mask[1:, 0]  # (N - 1)
    next_ca_pos = pred_atom_positions[1:, 1, :]  # (N - 1, 3)
    next_ca_mask = pred_atom_mask[1:, 1]  # (N - 1)
    has_no_gap_mask = ((residue_index[1:] - residue_index[:-1]) == 1.0).astype(ms.float32)

    # Compute loss for the C--N bond.
    c_n_bond_length = mnp.sqrt(1e-6 + mnp.sum(mnp.square(this_c_pos - next_n_pos), axis=-1))

    # The C-N bond to proline has slightly different length because of the ring.
    next_is_proline = Tensor((aatype[1:] == residue_constants.resname_to_idx['PRO'])).astype(ms.float32)
    # ((residue_index[1:] - residue_index[:-1]) == 1.0).astype(ms.float32)
    gt_length = ((1. - next_is_proline) * residue_constants.between_res_bond_length_c_n[0]
                 + next_is_proline * residue_constants.between_res_bond_length_c_n[1])
    gt_stddev = ((1. - next_is_proline) * residue_constants.between_res_bond_length_stddev_c_n[0] +
                 next_is_proline * residue_constants.between_res_bond_length_stddev_c_n[1])
    c_n_bond_length_error = mnp.sqrt(1e-6 + mnp.square(c_n_bond_length - gt_length))
    c_n_loss_per_residue = nn.ReLU()(c_n_bond_length_error - tolerance_factor_soft * gt_stddev)
    mask = this_c_mask * next_n_mask * has_no_gap_mask
    c_n_loss_mean = mnp.sum(mask * c_n_loss_per_residue) / (mnp.sum(mask) + 1e-6)
    c_n_violation_mask = mask * (c_n_bond_length_error > (tolerance_factor_hard * gt_stddev))

    # Compute loss for the angles.
    ca_c_bond_length = mnp.sqrt(1e-6 + mnp.sum(mnp.square(this_ca_pos - this_c_pos), axis=-1))
    n_ca_bond_length = mnp.sqrt(1e-6 + mnp.sum(mnp.square(next_n_pos - next_ca_pos), axis=-1))

    c_ca_unit_vec = (this_ca_pos - this_c_pos) / ca_c_bond_length[:, None]
    c_n_unit_vec = (next_n_pos - this_c_pos) / c_n_bond_length[:, None]
    n_ca_unit_vec = (next_ca_pos - next_n_pos) / n_ca_bond_length[:, None]

    ca_c_n_cos_angle = mnp.sum(c_ca_unit_vec * c_n_unit_vec, axis=-1)
    gt_angle = residue_constants.between_res_cos_angles_ca_c_n[0]
    gt_stddev = residue_constants.between_res_bond_length_stddev_c_n[0]
    ca_c_n_cos_angle_error = mnp.sqrt(1e-6 + mnp.square(ca_c_n_cos_angle - gt_angle))
    ca_c_n_loss_per_residue = nn.ReLU()(ca_c_n_cos_angle_error - tolerance_factor_soft * gt_stddev)
    mask = this_ca_mask * this_c_mask * next_n_mask * has_no_gap_mask
    ca_c_n_loss_mean = mnp.sum(mask * ca_c_n_loss_per_residue) / (mnp.sum(mask) + 1e-6)
    ca_c_n_violation_mask = mask * (ca_c_n_cos_angle_error > (tolerance_factor_hard * gt_stddev))

    c_n_ca_cos_angle = mnp.sum((-c_n_unit_vec) * n_ca_unit_vec, axis=-1)
    gt_angle = residue_constants.between_res_cos_angles_c_n_ca[0]
    gt_stddev = residue_constants.between_res_cos_angles_c_n_ca[1]
    c_n_ca_cos_angle_error = mnp.sqrt(1e-6 + mnp.square(c_n_ca_cos_angle - gt_angle))
    c_n_ca_loss_per_residue = nn.ReLU()(c_n_ca_cos_angle_error - tolerance_factor_soft * gt_stddev)
    mask = this_c_mask * next_n_mask * next_ca_mask * has_no_gap_mask
    c_n_ca_loss_mean = mnp.sum(mask * c_n_ca_loss_per_residue) / (mnp.sum(mask) + 1e-6)
    c_n_ca_violation_mask = mask * (c_n_ca_cos_angle_error > (tolerance_factor_hard * gt_stddev))

    # Compute a per residue loss (equally distribute the loss to both
    # neighbouring residues).
    per_residue_loss_sum = c_n_loss_per_residue + ca_c_n_loss_per_residue + c_n_ca_loss_per_residue
    per_residue_loss_sum = 0.5 * (mnp.pad(per_residue_loss_sum, [[0, 1]]) + mnp.pad(per_residue_loss_sum, [[1, 0]]))

    # Compute hard violations.
    per_residue_violation_mask = mnp.max(mnp.stack([c_n_violation_mask, ca_c_n_violation_mask, c_n_ca_violation_mask]),
                                         axis=0)
    per_residue_violation_mask = mnp.maximum(mnp.pad(per_residue_violation_mask, [[0, 1]]),
                                             mnp.pad(per_residue_violation_mask, [[1, 0]]))

    return c_n_loss_mean, ca_c_n_loss_mean, c_n_ca_loss_mean, per_residue_loss_sum, per_residue_violation_mask


def between_residue_clash(
        atom14_pred_positions,  # (N, 14, 3)
        atom14_atom_exists,  # (N, 14)
        atom14_atom_radius,  # (N, 14)
        residue_index,  # (N)
        c_one_hot,
        n_one_hot,
        overlap_tolerance_soft,
        overlap_tolerance_hard,
        cys_sg_idx):
    """Loss to penalize steric clashes between residues.

    This is a loss penalizing any steric clashes due to non bonded atoms in
    different peptides coming too close. This loss corresponds to the part with
    different residues of
    Jumper et al. (2021) Suppl. Sec. 1.9.11, eq 46.

    Args:
      atom14_pred_positions: Predicted positions of atoms in
        global prediction frame
      atom14_atom_exists: Mask denoting whether atom at positions exists for given
        amino acid type
      atom14_atom_radius: Van der Waals radius for each atom.
      residue_index: Residue index for given amino acid.
      overlap_tolerance_soft: Soft tolerance factor.
      overlap_tolerance_hard: Hard tolerance factor.

    Returns:
        'mean_loss': average clash loss
        'per_atom_loss_sum': sum of all clash losses per atom, shape (N, 14)
        'per_atom_clash_mask': mask whether atom clashes with any other atom shape (N, 14)
    """

    # Create the distance matrix.
    # (N, N, 14, 14)
    dists = mnp.sqrt(1e-10 + mnp.sum(
        mnp.square(atom14_pred_positions[:, None, :, None, :] - atom14_pred_positions[None, :, None, :, :]), axis=-1))

    # Create the mask for valid distances.
    # shape (N, N, 14, 14)
    dists_mask = atom14_atom_exists[:, None, :, None] * atom14_atom_exists[None, :, None, :]

    # Mask out all the duplicate entries in the lower triangular matrix.
    # Also mask out the diagonal (atom-pairs from the same residue) -- these atoms
    # are handled separately.
    dists_mask *= (residue_index[:, None, None, None] < residue_index[None, :, None, None])

    # Backbone C--N bond between subsequent residues is no clash.

    neighbour_mask = ((residue_index[:, None, None, None] + 1) == residue_index[None, :, None, None])
    c_n_bonds = neighbour_mask * c_one_hot[None, None, :, None] * n_one_hot[None, None, None, :]
    dists_mask *= (1. - c_n_bonds)

    # Disulfide bridge between two cysteines is no clash.

    cys_sg_one_hot = nn.OneHot(depth=14)(cys_sg_idx)
    disulfide_bonds = (cys_sg_one_hot[None, None, :, None] * cys_sg_one_hot[None, None, None, :])
    dists_mask *= (1. - disulfide_bonds)

    # Compute the lower bound for the allowed distances.
    # shape (N, N, 14, 14)
    dists_lower_bound = dists_mask * (atom14_atom_radius[:, None, :, None] + atom14_atom_radius[None, :, None, :])

    # Compute the error.
    # shape (N, N, 14, 14)
    dists_to_low_error = dists_mask * nn.ReLU()(dists_lower_bound - overlap_tolerance_soft - dists)

    # Compute the mean loss.
    # shape ()
    mean_loss = mnp.sum(dists_to_low_error) / (1e-6 + mnp.sum(dists_mask))

    # Compute the per atom loss sum.
    # shape (N, 14)
    per_atom_loss_sum = P.ReduceSum()(dists_to_low_error, (0, 2)) + P.ReduceSum()(dists_to_low_error, (1, 3))

    # Compute the hard clash mask.
    # shape (N, N, 14, 14)
    clash_mask = dists_mask * (dists < (dists_lower_bound - overlap_tolerance_hard))

    # Compute the per atom clash.
    # shape (N, 14)
    per_atom_clash_mask = mnp.maximum(mnp.max(clash_mask, axis=[0, 2]), mnp.max(clash_mask, axis=[1, 3]))

    return mean_loss, per_atom_loss_sum, per_atom_clash_mask


def within_residue_violations(
        atom14_pred_positions,  # (N, 14, 3)
        atom14_atom_exists,  # (N, 14)
        atom14_dists_lower_bound,  # (N, 14, 14)
        atom14_dists_upper_bound,  # (N, 14, 14)
        tighten_bounds_for_loss,
        dists_mask_i
):
    """Loss to penalize steric clashes within residues.

    This is a loss penalizing any steric violations or clashes of non-bonded atoms
    in a given peptide. This loss corresponds to the part with
    the same residues of
    Jumper et al. (2021) Suppl. Sec. 1.9.11, eq 46.

    Args:
      atom14_pred_positions: Predicted positions of atoms in
        global prediction frame
      atom14_atom_exists: Mask denoting whether atom at positions exists for given
        amino acid type
      atom14_dists_lower_bound: Lower bound on allowed distances.
      atom14_dists_upper_bound: Upper bound on allowed distances
      tighten_bounds_for_loss: Extra factor to tighten loss

    Returns:
        'per_atom_loss_sum': sum of all clash losses per atom, shape (N, 14)
        'per_atom_clash_mask': mask whether atom clashes with any other atom shape (N, 14)
    """

    # Compute the mask for each residue.
    # shape (N, 14, 14)
    dists_masks = (1. - dists_mask_i[None])
    dists_masks *= (atom14_atom_exists[:, :, None] * atom14_atom_exists[:, None, :])

    # Distance matrix
    # shape (N, 14, 14)
    dists = mnp.sqrt(1e-10 + mnp.sum(
        mnp.square(atom14_pred_positions[:, :, None, :] - atom14_pred_positions[:, None, :, :]), axis=-1))

    # Compute the loss.
    # shape (N, 14, 14)
    dists_to_low_error = nn.ReLU()(atom14_dists_lower_bound + tighten_bounds_for_loss - dists)
    dists_to_high_error = nn.ReLU()(dists - (atom14_dists_upper_bound - tighten_bounds_for_loss))
    loss = dists_masks * (dists_to_low_error + dists_to_high_error)

    # Compute the per atom loss sum.
    # shape (N, 14)
    per_atom_loss_sum = mnp.sum(loss, axis=1) + mnp.sum(loss, axis=2)

    # Compute the violations mask.
    # shape (N, 14, 14)
    lower = (dists < atom14_dists_lower_bound).astype(ms.int32)
    high = (dists > atom14_dists_upper_bound).astype(ms.int32)
    violations = dists_masks * ((lower + high).astype(bool))

    # Compute the per atom violations.
    # shape (N, 14)
    per_atom_violations = mnp.maximum(mnp.max(violations, axis=1), mnp.max(violations, axis=2))

    return per_atom_loss_sum, per_atom_violations


def find_structural_violations(atom14_atom_exists, residue_index, aatype, residx_atom14_to_atom37,
                               atom14_pred_positions, violation_tolerance_factor, clash_overlap_tolerance,
                               lower_bound, upper_bound, atomtype_radius, c_one_hot, n_one_hot, dists_mask_i,
                               cys_sg_idx):
    """Computes several checks for structural violations."""

    # Compute between residue backbone violations of bonds and angles.
    c_n_loss_mean, ca_c_n_loss_mean, c_n_ca_loss_mean, per_residue_loss_sum, per_residue_violation_mask = \
        between_residue_bond(
            pred_atom_positions=atom14_pred_positions,
            pred_atom_mask=atom14_atom_exists.astype(mnp.float32),
            residue_index=residue_index.astype(mnp.float32),
            aatype=aatype,
            tolerance_factor_soft=violation_tolerance_factor,
            tolerance_factor_hard=violation_tolerance_factor)

    # Compute the Van der Waals radius for every atom
    # (the first letter of the atom name is the element type).
    # Shape: (N, 14).
    atom14_atom_radius = atom14_atom_exists * P.Gather()(atomtype_radius, residx_atom14_to_atom37, 0)

    # Compute the between residue clash loss.
    mean_loss, per_atom_loss_sum, per_atom_clash_mask = between_residue_clash(
        atom14_pred_positions=atom14_pred_positions,
        atom14_atom_exists=atom14_atom_exists,
        atom14_atom_radius=atom14_atom_radius,
        residue_index=residue_index,
        c_one_hot=c_one_hot,
        n_one_hot=n_one_hot,
        overlap_tolerance_soft=clash_overlap_tolerance,
        overlap_tolerance_hard=clash_overlap_tolerance,
        cys_sg_idx=cys_sg_idx
    )

    # Compute all within-residue violations (clashes,
    # bond length and angle violations).
    atom14_dists_lower_bound = P.Gather()(lower_bound, aatype, 0)
    atom14_dists_upper_bound = P.Gather()(upper_bound, aatype, 0)
    per_atom_loss_sum, per_atom_violations = within_residue_violations(
        atom14_pred_positions=atom14_pred_positions,
        atom14_atom_exists=atom14_atom_exists,
        atom14_dists_lower_bound=atom14_dists_lower_bound,
        atom14_dists_upper_bound=atom14_dists_upper_bound,
        tighten_bounds_for_loss=0.0,
        dists_mask_i=dists_mask_i)

    # Combine them to a single per-residue violation mask (used later for LDDT).
    per_residue_violations_mask = mnp.max(mnp.stack([per_residue_violation_mask, mnp.max(per_atom_clash_mask, axis=-1),
                                                     mnp.max(per_atom_violations, axis=-1)]), axis=0)
    bonds_c_n_loss_mean = c_n_loss_mean
    angles_ca_c_n_loss_mean = ca_c_n_loss_mean
    angles_c_n_ca_loss_mean = c_n_ca_loss_mean
    connections_per_residue_loss_sum = per_residue_loss_sum
    connections_per_residue_violation_mask = per_residue_violation_mask
    clashes_mean_loss = mean_loss
    clashes_per_atom_loss_sum = per_atom_loss_sum
    clashes_per_atom_clash_mask = per_atom_clash_mask
    per_atom_loss_sum = per_atom_loss_sum
    per_atom_violations = per_atom_violations
    total_per_residue_violations_mask = per_residue_violations_mask
    return (bonds_c_n_loss_mean, angles_ca_c_n_loss_mean, angles_c_n_ca_loss_mean, connections_per_residue_loss_sum,
            connections_per_residue_violation_mask, clashes_mean_loss, clashes_per_atom_loss_sum,
            clashes_per_atom_clash_mask, per_atom_loss_sum, per_atom_violations, total_per_residue_violations_mask)
