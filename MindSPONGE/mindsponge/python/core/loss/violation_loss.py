"""violation loss calculation."""

import numpy as np
import mindspore.numpy as mnp
import mindspore.nn as nn
import mindspore as ms
from mindspore.ops import operations as P
from mindspore import Tensor

from ...common.utils import make_atom14_positions

from ...common import residue_constants, protein
from ..metrics.structure_violations import find_structural_violations


VIOLATION_TOLERANCE_ACTOR = 12.0
CLASH_OVERLAP_TOLERANCE = 1.5
C_ONE_HOT = nn.OneHot(depth=14)(Tensor(2, ms.int32))
N_ONE_HOT = nn.OneHot(depth=14)(Tensor(0, ms.int32))
DISTS_MASK_I = mnp.eye(14, 14)
CYS_SG_IDX = Tensor(5, ms.int32)
ATOMTYPE_RADIUS = Tensor(np.array(
    [1.55, 1.7, 1.7, 1.7, 1.52, 1.7, 1.7, 1.7, 1.52, 1.52, 1.8, 1.7, 1.7, 1.7, 1.55, 1.55,
     1.52, 1.52, 1.8, 1.7, 1.7, 1.7, 1.7, 1.55, 1.55, 1.55, 1.52, 1.52, 1.7, 1.55, 1.55,
     1.52, 1.7, 1.7, 1.7, 1.55, 1.52]), ms.float32)
LOWER_BOUND, UPPER_BOUND, RESTYPE_ATOM14_BOUND_STDDEV = \
    residue_constants.make_atom14_dists_bounds(overlap_tolerance=1.5, bond_length_tolerance_factor=12.0)
LOWER_BOUND = Tensor(LOWER_BOUND, ms.float32)
UPPER_BOUND = Tensor(UPPER_BOUND, ms.float32)
RESTYPE_ATOM14_BOUND_STDDEV = Tensor(RESTYPE_ATOM14_BOUND_STDDEV, ms.float32)

def get_violation_loss(pdb_path):
    """calculate violations loss for a given pdb
    Args:
        pdb_path: absolute path of pdb

    Returns:
        pdb's violations loss
    """

    with open(pdb_path, 'r') as f:
        prot_pdb = protein.from_pdb_string(f.read())
    aatype = prot_pdb.aatype
    atom37_positions = prot_pdb.atom_positions.astype(np.float32)
    atom37_mask = prot_pdb.atom_mask.astype(np.float32)

    # get ground truth of atom14
    features = {'aatype': aatype,
                'all_atom_positions': atom37_positions,
                'all_atom_mask': atom37_mask}
    features = make_atom14_positions(features)

    features["residue_index"] = prot_pdb.residue_index

    atom14_atom_exists = Tensor(features["atom14_gt_exists"]).astype(ms.float32)
    residue_index = Tensor(features["residue_index"]).astype(ms.float32)
    residx_atom14_to_atom37 = Tensor(features["residx_atom14_to_atom37"]).astype(ms.int32)
    atom14_pred_positions = Tensor(features["atom14_gt_positions"]).astype(ms.float32)
    aatype = Tensor(aatype).astype(ms.int32)

    (bonds_c_n_loss_mean, angles_ca_c_n_loss_mean, angles_c_n_ca_loss_mean, _, _, _, clashes_per_atom_loss_sum,
     _, per_atom_loss_sum, _, _) = \
        find_structural_violations(atom14_atom_exists, residue_index, aatype, residx_atom14_to_atom37,
                                   atom14_pred_positions, VIOLATION_TOLERANCE_ACTOR,
                                   CLASH_OVERLAP_TOLERANCE, LOWER_BOUND, UPPER_BOUND, ATOMTYPE_RADIUS,
                                   C_ONE_HOT, N_ONE_HOT, DISTS_MASK_I, CYS_SG_IDX)
    num_atoms = P.ReduceSum()(atom14_atom_exists.astype(ms.float32))
    structure_violation_loss = bonds_c_n_loss_mean + angles_ca_c_n_loss_mean + angles_c_n_ca_loss_mean +\
                               P.ReduceSum()(clashes_per_atom_loss_sum + per_atom_loss_sum) / (1e-6 + num_atoms)

    return structure_violation_loss
