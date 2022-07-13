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
'''r3 numpy'''

import collections
from typing import List

import numpy as np


# Init of vectors.
Vecs = collections.namedtuple('Vecs', ['x', 'y', 'z'])

# Init of rotations.
Rots = collections.namedtuple('Rots', ['xx', 'xy', 'xz',
                                       'yx', 'yy', 'yz',
                                       'zx', 'zy', 'zz'])
# Init of rigids.
Rigids = collections.namedtuple('Rigids', ['rot', 'trans'])


def squared_difference(x, y):
    """get suqare"""
    out = np.square(x - y)
    return out


def invert_rigids(r: Rigids) -> Rigids:
    """get invert rigids"""
    rotation_invert = invert_rots(r.rot)
    trans = rots_mul_vecs(rotation_invert, r.trans)
    trans_invert = Vecs(-trans.x, -trans.y, -trans.z)
    return Rigids(rotation_invert, trans_invert)


def invert_rots(inputs: Rots) -> Rots:
    """get invert of rotation"""
    return Rots(inputs.xx, inputs.yx, inputs.zx,
                inputs.xy, inputs.yy, inputs.zy,
                inputs.xz, inputs.yz, inputs.zz)


def rigids_from_3_points(point_on_neg_x_axis: Vecs,
                         origin: Vecs,
                         point_on_xy_plane: Vecs) -> Rigids:
    """construct rigids from inputs 3 point"""
    e0_unnorm = vecs_sub(origin, point_on_neg_x_axis)
    e1_unnorm = vecs_sub(point_on_xy_plane, origin)
    rotations = rots_from_two_vecs(
        e0_unnormalized=e0_unnorm,
        e1_unnormalized=e1_unnorm)

    return Rigids(rot=rotations, trans=origin)


def rigids_from_list(inputs: List[np.ndarray]) -> Rigids:
    """get rigids from inputs list"""
    assert len(inputs) == 12
    return Rigids(Rots(*(inputs[:9])), Vecs(*(inputs[9:])))


def rigids_from_tensor4x4(
        inputs: np.ndarray  # inputs shape (..., 4, 4)
) -> Rigids:
    """get rigids from input tensor"""
    assert inputs.shape[-1] == 4
    assert inputs.shape[-2] == 4
    return Rigids(
        Rots(inputs[..., 0, 0], inputs[..., 0, 1], inputs[..., 0, 2],
             inputs[..., 1, 0], inputs[..., 1, 1], inputs[..., 1, 2],
             inputs[..., 2, 0], inputs[..., 2, 1], inputs[..., 2, 2]),
        Vecs(inputs[..., 0, 3], inputs[..., 1, 3], inputs[..., 2, 3]))


def rigids_from_tensor_flat9(
        inputs: np.ndarray  # inputs shape (..., 9)
) -> Rigids:
    """get rigids from flat input tensor"""
    assert inputs.shape[-1] == 9
    e0 = Vecs(inputs[..., 0], inputs[..., 1], inputs[..., 2])
    e1 = Vecs(inputs[..., 3], inputs[..., 4], inputs[..., 5])
    trans = Vecs(inputs[..., 6], inputs[..., 7], inputs[..., 8])
    rots = rots_from_two_vecs(e0, e1)
    return Rigids(rot=rots, trans=trans)


def rigids_from_tensor_flat12(
        inputs: np.ndarray  # inputs shape (..., 12)
) -> Rigids:
    """get rigids from flat input tensor"""
    assert inputs.shape[-1] == 12
    out = np.moveaxis(inputs, -1, 0)
    return Rigids(Rots(*out[:9]), Vecs(*out[9:]))


def rigids_mul_rigids(x: Rigids, y: Rigids) -> Rigids:
    """get results of rigids x multiply rigids y"""
    return Rigids(
        rots_mul_rots(x.rot, y.rot),
        vecs_add(x.trans, rots_mul_vecs(x.rot, y.trans)))


def rigids_mul_rots(x: Rigids, y: Rots) -> Rigids:
    """get results rigids x multiply rigids y"""
    rigids = Rigids(rots_mul_rots(x.rot, y), x.trans)
    return rigids


def rigids_mul_vecs(x: Rigids, y: Vecs) -> Vecs:
    """get results rigids x multiply vecs y"""
    vecs_res = vecs_add(rots_mul_vecs(x.rot, y), x.trans)
    return vecs_res


def rigids_to_list(inputs: Rigids) -> List[np.ndarray]:
    """transfer rigids to list"""
    return list(inputs.rot) + list(inputs.trans)


def rigids_to_tensor_flat9(
        inputs: Rigids
) -> np.ndarray:  # outputs shape (..., 9)
    """transfer rigids to flat tensor"""
    return np.stack(
        [inputs.rot.xx, inputs.rot.yx, inputs.rot.zx, inputs.rot.xy, inputs.rot.yy, inputs.rot.zy]
        + list(inputs.trans), axis=-1)


def rigids_to_tensor_flat12(
        inputs: Rigids
) -> np.ndarray:  # outputs shape (..., 12)
    """transfer rigids to flat tensor"""
    return np.stack(list(inputs.rot) + list(inputs.trans), axis=-1)


def rots_from_tensor3x3(
        inputs: np.ndarray,  # inputs shape (..., 3, 3)
) -> Rots:
    """get rotation from inputs tensor"""
    assert inputs.shape[-1] == 3
    assert inputs.shape[-2] == 3
    return Rots(inputs[..., 0, 0], inputs[..., 0, 1], inputs[..., 0, 2],
                inputs[..., 1, 0], inputs[..., 1, 1], inputs[..., 1, 2],
                inputs[..., 2, 0], inputs[..., 2, 1], inputs[..., 2, 2])


def rots_from_two_vecs(e0_unnormalized: Vecs, e1_unnormalized: Vecs) -> Rots:
    """get rotation from two vectors"""
    e0 = vecs_robust_normalize(e0_unnormalized)

    vecs = vecs_dot_vecs(e1_unnormalized, e0)
    e1 = Vecs(e1_unnormalized.x - vecs * e0.x,
              e1_unnormalized.y - vecs * e0.y,
              e1_unnormalized.z - vecs * e0.z)
    e1 = vecs_robust_normalize(e1)
    e2 = vecs_cross_vecs(e0, e1)

    return Rots(e0.x, e1.x, e2.x, e0.y, e1.y, e2.y, e0.z, e1.z, e2.z)


def rots_mul_rots(x: Rots, y: Rots) -> Rots:
    """get result of rots x multiply rots y"""
    vecs0 = rots_mul_vecs(x, Vecs(y.xx, y.yx, y.zx))
    vecs1 = rots_mul_vecs(x, Vecs(y.xy, y.yy, y.zy))
    vecs2 = rots_mul_vecs(x, Vecs(y.xz, y.yz, y.zz))
    return Rots(vecs0.x, vecs1.x, vecs2.x, vecs0.y, vecs1.y, vecs2.y, vecs0.z, vecs1.z, vecs2.z)


def rots_mul_vecs(x: Rots, y: Vecs) -> Vecs:
    """get result of rots x multiply vecs y"""
    return Vecs(x.xx * y.x + x.xy * y.y + x.xz * y.z,
                x.yx * y.x + x.yy * y.y + x.yz * y.z,
                x.zx * y.x + x.zy * y.y + x.zz * y.z)


def vecs_add(a: Vecs, b: Vecs) -> Vecs:
    """get result of vecs1 add vecs2"""
    return Vecs(a.x + b.x, a.y + b.y, a.z + b.z)


def vecs_dot_vecs(a: Vecs, b: Vecs) -> np.ndarray:
    """get res of vecs1 dot vecs2"""
    return a.x * b.x + a.y * b.y + a.z * b.z


def vecs_cross_vecs(a: Vecs, b: Vecs) -> Vecs:
    """get result of vecs a cross multiply vecs b"""
    return Vecs(a.y * b.z - a.z * b.y,
                a.z * b.x - a.x * b.z,
                a.x * b.y - a.y * b.x)


def vecs_from_tensor(inputs: np.ndarray  # inputs shape (..., 3)
                     ) -> Vecs:
    """get vectors from input tensor"""
    num_components = inputs.shape[-1]
    assert num_components == 3
    return Vecs(inputs[..., 0], inputs[..., 1], inputs[..., 2])


def vecs_robust_normalize(inputs: Vecs, eps: float = 1e-8) -> Vecs:
    """get normalization vecs"""
    normalization = vecs_robust_norm(inputs, eps)
    return Vecs(inputs.x / normalization, inputs.y / normalization, inputs.z / normalization)


def vecs_robust_norm(inputs: Vecs, eps: float = 1e-8) -> np.ndarray:
    """get norm of vectors"""
    return np.sqrt(np.square(inputs.x) + np.square(inputs.y) + np.square(inputs.z) + eps)


def vecs_sub(a: Vecs, b: Vecs) -> Vecs:
    """get result of vectors a sub vectors b"""
    return Vecs(a.x - b.x, a.y - b.y, a.z - b.z)


def vecs_squared_distance(a: Vecs, b: Vecs) -> np.ndarray:
    """get square distance of vectors a and vectors b."""
    return (squared_difference(a.x, b.x) +
            squared_difference(a.y, b.y) +
            squared_difference(a.z, b.z))


def vecs_to_tensor(inputs: Vecs
                   ) -> np.ndarray:  # inputs shape(..., 3)
    """get tensor from vectors"""
    outputs = np.stack([inputs.x, inputs.y, inputs.z], axis=-1)
    return outputs
