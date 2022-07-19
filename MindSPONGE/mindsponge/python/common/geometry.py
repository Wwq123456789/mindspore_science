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
"""Geometry"""
import numpy as np
import mindspore.numpy as mnp
from mindspore import Tensor
from mindspore.ops import composite as C, operations as P

QUAT_MULTIPLY = np.zeros((4, 4, 4), dtype=np.float32)
QUAT_MULTIPLY[:, :, 0] = [[1, 0, 0, 0],
                          [0, -1, 0, 0],
                          [0, 0, -1, 0],
                          [0, 0, 0, -1]]

QUAT_MULTIPLY[:, :, 1] = [[0, 1, 0, 0],
                          [1, 0, 0, 0],
                          [0, 0, 0, 1],
                          [0, 0, -1, 0]]

QUAT_MULTIPLY[:, :, 2] = [[0, 0, 1, 0],
                          [0, 0, 0, -1],
                          [1, 0, 0, 0],
                          [0, 1, 0, 0]]

QUAT_MULTIPLY[:, :, 3] = [[0, 0, 0, 1],
                          [0, 0, 1, 0],
                          [0, -1, 0, 0],
                          [1, 0, 0, 0]]

QUAT_MULTIPLY_BY_VEC = Tensor(QUAT_MULTIPLY[:, 1:, :])

QUAT_TO_ROT = np.zeros((4, 4, 3, 3), dtype=np.float32)

QUAT_TO_ROT[0, 0] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # rr
QUAT_TO_ROT[1, 1] = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]  # ii
QUAT_TO_ROT[2, 2] = [[-1, 0, 0], [0, 1, 0], [0, 0, -1]]  # jj
QUAT_TO_ROT[3, 3] = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]  # kk

QUAT_TO_ROT[1, 2] = [[0, 2, 0], [2, 0, 0], [0, 0, 0]]  # ij
QUAT_TO_ROT[1, 3] = [[0, 0, 2], [0, 0, 0], [2, 0, 0]]  # ik
QUAT_TO_ROT[2, 3] = [[0, 0, 0], [0, 0, 2], [0, 2, 0]]  # jk

QUAT_TO_ROT[0, 1] = [[0, 0, 0], [0, 0, -2], [0, 2, 0]]  # ir
QUAT_TO_ROT[0, 2] = [[0, 0, 2], [0, 0, 0], [-2, 0, 0]]  # jr
QUAT_TO_ROT[0, 3] = [[0, -2, 0], [2, 0, 0], [0, 0, 0]]  # kr

QUAT_TO_ROT = Tensor(QUAT_TO_ROT)

squeeze = C.MultitypeFuncGraph('squeeze')


@squeeze.register("Tensor")
def squeeze_tensor(x):
    return P.Squeeze()(x)


expand_dim = C.MultitypeFuncGraph('expand_dim')


@expand_dim.register("Tensor")
def expand_dim_tensor(x):
    return P.ExpandDims()(x, -1)


minus = C.MultitypeFuncGraph('minus')


@minus.register("Tensor", "Tensor")
def minus_tensor(x, y):
    return x - y


add = C.MultitypeFuncGraph('add')


@add.register("Tensor", "Tensor")
def add_tensor(x, y):
    return x + y


def apply_rot_to_vec(rot, vec):
    """apply rot to vec"""
    rotated_vec = (rot[0] * vec[0] + rot[1] * vec[1] + rot[2] * vec[2],
                   rot[3] * vec[0] + rot[4] * vec[1] + rot[5] * vec[2],
                   rot[6] * vec[0] + rot[7] * vec[1] + rot[8] * vec[2])
    return rotated_vec


def apply_inverse_rot_to_vec(rot, vec):
    """apply inverse rot to vec"""
    rotated_vec = (rot[0] * vec[0] + rot[3] * vec[1] + rot[6] * vec[2],
                   rot[1] * vec[0] + rot[4] * vec[1] + rot[7] * vec[2],
                   rot[2] * vec[0] + rot[5] * vec[1] + rot[8] * vec[2])
    return rotated_vec


def multiply(a, b):
    """multiply"""
    c = (a[0] * b[0] + a[1] * b[3] + a[2] * b[6],
         a[0] * b[1] + a[1] * b[4] + a[2] * b[7],
         a[0] * b[2] + a[1] * b[5] + a[2] * b[8],
         a[3] * b[0] + a[4] * b[3] + a[5] * b[6],
         a[3] * b[1] + a[4] * b[4] + a[5] * b[7],
         a[3] * b[2] + a[4] * b[5] + a[5] * b[8],
         a[6] * b[0] + a[7] * b[3] + a[8] * b[6],
         a[6] * b[1] + a[7] * b[4] + a[8] * b[7],
         a[6] * b[2] + a[7] * b[5] + a[8] * b[8])
    return c


def make_transform_from_reference(point_a, point_b, point_c):
    '''construct rotation and translation from given points
    calculate the rotation matrix and translation meets:
    a) point_b is the original point
    b) point_c is on the x_axis
    c) the plane a-b-c is on the x-y plane
    Return:
    rotation: Tuple [xx, xy, xz, yx, yy, yz, zx, zy, zz]
    translation: Tuple [x, y, z]
    '''

    # step 1 : shift the crd system by -point_b (point_b is the origin)
    translation = -point_b
    point_c = point_c + translation
    point_a = point_a + translation
    # step 2: rotate the crd system around z-axis to put point_c on x-z plane
    c_x, c_y, c_z = P.Split(-1, 3)(point_c)
    c_x = P.Squeeze()(c_x)
    c_y = P.Squeeze()(c_y)
    c_z = P.Squeeze()(c_z)
    sin_c1 = -c_y / mnp.sqrt(1e-20 + c_x ** 2 + c_y ** 2)
    cos_c1 = c_x / mnp.sqrt(1e-20 + c_x ** 2 + c_y ** 2)
    zeros = mnp.zeros_like(sin_c1)
    ones = mnp.ones_like(sin_c1)
    c1_rot_matrix = (cos_c1, -sin_c1, zeros,
                     sin_c1, cos_c1, zeros,
                     zeros, zeros, ones)
    # step 2 : rotate the crd system around y_axis to put point_c on x-axis
    sin_c2 = c_z / mnp.sqrt(1e-20 + c_x ** 2 + c_y ** 2 + c_z ** 2)
    cos_c2 = mnp.sqrt(c_x ** 2 + c_y ** 2) / mnp.sqrt(1e-20 + c_x ** 2 + c_y ** 2 + c_z ** 2)
    c2_rot_matrix = (cos_c2, zeros, sin_c2,
                     zeros, ones, zeros,
                     -sin_c2, zeros, cos_c2)
    c_rot_matrix = multiply(c2_rot_matrix, c1_rot_matrix)
    # step 3: rotate the crd system in y-z plane to put point_a in x-y plane
    vec_a_x, vec_a_y, vec_a_z = P.Split(-1, 3)(point_a)
    vec_a_x = P.Squeeze()(vec_a_x)
    vec_a_y = P.Squeeze()(vec_a_y)
    vec_a_z = P.Squeeze()(vec_a_z)
    vec_a = (vec_a_x, vec_a_y, vec_a_z)
    _, rotated_a_y, rotated_a_z = apply_rot_to_vec(c_rot_matrix, vec_a)

    sin_n = -rotated_a_z / mnp.sqrt(1e-20 + rotated_a_y ** 2 + rotated_a_z ** 2)
    cos_n = rotated_a_y / mnp.sqrt(1e-20 + rotated_a_y ** 2 + rotated_a_z ** 2)
    a_rot_matrix = (ones, zeros, zeros,
                    zeros, cos_n, -sin_n,
                    zeros, sin_n, cos_n)
    rotation_matrix = multiply(a_rot_matrix, c_rot_matrix)
    translation = point_b
    trans_x, trans_y, trans_z = P.Split(-1, 3)(translation)
    trans_x = P.Squeeze()(trans_x)
    trans_y = P.Squeeze()(trans_y)
    trans_z = P.Squeeze()(trans_z)
    translation = (trans_x, trans_y, trans_z)
    return rotation_matrix, translation


def rot_to_quat(rot, stack=False):
    if stack:
        rot = P.Reshape()(rot, P.Shape()(rot)[:-2] + (9,))
        rot = C.Map()(squeeze, P.Split(-1, 9)(rot))
    xx, xy, xz, yx, yy, yz, zx, zy, zz = rot
    quaternion = (1. / 3.) * mnp.stack((xx + yy + zz, zy - yz, xz - zx, yx - xy), axis=-1)
    return quaternion


def quat_affine(quaternion, translation, rotation=None, normalize=True, unstack_inputs=False):
    """create quat affine representations"""
    if unstack_inputs:
        if rotation is not None:
            rotation = P.Reshape()(rotation, P.Shape()(rotation)[:-2] + (9,))
            rotation = P.Split(-1, 9)(rotation)
            rotation = (P.Squeeze()(rotation[0]), P.Squeeze()(rotation[1]), P.Squeeze()(rotation[2]),
                        P.Squeeze()(rotation[3]), P.Squeeze()(rotation[4]), P.Squeeze()(rotation[5]),
                        P.Squeeze()(rotation[6]), P.Squeeze()(rotation[7]), P.Squeeze()(rotation[8]))
        translation_x, translation_y, translation_z = P.Split(-1, 3)(translation)
        translation_x = P.Squeeze()(translation_x)
        translation_y = P.Squeeze()(translation_y)
        translation_z = P.Squeeze()(translation_z)
        translation = (translation_x, translation_y, translation_z)
    if normalize and quaternion is not None:
        quaternion = quaternion / mnp.norm(quaternion, axis=-1, keepdims=True)
    if rotation is None:
        rotation = quat_to_rot(quaternion)
    return quaternion, rotation, translation


def quat_to_rot(normalized_quat):
    """Convert a normalized quaternion to a rotation matrix."""
    rot_tensor = mnp.sum(mnp.reshape(QUAT_TO_ROT, (4, 4, 9)) * normalized_quat[..., :, None, None] *
                         normalized_quat[..., None, :, None], axis=(-3, -2))
    rot_tensor = P.Split(-1, 9)(rot_tensor)
    rot_tensor = (P.Squeeze()(rot_tensor[0]), P.Squeeze()(rot_tensor[1]), P.Squeeze()(rot_tensor[2]),
                  P.Squeeze()(rot_tensor[3]), P.Squeeze()(rot_tensor[4]), P.Squeeze()(rot_tensor[5]),
                  P.Squeeze()(rot_tensor[6]), P.Squeeze()(rot_tensor[7]), P.Squeeze()(rot_tensor[8]))
    return rot_tensor


def generate_new_affine(num_residues):
    quaternion = mnp.tile(mnp.reshape(mnp.asarray([1., 0., 0., 0.]), [1, 4]), [num_residues, 1])
    translation = mnp.zeros([num_residues, 3])
    return quat_affine(quaternion, translation, unstack_inputs=True)


def invert_point(transformed_point, rotation, translation, extra_dims=0, stack=False):
    """invert_point"""
    if stack:
        rotation = P.Reshape()(rotation, P.Shape()(rotation)[:-2] + (9,))
        rotation = P.Split(-1, 9)(rotation)
        rotation = (P.Squeeze()(rotation[0]), P.Squeeze()(rotation[1]), P.Squeeze()(rotation[2]),
                    P.Squeeze()(rotation[3]), P.Squeeze()(rotation[4]), P.Squeeze()(rotation[5]),
                    P.Squeeze()(rotation[6]), P.Squeeze()(rotation[7]), P.Squeeze()(rotation[8]))
        translation_x, translation_y, translation_z = P.Split(-1, 3)(translation)
        translation_x = P.Squeeze()(translation_x)
        translation_y = P.Squeeze()(translation_y)
        translation_z = P.Squeeze()(translation_z)
        translation = (translation_x, translation_y, translation_z)
    for _ in range(extra_dims):
        rotation = (P.ExpandDims()(rotation[0], -1), P.ExpandDims()(rotation[1], -1), P.ExpandDims()(rotation[2], -1),
                    P.ExpandDims()(rotation[3], -1), P.ExpandDims()(rotation[4], -1), P.ExpandDims()(rotation[5], -1),
                    P.ExpandDims()(rotation[6], -1), P.ExpandDims()(rotation[7], -1), P.ExpandDims()(rotation[8], -1))
        translation = (P.ExpandDims()(translation[0], -1), P.ExpandDims()(translation[1], -1),
                       P.ExpandDims()(translation[2], -1),)
    rot_point = (transformed_point[0] - translation[0],
                 transformed_point[1] - translation[1],
                 transformed_point[2] - translation[2],)
    return apply_inverse_rot_to_vec(rotation, rot_point)


def quat_multiply_by_vec(quat, vec):
    """Multiply a quaternion by a pure-vector quaternion."""

    return mnp.sum(QUAT_MULTIPLY_BY_VEC * quat[..., :, None, None] * vec[..., None, :, None],
                   axis=(-3, -2))


def pre_compose(quaternion, rotation, translation, update):
    """Return a new QuatAffine which applies the transformation update first.

    Args:
    update: Length-6 vector. 3-vector of x, y, and z such that the quaternion
    update is (1, x, y, z) and zero for the 3-vector is the identity
    quaternion. 3-vector for translation concatenated.

    Returns:
    quaternion: [..., 4]
    rotation: Tuple [xx, xy, xz, yx, yy, yz, zx, zy, zz]
    translation: Tuple [x, y, z]
    """

    vector_quaternion_update, x, y, z = mnp.split(update, [3, 4, 5], axis=-1)
    trans_update = [mnp.squeeze(x, axis=-1), mnp.squeeze(y, axis=-1), mnp.squeeze(z, axis=-1)]
    new_quaternion = (quaternion + quat_multiply_by_vec(quaternion, vector_quaternion_update))
    rotated_trans_update = apply_rot_to_vec(rotation, trans_update)
    new_translation = [translation[0] + rotated_trans_update[0],
                       translation[1] + rotated_trans_update[1],
                       translation[2] + rotated_trans_update[2]]
    return quat_affine(new_quaternion, new_translation)


def scale_translation(translation, position_scale):
    """Return a new quat affine with a different scale for translation."""
    scaled = [translation[0] * position_scale,
              translation[1] * position_scale,
              translation[2] * position_scale,]
    return scaled


def to_tensor(quaternion, translation):
    translation = (P.ExpandDims()(translation[0], -1), P.ExpandDims()(translation[1], -1),
                   P.ExpandDims()(translation[2], -1),)
    return mnp.concatenate((quaternion,) + translation, axis=-1)


def from_tensor(tensor, normalize=False):
    quaternion, tx, ty, tz = mnp.split(tensor, [4, 5, 6], axis=-1)
    translation = (P.Squeeze()(tx), P.Squeeze()(ty), P.Squeeze()(tz))
    return quat_affine(quaternion, translation, normalize=normalize)


def vec_to_tensor(v):
    """vec_to_tensor"""
    return mnp.stack(v, axis=-1)


def apply_to_point(rotation, translation, point, extra_dims=0):
    """apply to point func"""
    for _ in range(extra_dims):
        rotation = (P.ExpandDims()(rotation[0], -1), P.ExpandDims()(rotation[1], -1), P.ExpandDims()(rotation[2], -1),
                    P.ExpandDims()(rotation[3], -1), P.ExpandDims()(rotation[4], -1), P.ExpandDims()(rotation[5], -1),
                    P.ExpandDims()(rotation[6], -1), P.ExpandDims()(rotation[7], -1), P.ExpandDims()(rotation[8], -1))
        translation = (P.ExpandDims()(translation[0], -1), P.ExpandDims()(translation[1], -1),
                       P.ExpandDims()(translation[2], -1),)
    rot_point = apply_rot_to_vec(rotation, point)
    result = (rot_point[0] + translation[0],
              rot_point[1] + translation[1],
              rot_point[2] + translation[2],)
    return result
