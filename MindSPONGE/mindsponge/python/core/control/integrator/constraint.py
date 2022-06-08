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
"""Module Usage
1. Define bonds of system `bonds` with shape (K,2);
2. Define Mass of atoms `M` in the system with shape (N,);
3. Given old coordinates 'crd' with shape (N,3);
4. Given new coordinates after dynamics update 'new_crd' with shape (N,3);
5. Get constraint coordinates `ccrd` by running `ccrd = Lincs(bonds, M, crd)(new_crd)`.
"""
import mindspore as ms
from mindspore import nn, Tensor, ops
from mindspore import numpy as msnp


class Bi(nn.Cell):
    """Module for calculating B matrix whose dimensions are: K.
    Args:
        crd(Tensor,N*3): The old coordinates of the system.
        bonds(Tensor,K*1): All the bonds need to optimize.
    Input:
        crd(Tensor,N*3): The new coordinates of the system.
    Return:
        A tensor(K*1) about all the bond distance.
    """

    def __init__(self, crd, bonds):
        super(Bi, self).__init__()
        self.crd = Tensor(crd, ms.float32)
        self.bonds = Tensor(bonds, ms.int32)
        self.norm = nn.Norm(-1)
        mask = np.zeros((crd.shape[0], bonds.shape[-2], crd.shape[-2]))
        np.put_along_axis(mask, bonds.asnumpy()[:, :, None, 0], 1, axis=-1)
        self.mask0 = Tensor(mask, ms.int32)[:, :, :, None]
        mask = np.zeros((crd.shape[0], bonds.shape[-2], crd.shape[-2]))
        np.put_along_axis(mask, bonds.asnumpy()[:, :, None, 1], 1, axis=-1)
        self.mask1 = Tensor(mask, ms.int32)[:, :, :, None]
        self.abs = ops.Abs()

    def construct(self, crd):
        norm1 = self.norm((self.mask0 * crd).sum(-2) - (self.mask1 * crd).sum(-2))
        norm2 = self.norm((self.mask0 * self.crd).sum(-2) - (self.mask1 * self.crd).sum(-2))
        return norm1 - norm2


class pBi(Bi):
    """Module for calculating the differentiation of B matrix whose dimensions are: K*N*D.
    Args:
        crd(Tensor,N*3): The old coordinates of the system.
        bonds(Tensor,K*1): All the bonds need to optimize.
    Input:
        crd(Tensor,N*3): The new coordinates of the system.
    Return:
        A tensor(K*N*3) about all the grad value of bond distance.
    """

    def __init__(self, crd, bonds):
        super(pBi, self).__init__(crd, bonds)
        self.grad = ops.grad
        shape = (crd.shape[0], bonds.shape[-2], crd.shape[-2], crd.shape[-1])
        self.broadcast = ops.BroadcastTo(shape)
        self.net = Bi(self.broadcast(crd[:, None, :, :]), bonds)

    def construct(self, crd):
        return self.grad(self.net)(self.broadcast(crd[:, None, :, :]))


class Lincs(nn.Cell):
    """Use LINCS constraint to limit the bond length.
    Args:
        bonds(Tensor,K*1): All the bonds need to optimize.
        Mii(Tensor,K*K): All the inverse mass of atoms in the diagonal elements.
        crd(Tensor,N*3): The old coordinates of the system.
    Input:
        new_crd(Tensor,N*3): The new coordinates of the system.
    Return:
        A tensor(K*1) about all the bond distance.
    """

    def __init__(self, bonds, invM, crd):
        super(Lincs, self).__init__()
        self.bonds = Tensor(bonds, ms.int32)
        self.crd = Tensor(crd, ms.float32)
        iinvM = msnp.identity(invM.shape[-1])
        identity_invM = ops.BroadcastTo((bonds.shape[0],) + iinvM.shape)(iinvM) * invM[:, None, :]
        self.Mii = identity_invM
        self.B_value = pBi(crd, bonds)
        shape = (crd.shape[0], bonds.shape[-2], crd.shape[-2], crd.shape[-1])
        self.broadcast = ops.BroadcastTo(shape)
        self.Bi = Bi(self.broadcast(crd[:, None, :, :]), bonds)
        self.norm = nn.Norm(-1)
        self.inv = ops.MatrixInverse(adjoint=False)
        self.squeeze = ops.Squeeze()
        self.einsum0 = ops.Einsum('ijk,ilkm->iljm')
        self.einsum1 = ops.Einsum('ijkl,imkl->ijm')
        self.einsum2 = ops.Einsum('ijkl,ikl->ij')
        self.einsum3 = ops.Einsum('ijk,ik->ij')
        self.einsum4 = ops.Einsum('ijkl,ij->ikl')
        self.einsum5 = ops.Einsum('ijk,ikl->ijl')
        self.norm = nn.Norm(-1)
        mask = np.zeros((crd.shape[0], bonds.shape[-2], crd.shape[-2]))
        np.put_along_axis(mask, bonds.asnumpy()[:, :, None, 0], 1, axis=-1)
        self.mask0 = Tensor(mask, ms.int32)[:, :, :, None]
        mask = np.zeros((crd.shape[0], bonds.shape[-2], crd.shape[-2]))
        np.put_along_axis(mask, bonds.asnumpy()[:, :, None, 1], 1, axis=-1)
        self.mask1 = Tensor(mask, ms.int32)[:, :, :, None]

    def construct(self, new_crd):
        B_value = self.B_value(new_crd)
        ccrd = new_crd.copy()
        tmp0 = self.einsum0((self.Mii, B_value))
        tmp1 = self.einsum1((B_value, tmp0))
        tmp2 = self.inv(tmp1)
        di = self.norm((self.mask0 * self.broadcast(self.crd[:, None, :, :])).sum(-2) - \
                       (self.mask1 * self.broadcast(self.crd[:, None, :, :])).sum(-2))
        tmp3 = self.einsum2((B_value, new_crd)) - di
        tmp4 = self.einsum3((tmp2, tmp3))
        tmp5 = self.einsum4((B_value, tmp4))
        tmp6 = self.einsum5((self.Mii, tmp5))
        ccrd -= tmp6
        return ccrd
