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
"""harmonic oscillator"""
from ..energy import EnergyCell

class Oscillator(EnergyCell):
    """A harmonic oscillator force function.
    Args:
        old_crd(Tensor): The B*N*D initial coordinates of atoms, assuming to be 0 force.
        k(Tensor,int): The coefficients of atoms with shift.
    Parameters:
        new_crd(Tensor): The B*N*D final coordinates of atoms with shift.
    Returns:
        force(Tensor): The B*N*D oscillator force for atoms.
    Examples:
        >>> from mindspore import Tensor
        >>> import numpy as np
        >>> import mindspore as ms
        >>> k = Tensor(np.random.random((B,N,1)))
        >>> crd = Tensor(np.random.random((1,2,3))*10, ms.float32)
        >>> shift = Tensor(np.random.random((1,2,3)), ms.float32)-0.5
        >>> print (shift)
        [[[-0.0624128   0.391773    0.46366274]
          [-0.11655849  0.29172504  0.0288949 ]]]
        >>> new_crd = crd+shift
        >>> energy = Oscillator(crd,k)(new_crd)
        >>> print (energy)
        0.15279999618549867
        >>> force = -grad(Oscillator(crd,k))(new_crd)[0]
        >>> print (force)
        [[[-0.25446433  0.06396891 -0.16010271]
          [-0.02066533 -0.04866482 -0.30438188]]]
    """

    def __init__(self,
                 old_crd,
                 k,
                 nonh_mask,
                 pbc=None,
                 unit_length=None,
                 unit_energy=None,):
        super().__init__(
            unit_length=unit_length,
            unit_energy=unit_energy,
            pbc=pbc,
        )
        self.old_crd = old_crd
        self.k = k
        self.nonh_mask = 1 - nonh_mask

    def calculate(self, new_crd, pbc_box=None):
        shift = new_crd - self.old_crd
        energy = 0.5 * self.k * shift ** 2 * self.nonh_mask
        return energy.sum()


class FollowedOscillator(EnergyCell):
    """A harmonic oscillator force function.
    Args:
        old_crd(Tensor): The B*N*D initial coordinates of atoms, assuming to be 0 force.
        k(Tensor,int): The coefficients of atoms with shift.
    Parameters:
        new_crd(Tensor): The B*N*D final coordinates of atoms with shift.
    Returns:
        force(Tensor): The B*N*D oscillator force for atoms.
    Examples:
        >>> from mindspore import Tensor
        >>> import numpy as np
        >>> import mindspore as ms
        >>> k = Tensor(np.random.random((B,N,1)))
        >>> crd = Tensor(np.random.random((1,2,3))*10, ms.float32)
        >>> shift = Tensor(np.random.random((1,2,3)), ms.float32)-0.5
        >>> print (shift)
        [[[-0.0624128   0.391773    0.46366274]
          [-0.11655849  0.29172504  0.0288949 ]]]
        >>> new_crd = crd+shift
        >>> energy = Oscillator(crd,k)(new_crd)
        >>> print (energy)
        0.15279999618549867
        >>> force = -grad(Oscillator(crd,k))(new_crd)[0]
        >>> print (force)
        [[[-0.25446433  0.06396891 -0.16010271]
          [-0.02066533 -0.04866482 -0.30438188]]]
    """

    def __init__(self,
                 k,
                 nonh_mask,
                 pbc=None,
                 unit_length=None,
                 unit_energy=None,):
        super().__init__(
            unit_length=unit_length,
            unit_energy=unit_energy,
            pbc=pbc,
        )
        self.k = k
        self.nonh_mask = 1 - nonh_mask

    def calculate(self, old_crd, new_crd, pbc_box=None):
        shift = new_crd - old_crd
        energy = 0.5 * self.k * shift ** 2 * self.nonh_mask
        return energy.sum()
