# Copyright 2021-2023 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of MindSPONGE:
# MindSpore Simulation Package tOwards Next Generation molecular modelling.
#
# MindSPONGE is open-source software based on the AI-framework:
# MindSpore (https://www.mindspore.cn/)
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
"""Base energy cell"""

from typing import Union
import mindspore as ms
from mindspore import Tensor
from mindspore import ops
from mindspore.nn import Cell

from ...function import get_ms_array
from ...function.units import Units, Length, GLOBAL_UNITS


class EnergyCell(Cell):
    r"""Base class for energy terms.

        `EnergyCell` is usually used as a base class for individual energy terms in a classical force field.
        As the force field parameters usually has units, the units of the EnergyCell as an energy term
        should be the same as the units of the force field parameters, and not equal to the global units.

    Args:

        name (str):         Name of energy. Default: 'energy'

        length_unit (str):  Length unit. If None is given, it will be assigned with the global length unit.
                            Default: 'nm'

        energy_unit (str):  Energy unit. If None is given, it will be assigned with the global energy unit.
                            Default: 'kj/mol'

        use_pbc (bool):     Whether to use periodic boundary condition.

    Returns:

        energy (Tensor):    Tensor of shape `(B, 1)`. Data type is float.

    Supported Platforms:

        ``Ascend`` ``GPU``

    Symbols:

        B:  Batchsize, i.e. number of walkers in simulation

    """
    def __init__(self,
                 name: str = 'energy',
                 length_unit: str = 'nm',
                 energy_unit: str = 'kj/mol',
                 use_pbc: bool = None,
                 ):

        super().__init__()

        self._name = name

        self._use_pbc = use_pbc

        if length_unit is None:
            length_unit = GLOBAL_UNITS.length_unit
        if energy_unit is None:
            energy_unit = GLOBAL_UNITS.energy_unit
        self.units = Units(length_unit, energy_unit)

        self.input_unit_scale = 1
        self.cutoff = None
        self.identity = ops.Identity()

    @property
    def name(self) -> str:
        """name of energy"""
        return self._name

    @property
    def use_pbc(self) -> bool:
        """whether to use periodic boundary condition"""
        return self._use_pbc

    @property
    def length_unit(self) -> str:
        """length unit"""
        return self.units.length_unit

    @property
    def energy_unit(self) -> str:
        """energy unit"""
        return self.units.energy_unit

    def set_input_unit(self, length_unit: Union[str, Units, Length]):
        """set the length unit for the input coordinates"""
        if length_unit is None:
            self.input_unit_scale = 1
        elif isinstance(length_unit, (str, Units, float)):
            self.input_unit_scale = Tensor(
                self.units.convert_length_from(length_unit), ms.float32)
        else:
            raise TypeError(f'Unsupported type of `length_unit`: {type(length_unit)}')

        return self

    def set_cutoff(self, cutoff: float, unit: str = None):
        """set cutoff distances"""
        if cutoff is None:
            self.cutoff = None
        else:
            cutoff = get_ms_array(cutoff, ms.float32)
            self.cutoff = self.units.length(cutoff, unit)
        return self

    def set_pbc(self, use_pbc: bool):
        """set whether to use periodic boundary condition."""
        self._use_pbc = use_pbc
        return self

    def convert_energy_from(self, unit: str) -> float:
        """convert energy from outside unit to inside unit"""
        return self.units.convert_energy_from(unit)

    def convert_energy_to(self, unit: str) -> float:
        """convert energy from inside unit to outside unit"""
        return self.units.convert_energy_to(unit)

    def construct(self,
                  coordinate: Tensor,
                  neighbour_index: Tensor = None,
                  neighbour_mask: Tensor = None,
                  neighbour_coord: Tensor = None,
                  neighbour_distance: Tensor = None,
                  pbc_box: Tensor = None
                  ):
        r"""Calculate energy term.

        Args:
            coordinate (Tensor):            Tensor of shape (B, A, D). Data type is float.
                                            Position coordinate of atoms in system
            neighbour_index (Tensor):       Tensor of shape (B, A, N). Data type is int.
                                            Index of neighbour atoms. Default: None
            neighbour_mask (Tensor):        Tensor of shape (B, A, N). Data type is bool.
                                            Mask for neighbour index. Default: None
            neighbour_coord (Tensor):       Tensor of shape (B, A, N). Data type is bool.
                                            Position coorindates of neighbour atoms. Default: None
            neighbour_distance (Tensor):    Tensor of shape (B, A, N). Data type is float.
                                            Distance between neighbours atoms. Default: None
            inv_neigh_dis (Tensor):         Tensor of shape (B, A, N). Data type is float.
                                            Reciprocal of distances. Default: None
            pbc_box (Tensor):               Tensor of shape (B, D). Data type is float.
                                            Tensor of PBC box. Default: None

        Returns:
            energy (Tensor):    Tensor of shape (B, 1). Data type is float.

        Symbols:
            B:  Batchsize, i.e. number of walkers in simulation
            A:  Number of atoms.
            D:  Spatial dimension of the simulation system. Usually is 3.

        """
        raise NotImplementedError


class NonbondEnergy(EnergyCell):
    r"""Base cell for non-bonded energy terms.

    Args:

        name (str):             Name of energy.

        cutoff (Union[float, Length, Tensor]):  cutoff distance. Default: None

        length_unit (str):      Length unit. If None is given, it will be assigned with the global length unit.
                                Default: 'nm'

        energy_unit (str):      Energy unit. If None is given, it will be assigned with the global energy unit.
                                Default: 'kj/mol'

        use_pbc (bool):         Whether to use periodic boundary condition. Default: None

    """
    def __init__(self,
                 name: str,
                 cutoff: Union[float, Length, Tensor] = None,
                 length_unit: str = 'nm',
                 energy_unit: str = 'kj/mol',
                 use_pbc: bool = None,
                 ):

        super().__init__(
            name=name,
            length_unit=length_unit,
            energy_unit=energy_unit,
            use_pbc=use_pbc,
        )

        if isinstance(cutoff, Length):
            cutoff = cutoff(self.units)

        self.cutoff = None
        if cutoff is not None:
            self.cutoff = Tensor(cutoff, ms.float32)

    def construct(self,
                  coordinate: Tensor,
                  neighbour_index: Tensor = None,
                  neighbour_mask: Tensor = None,
                  neighbour_coord: Tensor = None,
                  neighbour_distance: Tensor = None,
                  pbc_box: Tensor = None
                  ):
        r"""Calculate energy term.

        Args:
            coordinate (Tensor):            Tensor of shape (B, A, D). Data type is float.
                                            Position coordinate of atoms in system
            neighbour_index (Tensor):       Tensor of shape (B, A, N). Data type is int.
                                            Index of neighbour atoms. Default: None
            neighbour_mask (Tensor):        Tensor of shape (B, A, N). Data type is bool.
                                            Mask for neighbour index. Default: None
            neighbour_coord (Tensor):       Tensor of shape (B, A, N). Data type is bool.
                                            Position coorindates of neighbour atoms. Default: None
            neighbour_distance (Tensor):    Tensor of shape (B, A, N). Data type is float.
                                            Distance between neighbours atoms. Default: None
            inv_neigh_dis (Tensor):         Tensor of shape (B, A, N). Data type is float.
                                            Reciprocal of distances. Default: None
            pbc_box (Tensor):               Tensor of shape (B, D). Data type is float.
                                            Tensor of PBC box. Default: None

        Returns:
            energy (Tensor):    Tensor of shape (B, 1). Data type is float.

        Symbols:
            B:  Batchsize, i.e. number of walkers in simulation
            A:  Number of atoms.
            D:  Spatial dimension of the simulation system. Usually is 3.

        """
        raise NotImplementedError
