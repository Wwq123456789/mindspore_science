# Copyright 2021-2022 @ Shenzhen Bay Laboratory &
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
"""
Base function for yaml
"""

from itertools import permutations
import numpy as np
from numpy import ndarray
import yaml


def update_dict(origin_dict: dict, new_dict: dict) -> dict:
    """update complex dict"""
    if new_dict is None:
        return origin_dict
    dictionary = origin_dict.copy()
    origin_dict.update()
    for k, v in new_dict.items():
        if k in dictionary.keys() and isinstance(dictionary.get(k), dict) and isinstance(v, dict):
            dictionary[k] = update_dict(dictionary[k], v)
        else:
            dictionary[k] = v
    return dictionary


def write_yaml(filename: str, data: dict):
    """write yaml file"""
    with open(filename, 'w', encoding="utf-8") as file:
        yaml.dump(data, file, sort_keys=False)


def read_yaml(filename: str) -> dict:
    """read yaml file"""
    with open(filename, 'r', encoding="utf-8") as file:
        data = yaml.safe_load(file.read())
    return data


def get_bonded_types(atom_types: ndarray, symbol: str = '-'):
    """get the types of bonded terms including bond, angle and dihedral"""
    num_atoms = atom_types.shape[-1]

    if num_atoms == 1:
        return atom_types

    types = atom_types[..., 0]
    for i in range(1, num_atoms):
        types = np.char.add(types, symbol)
        types = np.char.add(types, atom_types[..., i])

    return types


def get_dihedral_types(atom_types: ndarray, symbol: str = '-'):
    """ The multi atom name constructor. """
    num_atoms = atom_types.shape[-1]

    if num_atoms == 1:
        return atom_types

    types = atom_types[..., 0]
    for i in range(1, num_atoms):
        types = np.char.add(types, symbol)
        types = np.char.add(types, atom_types[..., i])

    inverse_types = atom_types[..., -1]
    for i in range(1, num_atoms):
        inverse_types = np.char.add(inverse_types, symbol)
        inverse_types = np.char.add(inverse_types, atom_types[..., -1-i])

    return types, inverse_types


def get_improper_types(atom_types: ndarray, symbol: str = '-'):
    """ The multi atom name constructor. """
    num_atoms = atom_types.shape[-1]

    if num_atoms == 1:
        return atom_types

    permuation_types = ()
    orders = ()
    for combination in permutations(range(num_atoms)):
        types = atom_types[..., combination[0]]
        for i in range(1, num_atoms):
            types = np.char.add(types, symbol)
            types = np.char.add(types, atom_types[..., combination[i]])
        permuation_types += (types,)
        orders += (combination,)

    return permuation_types, orders
