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
"""
Module used to load and parse charge information from files.
"""


def get_charge_from_file(file_name):
    """Function used to load charge from files generate from xponge.
    Args:
        file_name(str): The file path of charge file, absolute path is suggested.
    Returns:
        chargs(list): The builtin charge list with format float32.
    Example how to generate charge file:
        >>> import Xponge
        >>> import Xponge.forcefield.AMBER.ff14SB
        >>> Save_SPONGE_Input(ALA,'ALA')
        # The generated charge file name could be ALA_charge.txt
    """
    charges = []
    with open(file_name, 'r') as file:
        lines = file.readlines()
    nums = int(lines[0].strip())
    for line in lines[1:nums + 1]:
        charges.append(float(line.strip()))
    return charges
