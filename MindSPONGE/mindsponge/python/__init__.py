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
"""MindSPONGE"""

import time


def _mindspore_version_check():
    """
       Do the MindSpore version check for MindSPONGE. If the
       MindSpore can not be imported, it will raise ImportError. If its
       version is not compatibale with current MindSponge verision,
       it will print a warning.

       Raise:
           ImportError: If the MindSpore can not be imported.
       """

    # pylint: disable=import-outside-toplevel
    try:
        import mindspore as ms
        from mindspore import log as logger
    except ImportError:
        raise ImportError("Can not find MindSpore in current environment. Please install "
                          "MindSpore before using MindSpore Mindsponge, by following "
                          "the instruction at https://www.mindspore.cn/install")

    ms_version = ms.__version__[:5]
    required_mindspore_version = '2.0.0'
    logger.info("Current Mindspore version is {}".format(ms_version))
    ms_version = list(map(int, ms_version.split('.')))
    required_mindspore = list(map(int, required_mindspore_version.split('.')))
    max_len = max(len(ms_version), len(required_mindspore))
    ms_version += [0] * (max_len - len(ms_version))
    required_mindspore += [0] * (max_len - len(required_mindspore))
    if ms_version < required_mindspore:
        logger.warning("Current version of MindSpore is not compatible with MindSPONGE. "
                       "Some functions might not work or even raise error. Please install MindSpore "
                       "version >= {} For more details about dependency setting, please check "
                       "the instructions at MindSpore official website https://www.mindspore.cn/install "
                       "or check the README.md at https://gitee.com/mindspore/mindscience"
                       .format(required_mindspore_version))
        warning_countdown = 3
        for i in range(warning_countdown, 0, -1):
            logger.warning(
                f"Please pay attention to the above warning, countdown: {i}")
            time.sleep(1)

_mindspore_version_check()

# pylint: disable=wrong-import-position
from .system import Molecule, Protein
from .potential import PotentialCell, ForceFieldBase, ForceField
from .optimizer import Updater, UpdaterMD, DynamicUpdater
from .core import Sponge, WithEnergyCell, SimulationCell, RunOneStepCell, AnalysisCell
from .function.units import GLOBAL_UNITS, set_global_units
from .function.units import set_global_length_unit, set_global_energy_unit
from .pipeline import PipeLine
