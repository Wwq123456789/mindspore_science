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
utils module used for timing analysing and tmpdir generation eg.
"""

import contextlib
import tempfile
import shutil
import time

from typing import Optional, Sequence
from absl import logging


@contextlib.contextmanager
def tmpdir_manager(base_dir: Optional[str] = None):
    """Context manager that deletes a temporary directory on exit."""
    tmpdir = tempfile.mkdtemp(dir=base_dir)
    try:
        yield tmpdir
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@contextlib.contextmanager
def timing(msg: str):
    logging.info('Started %s', msg)
    tic = time.time()
    yield
    toc = time.time()
    logging.info('Finished %s in %.3f seconds', msg, toc - tic)


def to_a3m(sequences: Sequence[str]) -> str:
    """Converts sequences to an a3m file."""
    names = ['sequence %d' % i for i in range(1, len(sequences) + 1)]
    a3m = []
    for sequence, name in zip(sequences, names):
        a3m.append(u'>' + name + u'\n')
        a3m.append(sequence + u'\n')
    return ''.join(a3m)
