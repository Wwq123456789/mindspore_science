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
"""information"""

from mindspore.train.callback import Callback, RunContext
from ...core.space.system import SystemCell


class RunInfo(Callback):
    """run information"""

    def __init__(
            self,
            system: SystemCell,
            get_vloss: int = False,
            atom14_atom_exists=None,
            residue_index=None,
            residx_atom14_to_atom37=None,
            aatype=None,
            crd_mapping_masks=None,
            crd_mapping_ids=None,
            nonh_mask=None,
            print_interval=100,
    ):
        super().__init__()

        self.potential = None
        self.kinetic = None
        self.temperature = None
        self.tot_energy = None
        self.crd = None

        self.get_vloss = get_vloss
        self.atom14_atom_exists = atom14_atom_exists
        self.residue_index = residue_index
        self.residx_atom14_to_atom37 = residx_atom14_to_atom37
        self.aatype = aatype
        self.crd_mapping_masks = crd_mapping_masks
        self.crd_mapping_ids = crd_mapping_ids
        self.nonh_mask = nonh_mask
        self.print_interval = print_interval

    def __enter__(self):
        """Return the enter target."""
        return self

    def __exit__(self, *err):
        """Release resources here if have any."""

    def step_begin(self, run_context: RunContext):
        """
        Called before each step beginning.
        Args:
            run_context (RunContext): Include some information of the model.
        """
        cb_params = run_context.original_args()

        self.crd = cb_params.coordinates[0].squeeze().asnumpy()
        self.kinetic = cb_params.kinetic.squeeze().asnumpy()
        self.temperature = cb_params.temperature.squeeze().asnumpy()

    def begin(self, run_context: RunContext):
        """
        Called once before the network executing.
        Args:
            run_context (RunContext): Include some information of the model.
        """
        cb_params = run_context.original_args()
        print('>>>>>>>>>>>>>>>>>>>> Initialized <<<<<<<<<<<<<<<<<<<<')
        self.kinetic = cb_params.kinetic.squeeze().asnumpy()
        self.temperature = cb_params.temperature.squeeze().asnumpy()

    def step_end(self, run_context: RunContext):
        """
        Called after each step finished.
        Args:
            run_context (RunContext): Include some information of the model.
        """

        cb_params = run_context.original_args()
        step = cb_params.cur_step
        self.potential = cb_params.energy.squeeze().asnumpy()
        self.tot_energy = self.potential + self.kinetic

        info = 'Step: ' + str(step + 1) + ', '
        info += 'E_pot: ' + str(self.potential) + ', '
        info += 'E_kin: ' + str(self.kinetic) + ', '
        info += 'E_tot: ' + str(self.tot_energy) + ', '
        info += 'Temperature: ' + str(self.temperature) + ', '

        if (step + 1) % self.print_interval == 0:
            print(info)

    def end(self, run_context: RunContext):
        """
        Called once after network training.
        Args:
            run_context (RunContext): Include some information of the model.
        """

    def epoch_begin(self, run_context: RunContext):
        """
        Called before each epoch beginning.
        Args:
            run_context (RunContext): Include some information of the model.
        """

    def epoch_end(self, run_context: RunContext):
        """
        Called after each epoch finished.
        Args:
            run_context (RunContext): Include some information of the model.
        """
