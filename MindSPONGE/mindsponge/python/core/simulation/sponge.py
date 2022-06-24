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
"""sponge """
import os
import time
from collections.abc import Iterable

from mindspore import nn
from mindspore.nn import Cell
from mindspore.common import Tensor
from mindspore.nn.optim import Optimizer

from mindspore import context
from mindspore.context import ParallelMode
from mindspore.train.callback import Callback, RunContext, _InternalCallbackParam, _CallbackManager
from mindspore.common.api import _pynative_executor
from mindspore.parallel._utils import _get_parallel_mode, _get_device_num, _get_global_rank, \
    _get_parameter_broadcast, _device_number_check
from mindspore.parallel._ps_context import _is_role_pserver
from mindspore.nn.metrics import get_metrics, Metric
from ..potential.forcefield import ForceField
from .analyse import AnalyseCell
from .simulation import SimulationCell
from .onestep import RunOneStepCell


def _transfer_tensor_to_tuple(inputs):
    """
    If the input is a tensor, convert it to a tuple. If not, the output is unchanged.
    """
    if isinstance(inputs, Tensor):
        return (inputs,)

    return inputs


class _StepSync(Callback):
    @staticmethod
    def step_end(run_context):
        _pynative_executor.sync()


class Sponge():
    """sponge"""
    def __init__(
            self,
            network: Cell,
            potential: ForceField = None,
            integrator: Optimizer = None,
            metrics: Metric = None,
            analyse_network: AnalyseCell = None,
    ):

        self._system = network
        self._potential = potential
        self._integrator = integrator
        self._metrics = metrics
        self._analyse_network = analyse_network

        self._parallel_mode = _get_parallel_mode()
        self._device_number = _get_device_num()
        self._global_rank = _get_global_rank()
        self._parameter_broadcast = _get_parameter_broadcast()
        self._create_time = int(time.time() * 1e9)

        self._check_for_graph_cell()

        if potential is None:
            if integrator is None:
                self.simulation_network = network
                self.network = self.simulation_network.network
                self.integrator = self.simulation_network.integrator
            else:
                self.network = network
                self.integrator = integrator
                self.simulation_network = RunOneStepCell(self.network, self.integrator)
        else:
            self.network = SimulationCell(network, potential)
            if integrator is None:
                raise ValueError('integrator cannot be None is is given')
            self.integrator = integrator
            self.simulation_network = RunOneStepCell(self.network, self.integrator)

        self.potential = self.network.potential
        self.system = self.network.system

        self.units = self.system.units

        try:
            self.time_step = self.integrator.learning_rate.asnumpy()
        except AttributeError:
            self.time_step = 1.0

        self.coordinates = self.system.coordinates
        self.pbc_box = self.system.pbc_box
        self.neighbour_list = self.network.neighbour_list

        self.cutoff = self.neighbour_list.cutoff
        self.nl_update_steps = self.neighbour_list.update_steps

        # Avoiding the bug for return None type
        self.one_neighbour_terms = False
        if self.neighbour_list.no_mask and context.get_context("mode") == context.GRAPH_MODE:
            self.one_neighbour_terms = True

        self._metric_fns = None
        if metrics is not None:
            self._metric_fns = get_metrics(metrics)

        self.analyse_network = analyse_network
        if analyse_network is None and self._metric_fns is not None:
            self.analyse_network = AnalyseCell(self.system, self.potential, self.neighbour_list)

    def set_cutoff(self, cutoff):
        self.network.set_cutoff(cutoff)
        return self

    def _check_for_graph_cell(self):
        """Check for graph cell"""
        if not isinstance(self._system, nn.GraphCell):
            return

        if self._potential is not None or self._integrator is not None:
            raise ValueError("For 'Model', 'loss_fn' and 'optimizer' should be None when network is a GraphCell, "
                             "but got 'loss_fn': {}, 'optimizer': {}.".format(self._potential, self._integrator))

    @staticmethod
    def _transform_callbacks(callbacks):
        """Transform callback to a list."""
        if callbacks is None:
            return []

        if isinstance(callbacks, Iterable):
            return list(callbacks)

        return [callbacks]

    def run(self, steps, callbacks=None, dataset=None):
        """run"""
        cb_params = _InternalCallbackParam()
        cb_params.simulation_network = self.simulation_network
        cb_params.num_steps = steps

        if self.cutoff is None or steps < self.nl_update_steps:
            epoch = 1
            cycle_steps = steps
            rest_steps = 0
        else:
            epoch = steps // self.nl_update_steps
            cycle_steps = self.nl_update_steps
            rest_steps = steps - epoch * cycle_steps

        cb_params.num_steps = steps
        cb_params.time_step = self.time_step
        cb_params.num_epoch = epoch
        cb_params.cycle_steps = cycle_steps
        cb_params.rest_steps = rest_steps
        cb_params.cutoff = self.cutoff

        cb_params.mode = "simulation"
        cb_params.simulation_network = self.simulation_network
        cb_params.system = self.system
        cb_params.potential = self.potential
        cb_params.integrator = self.integrator
        cb_params.parallel_mode = self._parallel_mode
        cb_params.device_number = self._device_number
        cb_params.simulation_dataset = dataset
        cb_params.list_callback = self._transform_callbacks(callbacks)
        if context.get_context("mode") == context.PYNATIVE_MODE:
            cb_params.list_callback.insert(0, _StepSync())
            callbacks = cb_params.list_callback

        cb_params.coordinates = self.coordinates
        cb_params.pbc_box = self.pbc_box

        cb_params.kinetic = self.system.kinetic
        cb_params.temperature = self.system.temperature

        # build callback list
        with _CallbackManager(callbacks) as list_callback:
            self._simulation_process(epoch, cycle_steps, rest_steps, list_callback, cb_params)

        return self

    def _simulation_process(self, epoch, cycle_steps, rest_steps, list_callback=None, cb_params=None):
        """
        Training process. The data would be passed to network directly.

        Args:
            epoch (int): Total number of iterations on the data.
            train_dataset (Dataset): A training dataset iterator. If there is no
                                     loss_fn, a tuple with multiple data (data1, data2, data3, ...) should be
                                     returned and passed to the network. Otherwise, a tuple (data, label) should
                                     be returned. The data and label would be passed to the network and loss
                                     function respectively.
            list_callback (Callback): Executor of callback list. Default: None.
            cb_params (_InternalCallbackParam): Callback parameters. Default: None.
        """
        self._exec_preprocess(True)

        self.sim_step = 0
        self.sim_time = 0.0
        run_context = RunContext(cb_params)
        list_callback.begin(run_context)
        # used to stop training for early stop, such as stopAtTIme or stopATStep
        should_stop = False

        for i in range(epoch):
            cb_params.cur_epoch = i

            self.neighbour_list.update()
            index, mask = self.get_neighbour_list()

            should_stop = self._run_one_epoch(index, mask, cycle_steps, list_callback, cb_params, run_context)
            should_stop = should_stop or run_context.get_stop_requested()
            if should_stop:
                break

        if rest_steps > 0:
            index, mask = self.neighbour_list()
            self._run_one_epoch(index, mask, rest_steps, list_callback, cb_params, run_context)

        list_callback.end(run_context)

    def _run_one_epoch(self, index, mask, cycles, list_callback, cb_params, run_context):
        """run one epoch"""
        should_stop = False
        list_callback.epoch_begin(run_context)
        for j in range(cycles):

            cb_params.cur_step = self.sim_step
            cb_params.cur_time = self.sim_time
            list_callback.step_begin(run_context)

            energy, forces = self.simulation_network(index, mask)

            cb_params.energy = energy
            cb_params.forces = forces
            cb_params.velocities = self.system.velocities

            cb_params.kinetic = self.system.kinetic
            cb_params.temperature = self.system.temperature

            self.sim_step += 1
            self.sim_time += self.time_step

            list_callback.step_end(run_context)

            if _is_role_pserver():
                os._exit(0)
            should_stop = should_stop or run_context.get_stop_requested()
            if should_stop:
                break

        # if param is cache enable, flush data from cache to host before epoch end
        self._flush_from_cache(cb_params)

        list_callback.epoch_end(run_context)
        return should_stop

    def analyse(self, dataset=None, callbacks=None):
        """
        Evaluation API where the iteration is controlled by python front-end.

        Configure to pynative mode or CPU, the evaluating process will be performed with dataset non-sink mode.

        Note:
            If dataset_sink_mode is True, data will be sent to device. If the device is Ascend, features
            of data will be transferred one by one. The limitation of data transmission per time is 256M.
            When dataset_sink_mode is True, the step_end method of the Callback class will be executed when
            the epoch_end method is called.

        Args:
            valid_dataset (Dataset): Dataset to evaluate the model.
            callbacks (Optional[list(Callback)]): List of callback objects which should be executed
                while training. Default: None.
            dataset_sink_mode (bool): Determines whether to pass the data through dataset channel.
                Default: True.

        Returns:
            Dict, the key is the metric name defined by users and the value is the metrics value for
            the model in the test mode.

        Examples:
            >>> from mindspore import Model, nn
            >>>
            >>> # For details about how to build the dataset, please refer to the tutorial
            >>> # document on the official website.
            >>> dataset = create_custom_dataset()
            >>> net = Net()
            >>> loss = nn.SoftmaxCrossEntropyWithLogits()
            >>> model = Model(net, loss_fn=loss, optimizer=None, metrics={'acc'})
            >>> acc = model.eval(dataset, dataset_sink_mode=False)
        """

        _device_number_check(self._parallel_mode, self._device_number)
        if not self._metric_fns:
            raise ValueError("The model argument 'metrics' can not be None or empty, "
                             "you should set the argument 'metrics' for model.")

        cb_params = _InternalCallbackParam()
        cb_params.analyse_network = self.analyse_network
        if dataset is not None:
            cb_params.analysis_dataset = dataset
            cb_params.batch_num = dataset.get_dataset_size()
            cb_params.mode = "analyse"
            cb_params.cur_step_num = 0

        cb_params.list_callback = self._transform_callbacks(callbacks)

        self._clear_metrics()

        with _CallbackManager(callbacks) as list_callback:
            return self._analyse_process(dataset, list_callback, cb_params)

    def _analyse_process(self, dataset=None, list_callback=None, cb_params=None):
        """
        Evaluation. The data would be passed to network directly.

        Args:
            valid_dataset (Dataset): Dataset to evaluate the model.
            list_callback (Callback): Executor of callback list. Default: None.
            cb_params (_InternalCallbackParam): Callback parameters. Default: None.

        Returns:
            Dict, which returns the loss value and metrics values for the model in the test mode.
        """
        run_context = RunContext(cb_params)
        list_callback.begin(run_context)
        dataset_helper, _ = self._exec_preprocess(is_run=False,
                                                  dataset=dataset,
                                                  dataset_sink_mode=False)
        list_callback.epoch_begin(run_context)

        if dataset is None:
            cb_params.cur_step_num += 1
            list_callback.step_begin(run_context)
            outputs = self.analyse_network()
            cb_params.net_outputs = outputs
            list_callback.step_end(run_context)
            self._update_metrics(outputs)
        else:
            for next_element in dataset_helper:
                cb_params.cur_step_num += 1
                list_callback.step_begin(run_context)
                next_element = _transfer_tensor_to_tuple(next_element)
                outputs = self.analyse_network(*next_element)
                cb_params.net_outputs = outputs
                list_callback.step_end(run_context)
                self._update_metrics(outputs)

        list_callback.epoch_end(run_context)
        dataset.reset()
        metrics = self._get_metrics()
        cb_params.metrics = metrics
        list_callback.end(run_context)
        return metrics

    def _clear_metrics(self):
        """Clear metrics local values."""
        for metric in self._metric_fns.values():
            metric.clear()

    def _update_metrics(self, outputs):
        """Update metrics local values."""
        if isinstance(outputs, Tensor):
            outputs = (outputs,)
        if not isinstance(outputs, tuple):
            raise ValueError(f"The argument 'outputs' should be tuple, but got {type(outputs)}.")

        for metric in self._metric_fns.values():
            metric.update(*outputs)

    def _get_metrics(self):
        """Get metrics local values."""
        metrics = dict()
        for key, value in self._metric_fns.items():
            metrics[key] = value.eval()
        return metrics

    def get_neighbour_list(self):
        # Avoiding the bug for return None type
        mask = None
        if self.one_neighbour_terms:
            (index,) = self.neighbour_list()
        else:
            index, mask = self.neighbour_list()
        return index, mask

    def energy(self):
        index, mask = self.get_neighbour_list()
        return self.network(index, mask)

    def energy_and_force(self):
        index, mask = self.get_neighbour_list()
        return self.simulation_network(index, mask)

    def _exec_preprocess(self, is_run):
        """Initializes dataset."""
        if is_run:
            network = self.simulation_network
            phase = 'simulation'
        else:
            network = self.analyse_network
            phase = 'analyse'

        network.set_train(is_run)
        network.phase = phase

        if self._parallel_mode in (ParallelMode.SEMI_AUTO_PARALLEL, ParallelMode.AUTO_PARALLEL):
            network.set_auto_parallel()

        return network

    def _flush_from_cache(self, cb_params):
        """Flush cache data to host if tensor is cache enable."""
        params = cb_params.simulation_network.get_parameters()
        for param in params:
            if param.cache_enable:
                Tensor(param).flush_from_cache()

    @property
    def create_time(self):
        return self._create_time
