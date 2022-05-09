import numpy as np
import mindspore.common.dtype as mstype
import mindspore.numpy as mnp
from mindspore import nn
from mindspore import Tensor, Parameter
from mindspore.ops import functional as F
import mindspore.ops as ops
from .potential import Force
from .control import temperature, LeapFrogLiuJian, LeapFrog, GradientDescent
import logging
class Simulation:
    '''simulation'''
    def __init__(self, space, energy_func, integrator = None, mode = None):
        if integrator is None and mode is None:
            raise ValueError("integrator or mode should not be None at the same time")
        self._space = space
        if mode == "NVE":
            self._integrator = LeapFrog(self._space)
        elif mode == "Minimization":
            self._integrator = GradientDescent(self._space)
        elif mode == "NPT" or mode == "NVT":
            self._integrator = LeapFrogLiuJian(self._space)
        self._energy_fn = energy_func
        self._force_fn = Force(self._energy_fn, self._space)
        if integrator is not None:
            self._integrator = integrator
        self._sim = self._build()

    def _build(self):
        class RunOneStepCell(nn.Cell):
            def __init__(self, energy_func, force_func, integrator, space):
                super(RunOneStepCell, self).__init__()
                self.mode = space.mode
                self.energy_func = energy_func
                self.force_func = force_func
                self.integrator = integrator
                self.residue_numbers = space.residue_numbers
                self.atom_mass = space.mass
            def construct(self, energy_inputs = None, integrator_inputs = None):
                if energy_inputs == None:
                    total_energy = self.energy_func()
                else:
                    total_energy = self.energy_func(*energy_inputs)
                force = self.force_func()
                if integrator_inputs == None:
                    integrator_inputs = (force,)
                else:
                    integrator_inputs = (force,) + integrator_inputs
                crd, vel = self.integrator(*integrator_inputs)
                temp = temperature(self.atom_mass, vel)
                return total_energy, temp

        sim = RunOneStepCell(self._energy_fn, self._force_fn, self._integrator, self._space)
        return sim

    def run(self, step, energy_inputs=None, integrator_inputs=None):
        for _ in range(step):
            total_ene, temperature = self._sim(energy_inputs, integrator_inputs)
            print(total_ene, temperature)

