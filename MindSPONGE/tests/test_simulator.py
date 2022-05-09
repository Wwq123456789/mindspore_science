from mindspore import context, Tensor
from mindsponge.core.potential import BasicEnergy
from mindsponge.core.space import Space
from mindsponge.core.control import LeapFrogLiuJian
from mindsponge.core import Simulation
from mindsponge.data import system
from mindsponge.common import load_config
import mindspore.common.dtype as mstype
import numpy as np

context.set_context(mode=context.GRAPH_MODE,
                    device_target="Ascend",
                    variable_memory_max_size="31GB",
                    device_id=0,
                    save_graphs=False,
                    save_graphs_path="./graph1/")

config = load_config("./polypeptide.yaml")

#load AMBER Format File (topology file and coordinates file)
system = system(config)
# print(system.velocities)
# _type_list = [np.int32, np.float32]
# for key in vars(system):
#     if type(vars(system)[key]) in _type_list:
#         print(key)
#basic system information converter (Only coordinates and velocities are Parameter can be updated)
space = Space(system=system)
# for key in vars(space):
#     if type(vars(space)[key]) is Tensor:
#         print(key, vars(space)[key].shape, vars(space)[key].dtype)
# #define energy function(Contain angle, bond, dihedral and non bond)
energy_fn = BasicEnergy(space)

# #define integrator
integrator = LeapFrogLiuJian(space)

random_force = Tensor(np.random.standard_normal(size=(space.atom_numbers, 3)), mstype.float32)

# #define simulator (initial simulation obj)
# # simulator = Simulation(space = space, energy_func = energy_fn, integrator = integrator, mode = config.mode)
simulator = Simulation(space = space, energy_func = energy_fn, mode = "NVE")
# #run simulation
energy_inputs = (space.box_len, space.charge)
# integrator_inputs = (random_force,)
# integrator_inputs = ()
simulator.run(1, energy_inputs = energy_inputs)
