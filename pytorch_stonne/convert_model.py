import torch.nn as nn
import copy
from SimulatedConv import SimulatedConv2d
from SimulatedLinear import SimulatedLinear

path_to_arch_file = '../simulation_files/maeri_128mses_128_bw.cfg'
path_to_tile = '../minibenchmarks/dogsandcats_tile.txt'


def create_sim_model_copy(net):

    model = copy.deepcopy(net)
    model = model.cpu()  # Make sure the new model is on CPU since stonne only runs on CPU.

    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            model._modules[name] = SimulatedConv2d(layer,
                                                   path_to_arch_file=path_to_arch_file,
                                                   path_to_tile=path_to_tile,
                                                   sparsity_ratio=0.0)

        if isinstance(layer, nn.Linear):
            model._modules[name] = SimulatedLinear(layer,
                                                   path_to_arch_file=path_to_arch_file,
                                                   path_to_tile=path_to_tile,
                                                   sparsity_ratio=0.0)

    return model