import torch
import torch.nn as nn
import pdb

from SimulatedConv import SimulatedConv2d
from SimulatedLinear import SimulatedLinear


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(5, 5, (5, 5), groups=1)  # in_channels=5, out_channels=5, filter_size=5
        self.relu1 = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(10, 5)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = x.view(-1, 180)
        x = self.fc1(x)
        x = self.relu2(x)
        return x


path_to_arch_file = '../simulation_files/maeri_128mses_128_bw.cfg'
path_to_tile = '../minibenchmarks/dogsandcats_tile.txt'


def change_layers_to_sim(model):
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            in_ch = net._modules['conv1'].in_channels
            out_ch = net._modules['conv1'].out_channels
            kernel_size = net._modules['conv1'].kernel_size
            stride = net._modules['conv1'].stride
            model._modules[name] = SimulatedConv2d(in_ch, out_ch, kernel_size,
                                                   stride=stride,
                                                   groups=1,
                                                   path_to_arch_file=path_to_arch_file,
                                                   path_to_tile=path_to_tile,
                                                   sparsity_ratio=0.0)
        if isinstance(layer, nn.Linear):
            in_ch = net._modules['conv1'].in_channels
            out_ch = net._modules['conv1'].out_channels
            model._modules[name] = SimulatedLinear(in_ch, out_ch,
                                                   path_to_arch_file=path_to_arch_file,
                                                   path_to_tile=path_to_tile,
                                                   sparsity_ratio=0.0)


device = torch.device('cpu')

net = Net().cpu()
# print(net)
input_test = torch.randn(1, 5, 10, 10)
# print('Basic conv', net(input_test))

net2 = Net().cpu()
change_layers_to_sim(net2)

result = net2(input_test)
# print('STONNE conv', result)
