import torch
import torch.nn as nn
from SimulatedConv import SimulatedConv2d
from SimulatedLinear import SimulatedLinear

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(5, 5, (5, 5), groups=1)  # in_channels=5, out_channels=5, filter_size=5


    def forward(self, x):
        x = self.conv1(x)
        return x


path_to_arch_file = '../simulation_files/maeri_128mses_128_bw.cfg'
path_to_tile = '../minibenchmarks/dogsandcats_tile.txt'

class NetSim(nn.Module):
    def __init__(self):
        super().__init__()
        self.path_to_arch_file = ""
        self.path_to_tile = ""
        self.conv1 = SimulatedConv2d(5, 5, (5, 5), groups=1,
                                     path_to_arch_file=path_to_arch_file,
                                     path_to_tile=path_to_tile,
                                     sparsity_ratio=0.0)
        # self.fc1 = SimulatedLinear()

    def forward(self, x):
        x = self.conv1(x)
        # x = x.view(-1 , )
        return x

device = torch.device('cpu')

net = Net().cpu()
# print(net)
input_test = torch.randn(1, 5, 10, 10)
# print('Basic conv', net(input_test))

net2 = NetSim().cpu()
result = net2(input_test)
# print('STONNE conv', result)
