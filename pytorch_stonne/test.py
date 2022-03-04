import torch
import torch.nn as nn
from convert_model import create_sim_model_copy
import pdb

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, (5, 5), groups=1)  # in_channels=5, out_channels=5, filter_size=5
        self.relu1 = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(36, 5)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = x.view(-1, 36)
        x = self.fc1(x)
        x = self.relu2(x)
        return x


device = torch.device('cpu')

net = Net().cpu()
# print(net)
input_test = torch.randn(1, 1, 10, 10)
print('Basic conv', net(input_test))

# Copy the provided model and change conv2d and linear layers to simulated versions.
# path to arch file and path to tile are provided in covert_model.py
# If necessary, you can provide it as a parameter to this function and have multiple simulated nets,
# to try out different architectures/tiling configurations.
net2 = create_sim_model_copy(net)

result = net2(input_test)
print('STONNE conv', result)
