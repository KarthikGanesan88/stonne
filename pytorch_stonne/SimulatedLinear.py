import math

import torch
from torch import Tensor
import torch.nn as nn

class SimulatedLinear(nn.Module):

    def __init__(
            self,
            original_layer: nn.Linear,
            path_to_arch_file: str,
            path_to_tile: str,
            sparsity_ratio: float,

    ) -> None:
        super(SimulatedLinear, self).__init__()
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.weight = original_layer.weight
        self.bias = original_layer.bias
        self.path_to_arch_file = path_to_arch_file
        self.path_to_tile = path_to_tile
        self.sparsity_ratio = sparsity_ratio

    def forward(self, input: Tensor) -> Tensor:
        import torch_stonne
        output = torch_stonne.simulated_linear_forward(self.__class__.__name__, input, self.weight,
                                                       self.path_to_arch_file, self.path_to_tile,
                                                       self.sparsity_ratio, True  # To transpose the matrices
                                                       )
        if self.bias is not None:
            output += self.bias
        return output

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
