# coding=utf-8
import math
import warnings

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

class SimulatedConv2d(nn.Module):

    def __init__(
            self,
            original_layer: nn.Conv2d,
            path_to_arch_file: str = '',
            path_to_tile: str = '',
            sparsity_ratio: float = 0
    ) -> None:
        super(SimulatedConv2d, self).__init__()
        self.in_channels = original_layer.in_channels
        self._reversed_padding_repeated_twice = original_layer._reversed_padding_repeated_twice
        self.out_channels = original_layer.out_channels
        self.kernel_size = original_layer.kernel_size
        self.stride = original_layer.stride
        self.padding = original_layer.padding
        self.dilation = original_layer.dilation
        self.output_padding = original_layer.output_padding
        self.groups = original_layer.groups
        self.padding_mode = original_layer.padding_mode
        self.weight = original_layer.weight
        self.bias = original_layer.bias

        # STONNE specific parameters
        self.path_to_arch_file = path_to_arch_file
        self.path_to_tile = path_to_tile
        self.sparsity_ratio = sparsity_ratio

    def forward(self, input: Tensor) -> Tensor:
        import torch_stonne
        if self.padding_mode != 'zeros':
            output = torch_stonne.simulated_conv_forward(self.__class__.__name__,
                                                         F.pad(input, self._reversed_padding_repeated_twice,
                                                               mode=self.padding_mode), self.weight, self.stride,
                                                         _pair(0), self.dilation, self.groups, self.path_to_arch_file,
                                                         self.path_to_tile, self.sparsity_ratio)
        else:
            output = torch_stonne.simulated_conv_forward(self.__class__.__name__, input, self.weight, self.stride,
                                                         self.padding, self.dilation, self.groups,
                                                         self.path_to_arch_file, self.path_to_tile, self.sparsity_ratio)

        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)
        return output

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)
