import torch
import torch.nn as nn

from .wrappers import Conv2d


class DepthwiseConv(nn.Module):
    
    def __init__(self, input_dim, output_dim, kernel_size=3, stride=1, padding=1, bias=True, norm=None, activation=None):
        super(DepthwiseConv, self).__init__()

        self.norm = norm
        self.activation = activation
        
        self.depthwise = nn.Conv2d(input_dim, input_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, groups=input_dim)
        self.pointwise = nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=bias)

    def forward(self, x):

        x = self.depthwise(x)
        x = self.pointwise(x)

        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)

        return x