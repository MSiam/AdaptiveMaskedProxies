import torch.nn as nn
import torch.nn.functional as F
import torch

class CosineSimLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias, use_scale=False):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias)
        self.use_scale = use_scale
        if use_scale:
            self.scale = nn.Parameter(torch.ones(1)*20.0)

    def set_scale(self, use_scale=False):
        self.use_scale = use_scale
        if use_scale:
            self.scale = nn.Parameter(torch.ones(1)*20.0)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        self.conv.weight.data = F.normalize(self.conv.weight.data, dim=1)
        out = self.conv(x)
        if self.use_scale:
            out = self.scale * out
        return out

