import functools

import torch.nn as nn
import torch.nn.functional as F
import torch

from ptsemseg.models.dilated_fcn import dilated_fcn8s, dilated_fcn32s

# FCN 8s
class reduced_fcn8s(dilated_fcn8s):
    def __init__(self, *args, **kwargs):
        super(reduced_fcn8s, self).__init__(*args, **kwargs)
        self.fconv_block = nn.Sequential(nn.Conv2d(512, 256, 1))

class reduced_fcn32s(dilated_fcn32s):
    def __init__(self, *args, **kwargs):
        super(reduced_fcn32s, self).__init__(*args, **kwargs)
        self.fconv_block = nn.Sequential(nn.Conv2d(512, 256, 1))

