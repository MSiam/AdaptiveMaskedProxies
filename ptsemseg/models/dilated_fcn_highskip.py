import functools

import torch.nn as nn
import torch.nn.functional as F
import torch

from ptsemseg.models.utils import get_upsampling_weight, l2_norm
from ptsemseg.loss import cross_entropy2d
import math

from skimage.morphology import thin
from scipy import ndimage
import copy
import numpy as np
from ptsemseg.models.fcn import fcn8s
from ptsemseg.models.utils import freeze_weights, \
                                  masked_embeddings, \
                                  weighted_masked_embeddings, \
                                  compute_weight

# FCN 8s
class dilated_fcn8s_highskip(fcn8s):
    def __init__(self, *args, **kwargs):
        super(dilated_fcn8s_highskip, self).__init__(*args, **kwargs)

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, dilation=2, padding=2),
            nn.ReLU(inplace=True),
#            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, dilation=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, dilation=4, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, dilation=4, padding=4),
            nn.ReLU(inplace=True),
#            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.score_channels = [128, 64]
        self.score_pool4 = nn.Conv2d(128, self.n_classes, 1, bias=False)
        self.score_pool3 = nn.Conv2d(64, self.n_classes, 1, bias=False)

    def extract(self, x, label):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)
        conv5 = self.conv_block5(conv4)
        fconv = self.fconv_block(conv5)

        if self.use_norm_weights:
            fconv_norm = l2_norm(fconv)
            conv2_norm = l2_norm(conv2)
            conv1_norm = l2_norm(conv1)
        else:
            fconv_norm = fconv
            conv1_norm = conv1
            conv2_norm = conv2

        if self.weighted_mask:
            fconv_pooled = weighted_masked_embeddings(fconv_norm.shape, label,
                                                      fconv_norm, self.n_classes,
                                                      pad=True)
            conv1_pooled = weighted_masked_embeddings(conv1_norm.shape, label,
                                                      conv1_norm, self.n_classes,
                                                      pad=True)
            conv2_pooled = weighted_masked_embeddings(conv2_norm.shape, label,
                                                      conv2_norm, self.n_classes,
                                                      pad=True)
        else:
            fconv_pooled = masked_embeddings(fconv_norm.shape, label, fconv_norm,
                                             self.n_classes, pad=True)
            conv1_pooled = masked_embeddings(conv1_norm.shape, label, conv1_norm,
                                             self.n_classes, pad=True)
            conv2_pooled = masked_embeddings(conv2_norm.shape, label, conv2_norm,
                                             self.n_classes, pad=True)

        return fconv_pooled, conv2_pooled, conv1_pooled

    def forward(self, x):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)
        conv5 = self.conv_block5(conv4)
        fconv = self.fconv_block(conv5)

        if self.use_norm:
            fconv = l2_norm(fconv)
        if self.use_scale:
            fconv = self.scale * fconv

        score = self.classifier(fconv)

        if self.use_norm:
            conv2 = l2_norm(conv2)
            conv1 = l2_norm(conv1)

        if self.use_scale:
            conv2 = self.scale * conv2
            conv1 = self.scale * conv1

        if self.multires:
            score_pool4 = self.score_pool4(conv2)
            score_pool3 = self.score_pool3(conv1)
            score = F.upsample(score, score_pool4.size()[2:])
            score += score_pool4
            score = F.upsample(score, score_pool3.size()[2:])
            score += score_pool3

        out = F.upsample(score, x.size()[2:])

        return out

