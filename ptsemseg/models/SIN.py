import functools

import torch.nn as nn
import torch.nn.functional as F
import torch

import math

from skimage.morphology import thin
from scipy import ndimage
import copy
import numpy as np
from torch.utils import model_zoo
from ptsemseg.models.utils import freeze_weights, \
                                  masked_embeddings, \
                                  weighted_masked_embeddings, \
                                  compute_weight
from ptsemseg.models.resnet import Bottleneck, conv1x1, conv3x3
import torchvision
import copy

class SIN(nn.Module):
    def __init__(self, layers, n_classes=21, zero_init_residual=False):
        super(SIN, self).__init__()

        ####################### Construct ResNet 50 8stride Model #####################################
        self.inplanes = 64
        self.n_classes = n_classes
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(Bottleneck, 64, layers[0])
        self.layer2 = self._make_layer(Bottleneck, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, layers[2], stride=2)
        self.score_channels = [512, 512, 256, 64]

        self.proj = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1)

        self.classifier = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Dropout2d(),
            nn.Conv2d(self.score_channels[0], self.n_classes, 1, bias=False),
            )
        self.score_pool4 = nn.Conv2d(self.score_channels[1], self.n_classes, 1, bias=False)
        self.score_pool3 = nn.Conv2d(self.score_channels[2], self.n_classes, 1, bias=False)
        self.score_pool2 = nn.Conv2d(self.score_channels[3], self.n_classes, 1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        ################################# Load Stylied ImageNet Model###############################
        model_urls = {
        'resnet50_trained_on_SIN': \
                'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/6f41d2e86fc60566f78de64ecff35cc61eb6436f/resnet50_train_60_epochs-c8e5653e.pth.tar',
        'resnet50_trained_on_SIN_and_IN': \
                'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_train_45_epochs_combined_IN_SF-2a0d100e.pth.tar',
        'resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN': \
                'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_finetune_60_epochs_lr_decay_after_30_start_resnet50_train_45_epochs_combined_IN_SF-ca06340c.pth.tar',
        }

        temp_model = torchvision.models.resnet50(pretrained=False)
        temp_model = torch.nn.DataParallel(temp_model).cuda()
        checkpoint = model_zoo.load_url(model_urls['resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN'])
        temp_model.load_state_dict(checkpoint["state_dict"])
        self.init_resnet50_params(temp_model)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def init_resnet50_params(self, model_weights):
        model_layers = dict(model_weights.module.named_modules())
        my_layers = dict(self.named_modules())

        for k, v in model_layers.items():
            if k not in my_layers.keys():
                continue
            if isinstance(my_layers[k], nn.Conv2d) or isinstance(my_layers[k], nn.BatchNorm2d):
                assert my_layers[k].weight.size() == model_layers[k].weight.size()
                my_layers[k].weight.data = model_layers[k].weight.data
                if my_layers[k].bias is not None:
                    my_layers[k].bias.data = model_layers[k].bias.data
                if isinstance(my_layers[k], nn.BatchNorm2d):
                    my_layers[k].running_mean.data = model_layers[k].running_mean.data
                    my_layers[k].running_var.data = model_layers[k].running_var.data

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.relu(self.bn1(x1))
        x2 = self.maxpool(x1)
        x3 = self.layer1(x2)
        x4 = self.layer2(x3)
        x5 = self.layer3(x4)

        proj = self.proj(x5)
        score = self.classifier(proj)
        score_pool2 = self.score_pool2(x2)
        score_pool3 = self.score_pool3(x3)
        score_pool4 = self.score_pool4(x4)
        score = F.upsample(score, score_pool4.size()[2:])
        score += score_pool4

        score = F.upsample(score, score_pool3.size()[2:])
        score += score_pool3

        score = F.upsample(score, score_pool2.size()[2:])
        score += score_pool2


        out = F.upsample(score, x.size()[2:])

        return out

    def extract(self, x, label):
        x1 = self.conv1(x)
        x1 = self.relu(self.bn1(x1))
        x2 = self.maxpool(x1)
        x3 = self.layer1(x2)
        x4 = self.layer2(x3)
        x5 = self.layer3(x4)

        fconv_pooled = masked_embeddings(x5.shape, label, x5,
                                         self.n_classes)
        conv3_pooled = masked_embeddings(x3.shape, label, x3,
                                         self.n_classes)
        conv4_pooled = masked_embeddings(x4.shape, label, x4,
                                         self.n_classes)

        return fconv_pooled, conv4_pooled, conv3_pooled
