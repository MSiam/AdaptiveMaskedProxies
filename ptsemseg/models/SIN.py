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
import torchvision

class SIN(nn.Module):
    def __init__(self, n_classes=21):
        super(SIN, self).__init__()
        self.n_classes = n_classes
        self.score_channels = [1024, 512, 256]

        model_urls = {
        'resnet50_trained_on_SIN': \
                'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/6f41d2e86fc60566f78de64ecff35cc61eb6436f/resnet50_train_60_epochs-c8e5653e.pth.tar',
        'resnet50_trained_on_SIN_and_IN': \
                'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_train_45_epochs_combined_IN_SF-2a0d100e.pth.tar',
        'resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN': \
                'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_finetune_60_epochs_lr_decay_after_30_start_resnet50_train_45_epochs_combined_IN_SF-ca06340c.pth.tar',
        }

        self.model = torchvision.models.resnet50(pretrained=False)
        checkpoint = model_zoo.load_url(model_urls['resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN'])
        self.model = torch.nn.DataParallel(self.model).cuda()
        self.model.load_state_dict(checkpoint["state_dict"])

        self.extractor = list(list(self.model.children())[0].children())[:-3]
        self.classifier = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(self.score_channels[0], self.n_classes, 1, bias=False),
            )
        self.score_pool4 = nn.Conv2d(self.score_channels[1], self.n_classes, 1, bias=False)
        self.score_pool3 = nn.Conv2d(self.score_channels[2], self.n_classes, 1, bias=False)


    def forward(self, x):
        feats = x
        for m in range(len(self.extractor)-3):
            feats = self.extractor[m](feats)
            print('FEATS ', feats.shape)
        conv4 = self.extractor[-3](feats)
        conv3 = self.extractor[-2](conv4)
        fconv = self.extractor[-1](conv3)
        score = self.classifier(fconv)
        score_pool4 = self.score_pool4(conv4)
        score_pool3 = self.score_pool3(conv3)
        score = F.upsample(score, score_pool4.size()[2:])
        score += score_pool4
        score = F.upsample(score, score_pool3.size()[2:])
        score += score_pool3

        out = F.upsample(score, x.size()[2:])

        return out

    def extract(self, x, label):
        feats = x
        for m in range(len(self.extractor)-3):
            feats = self.extractor[m](feats)
            print('FEATS ', feats.shape)

        conv4 = self.extractor[-3](feats)
        conv3 = self.extractor[-2](conv4)
        fconv = self.extractor[-1](conv3)

        fconv_pooled = masked_embeddings(fconv.shape, label, fconv,
                                         self.n_classes)
        conv3_pooled = masked_embeddings(conv3.shape, label, conv3,
                                         self.n_classes)
        conv4_pooled = masked_embeddings(conv4.shape, label, conv4,
                                         self.n_classes)

        return fconv_pooled, conv4_pooled, conv3_pooled

    def imprint(self, images, labels, nchannels, alpha):
        with torch.no_grad():
            embeddings = None
            for ii, ll in zip(images, labels):
                #ii = ii.unsqueeze(0)
                ll = ll[0]
                if embeddings is None:
                    embeddings, early_embeddings, vearly_embeddings = self.extract(ii, ll)
                    embeddings = embeddings
                    early_embeddings = early_embeddings
                    vearly_embeddings = vearly_embeddings
                else:
                    embeddings_, early_embeddings_, vearly_embeddings_ = self.extract(ii, ll)
                    embeddings = torch.cat((embeddings, embeddings_), 0)
                    early_embeddings = torch.cat((early_embeddings, early_embeddings_), 0)
                    vearly_embeddings = torch.cat((vearly_embeddings, vearly_embeddings_), 0)

            # Imprint weights for last score layer
            nclasses = self.n_classes
            self.n_classes = 17

            weight = compute_weight(embeddings, nclasses, labels,
                                         self.classifier[2].weight.data, alpha=alpha)
            self.classifier[2] = nn.Conv2d(nchannels, self.n_classes, 1, bias=False)
            self.classifier[2].weight.data = weight

            weight4 = compute_weight(early_embeddings, nclasses, labels,
                                     self.score_pool4.weight.data, alpha=alpha)
            self.score_pool4 = nn.Conv2d(self.score_channels[0], self.n_classes, 1, bias=False)
            self.score_pool4.weight.data = weight4

            weight3 = compute_weight(vearly_embeddings, nclasses, labels,
                                     self.score_pool3.weight.data, alpha=alpha)
            self.score_pool3 = nn.Conv2d(self.score_channels[1], self.n_classes, 1, bias=False)
            self.score_pool3.weight.data = weight3

            assert self.classifier[2].weight.is_cuda
            assert self.score_pool3.weight.is_cuda
            assert self.score_pool4.weight.is_cuda
            assert self.score_pool3.weight.data.shape[1] == self.score_channels[1]
            assert self.classifier[2].weight.data.shape[1] == 256
            assert self.score_pool4.weight.data.shape[1] == self.score_channels[0]

    def save_original_weights(self):
        self.original_weights = []
        self.original_weights.append(copy.deepcopy(self.classifier[2].weight.data))
        self.original_weights.append(copy.deepcopy(self.score_pool4.weight.data))
        self.original_weights.append(copy.deepcopy(self.score_pool3.weight.data))


    def reverse_imprinting(self, nchannels, cl=False):
        if cl:
            print('reverse with enabled continual learning')
            self.n_classes = 16
            weight = copy.deepcopy(self.classifier[2].weight.data[:-1, ...])
            self.classifier[2] = nn.Conv2d(nchannels, self.n_classes, 1, bias=False)
            self.classifier[2].weight.data = weight

            weight_sp4 = copy.deepcopy(self.score_pool4.weight.data[:-1, ...])
            self.score_pool4 = nn.Conv2d(self.score_channels[0], self.n_classes, 1, bias=False)
            self.score_pool4.weight.data = weight_sp4

            weight_sp3 = copy.deepcopy(self.score_pool3.weight.data[:-1, ...])
            self.score_pool3 = nn.Conv2d(self.score_channels[1], self.n_classes, 1, bias=False)
            self.score_pool3.weight.data = weight_sp3
        else:
            print('No Continual Learning for Bg')
            self.n_classes = 16
            self.classifier[2] = nn.Conv2d(nchannels, self.n_classes, 1, bias=False)
            self.classifier[2].weight.data = copy.deepcopy(self.original_weights[0])

            self.score_pool4 = nn.Conv2d(self.score_channels[0], self.n_classes, 1, bias=False)
            self.score_pool4.weight.data = copy.deepcopy(self.original_weights[1])

            self.score_pool3 = nn.Conv2d(self.score_channels[1], self.n_classes, 1, bias=False)
            self.score_pool3.weight.data = copy.deepcopy(self.original_weights[2])

        assert self.score_pool3.weight.data.shape[1] == self.score_channels[1]
        assert self.classifier[2].weight.data.shape[1] == 256
        assert self.score_pool4.weight.data.shape[1] == self.score_channels[0]

    def init_vgg16_params(self, vgg16, copy_fc8=False):
        blocks = [
            self.conv_block1,
            self.conv_block2,
            self.conv_block3,
            self.conv_block4,
            self.conv_block5,
        ]

        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(vgg16.features.children())

        for idx, conv_block in enumerate(blocks):
            for l1, l2 in zip(features[ranges[idx][0] : ranges[idx][1]], conv_block):
                if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data
#        for i1, i2 in zip([0, 3], [0, 3]):
#            l1 = vgg16.classifier[i1]
#            l2 = self.fconv_block[i2]
#            l2.weight.data = l1.weight.data.view(l2.weight.size())
#            l2.bias.data = l1.bias.data.view(l2.bias.size())
#        n_class = self.classifier[2].weight.size()[0]
#        if copy_fc8:
#            l1 = vgg16.classifier[6]
#            l2 = self.classifier[2]
#            l2.weight.data = l1.weight.data[:n_class, :].view(l2.weight.size())
#            l2.bias.data = l1.bias.data[:n_class]

    def freeze_weights_extractor(self):
        freeze_weights(self.conv_block1)
        freeze_weights(self.conv_block2)
        freeze_weights(self.conv_block3)
        freeze_weights(self.conv_block4)
        freeze_weights(self.conv_block5)

    def freeze_all_except_classifiers(self):
        self.freeze_weights_extractor()
        freeze_weights(self.fconv_block)
