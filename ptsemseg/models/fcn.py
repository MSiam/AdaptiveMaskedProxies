import functools

import torch.nn as nn
import torch.nn.functional as F
import torch

from ptsemseg.models.utils import get_upsampling_weight, l2_norm
from ptsemseg.models.cosine_sim import CosineSimLayer

from ptsemseg.loss import cross_entropy2d
import math

from skimage.morphology import thin
from scipy import ndimage
import copy
import numpy as np
from ptsemseg.models.utils import freeze_weights, \
                                  masked_embeddings, \
                                  weighted_masked_embeddings, \
                                  compute_weight
import matplotlib.pyplot as plt

# FCN 8s
class fcn8s(nn.Module):
    def __init__(self, n_classes=21, learned_billinear=False,
                 use_norm=False, use_scale=False, lower_dim=True,
                 weighted_mask=False, offsetting=False, use_norm_weights=False,
                 use_normalize_train=False):
        super(fcn8s, self).__init__()
        self.learned_billinear = learned_billinear
        self.n_classes = n_classes
        self.loss = functools.partial(cross_entropy2d,
                                      size_average=False)
        self.use_norm = use_norm
        self.use_norm_weights = use_norm_weights
        self.use_normalize_train = use_normalize_train

        self.multires = True
        self.weighted_mask = weighted_mask
        self.offsetting = offsetting

        self.score_channels = [512, 256]

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        if self.use_normalize_train:
            last_layer_cls = CosineSimLayer
        else:
            last_layer_cls = nn.Conv2d

        if lower_dim:
            self.fconv_block = nn.Sequential(
                nn.Conv2d(512, 4096, 1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(),
                nn.Conv2d(4096, 4096, 1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(),
                nn.Conv2d(4096, 256, 1),
            )
            self.classifier = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Dropout2d(),
                last_layer_cls(256, self.n_classes, 1, bias=False),
            )
        else:
            self.fconv_block = nn.Sequential(
                nn.Conv2d(512, 4096, 7),
                nn.ReLU(inplace=True),
                nn.Dropout2d(),
                nn.Conv2d(4096, 4096, 1),
            )
            self.classifier = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Dropout2d(),
                last_layer_cls(4096, self.n_classes, 1, bias=False),
            )

        self.score_pool4 = last_layer_cls(512, self.n_classes, 1, bias=False)
        self.score_pool3 = last_layer_cls(256, self.n_classes, 1, bias=False)

        if use_normalize_train and use_scale:
            self.classifier[2].set_scale(use_scale)
            self.score_pool3.set_scale(use_scale)
            self.score_pool4.set_scale(use_scale)

        if self.learned_billinear:
            self.upscore2 = nn.ConvTranspose2d(self.n_classes, self.n_classes, 4,
                                               stride=2, bias=False)
            self.upscore4 = nn.ConvTranspose2d(self.n_classes, self.n_classes, 4,
                                               stride=2, bias=False)
            self.upscore8 = nn.ConvTranspose2d(self.n_classes, self.n_classes, 16,
                                               stride=8, bias=False)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.copy_(get_upsampling_weight(m.in_channels,
                                                          m.out_channels,
                                                          m.kernel_size[0]))

    def forward(self, x):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)
        conv5 = self.conv_block5(conv4)
        fconv = self.fconv_block(conv5)

        score = self.classifier(fconv)
        score_pool4 = self.score_pool4(conv4)
        score_pool3 = self.score_pool3(conv3)
        score = F.upsample(score, score_pool4.size()[2:])
        score += score_pool4
        score = F.upsample(score, score_pool3.size()[2:])
        score += score_pool3
        out = F.upsample(score, x.size()[2:])
        return out

    def ensemble_classify(self, preds):
        for i in range(preds[0].shape[1]):
            for j in range(preds[0].shape[2]):
                if preds[1][0, i, j] not in [0, 16]:
                    preds[0][0, i, j] = preds[1][0, i, j]
        return preds[0]

    def extract(self, x, label):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)
        conv5 = self.conv_block5(conv4)
        fconv = self.fconv_block(conv5)

        fconv_pooled = masked_embeddings(fconv.shape, label, fconv,
                                         self.n_classes)
        conv3_pooled = masked_embeddings(conv3.shape, label, conv3,
                                         self.n_classes)
        conv4_pooled = masked_embeddings(conv4.shape, label, conv4,
                                         self.n_classes)

        return fconv_pooled, conv4_pooled, conv3_pooled

    def imprint(self, images, labels, alpha, random=False, new_class=True):
        with torch.no_grad():
            embeddings = None
            for ii, ll in zip(images, labels):
                #ii = ii.unsqueeze(0)
                ll = ll[0]
                if embeddings is None:
                    embeddings, early_embeddings, vearly_embeddings = self.extract(ii, ll)
                else:
                    embeddings_, early_embeddings_, vearly_embeddings_ = self.extract(ii, ll)
                    embeddings = torch.cat((embeddings, embeddings_), 0)
                    early_embeddings = torch.cat((early_embeddings, early_embeddings_), 0)
                    vearly_embeddings = torch.cat((vearly_embeddings, vearly_embeddings_), 0)
            # Imprint weights for last score layer
            nclasses = self.n_classes
            self.n_classes = 17
            nchannels = embeddings.shape[2]

            if self.use_normalize_train:
                final_cls_weight = self.classifier[2].conv.weight.data
            else:
                final_cls_weight = self.classifier[2].weight.data

            weight = compute_weight(embeddings, nclasses, labels,
                                    final_cls_weight,
                                    alpha=alpha, new_class=new_class)

            if self.use_normalize_train:
                self.classifier[2] = CosineSimLayer(nchannels, self.n_classes, 1, bias=False)
                if not random:
                    self.classifier[2].conv.weight.data = weight
                else:
                    self.classifier[2].conv.weight.data[:-1, ...] = copy.deepcopy(self.original_weights[0])
                    self.classifier[2].conv.weight.data = self.classifier[2].conv.weight.data.cuda()
            else:
                self.classifier[2] = nn.Conv2d(nchannels, self.n_classes, 1, bias=False)
                if not random:
                    self.classifier[2].weight.data = weight
                else:
                    self.classifier[2].weight.data[:-1, ...] = copy.deepcopy(self.original_weights[0])
                    self.classifier[2].weight.data = self.classifier[2].conv.weight.data.cuda()

            if self.use_normalize_train:
                score4_cls_weight = self.score_pool4.conv.weight.data
            else:
                score4_cls_weight = self.score_pool4.weight.data

            weight4 = compute_weight(early_embeddings, nclasses, labels,
                                     score4_cls_weight,
                                     alpha=alpha, new_class=new_class)

            if self.use_normalize_train:
                self.score_pool4 = CosineSimLayer(self.score_channels[0], self.n_classes, 1, bias=False)
                if not random:
                    self.score_pool4.conv.weight.data = weight4
                else:
                    self.score_pool4.conv.weight.data[:-1, ...] = copy.deepcopy(self.original_weights[1])
                    self.score_pool4.conv.weight.data = self.score_pool4.conv.weight.data.cuda()

            else:
                self.score_pool4 = nn.Conv2d(self.score_channels[0], self.n_classes, 1, bias=False)
                if not random:
                    self.score_pool4.weight.data = weight4
                else:
                    self.score_pool4.weight.data[:-1, ...] = copy.deepcopy(self.original_weights[1])
                    self.score_pool4.weight.data = self.score_pool4.weight.data.cuda()

            if self.use_normalize_train:
                score3_cls_weight = self.score_pool3.conv.weight.data
            else:
                score3_cls_weight = self.score_pool3.weight.data

            weight3 = compute_weight(vearly_embeddings, nclasses, labels,
                                     score3_cls_weight,
                                     alpha=alpha, new_class=new_class)
            if self.use_normalize_train:
                self.score_pool3 = CosineSimLayer(self.score_channels[1], self.n_classes, 1, bias=False)
                if not random:
                    self.score_pool3.conv.weight.data = weight3
                else:
                    self.score_pool3.conv.weight.data[:-1, ...] = copy.deepcopy(self.original_weights[2])
                    self.score_pool3.conv.weight.data = self.score_pool3.conv.weight.data.cuda()

                assert self.classifier[2].conv.weight.is_cuda
                assert self.score_pool3.conv.weight.is_cuda
                assert self.score_pool4.conv.weight.is_cuda
                assert self.score_pool3.conv.weight.data.shape[1] == self.score_channels[1]
                assert self.classifier[2].conv.weight.data.shape[1] == 256
                assert self.score_pool4.conv.weight.data.shape[1] == self.score_channels[0]
            else:
                self.score_pool3 = nn.Conv2d(self.score_channels[1], self.n_classes, 1, bias=False)
                if not random:
                    self.score_pool3.weight.data = weight3
                else:
                    self.score_pool3.weight.data[:-1, ...] = copy.deepcopy(self.original_weights[2])
                    self.score_pool3.weight.data = self.score_pool3.weight.data.cuda()

                assert self.classifier[2].weight.is_cuda
                assert self.score_pool3.weight.is_cuda
                assert self.score_pool4.weight.is_cuda
                assert self.score_pool3.weight.data.shape[1] == self.score_channels[1]
                assert self.classifier[2].weight.data.shape[1] == 256
                assert self.score_pool4.weight.data.shape[1] == self.score_channels[0]

    def save_original_weights(self):
        self.original_weights = []
        if self.use_normalize_train:
            self.original_weights.append(copy.deepcopy(self.classifier[2].conv.weight.data))
            self.original_weights.append(copy.deepcopy(self.score_pool4.conv.weight.data))
            self.original_weights.append(copy.deepcopy(self.score_pool3.conv.weight.data))
        else:
            self.original_weights.append(copy.deepcopy(self.classifier[2].weight.data))
            self.original_weights.append(copy.deepcopy(self.score_pool4.weight.data))
            self.original_weights.append(copy.deepcopy(self.score_pool3.weight.data))


    def reverse_imprinting(self):
        if self.use_normalize_train:
            nchannels = self.classifier[2].conv.weight.data.shape[1]
            print('No Continual Learning for Bg')
            self.n_classes = 16
            self.classifier[2] = CosineSimLayer(nchannels, self.n_classes, 1, bias=False)
            self.classifier[2].conv.weight.data = copy.deepcopy(self.original_weights[0])

            self.score_pool4 = CosineSimLayer(self.score_channels[0], self.n_classes, 1, bias=False)
            self.score_pool4.conv.weight.data = copy.deepcopy(self.original_weights[1])

            self.score_pool3 = CosineSimLayer(self.score_channels[1], self.n_classes, 1, bias=False)
            self.score_pool3.conv.weight.data = copy.deepcopy(self.original_weights[2])

            assert self.score_pool3.conv.weight.data.shape[1] == self.score_channels[1]
            assert self.classifier[2].conv.weight.data.shape[1] == 256
            assert self.score_pool4.conv.weight.data.shape[1] == self.score_channels[0]
        else:
            nchannels = self.classifier[2].weight.data.shape[1]
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

    def iterative_imprinting(self, sprt_images, qry_images, sprt_labels,
                             alpha, itr=1):
        self.imprint(sprt_images, sprt_labels, alpha=alpha)

        self.eval()
        it_alpha = 0.1
        outputs = self(qry_images)
        pseudo = self.gen_pseudo(outputs)
        self.imprint([qry_images], [pseudo], alpha=it_alpha, new_class=False)

        # to overcome incorrect predictions in pseudolabel previously around boundary
        pseudo = torch.zeros(pseudo.shape)
        self.imprint([qry_images], [pseudo], alpha=it_alpha, new_class=False)

    def gen_pseudo(self, preds):
        pseudo = 250*torch.zeros((preds.shape[0], preds.shape[2], preds.shape[3]))
        preds = nn.Softmax(dim=1)(preds)
        pseudo[preds[:,-1,:,:]>0.7] = 16
        pseudo[preds[:,-1,:,:]<0.3] = 0
        return pseudo

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

    def freeze_weights_extractor(self):
        freeze_weights(self.conv_block1)
        freeze_weights(self.conv_block2)
        freeze_weights(self.conv_block3)
        freeze_weights(self.conv_block4)
        freeze_weights(self.conv_block5)

    def freeze_all_except_classifiers(self):
        self.freeze_weights_extractor()
        freeze_weights(self.fconv_block)
