import os
from os.path import join as pjoin
import collections
import json
import torch
import numpy as np
import scipy.misc as m
import scipy.io as io
import matplotlib.pyplot as plt
import glob

from PIL import Image
from tqdm import tqdm
from torch.utils import data
from torchvision import transforms
import cv2
from ptsemseg.loader.pascal_voc_loader import pascalVOCLoader

class pascalVOCIgnoreLoader(pascalVOCLoader):
    def __init__(
        self,
        root,
        split="train_aug",
        is_transform=False,
        img_size=512,
        augmentations=None,
        img_norm=True,
        fold=None,
        n_classes=21,
        ignore=None
    ):
        self.ignore = ignore
        super(pascalVOCIgnoreLoader, self).__init__(root, split=split,
                                  is_transform=is_transform, img_size=img_size,
                                  augmentations=augmentations, img_norm=img_norm,
                                  n_classes=n_classes)

    def __getitem__(self, index):
        im_name = self.files[self.split][index]
        im_path = pjoin(self.root, "JPEGImages", im_name + ".jpg")

        if self.ignore is not None:
            lbl_path = pjoin(self.root, "SegmentationClass/pre_encoded_ignore_"+self.ignore, im_name + ".png")
        else:
            raise("Unknown ignore class")

        im = Image.open(im_path)
        lbl = Image.open(lbl_path)
        if self.augmentations is not None:
            im, lbl = self.augmentations(im, lbl)
        if self.is_transform:
            im, lbl = self.transform(im, lbl)
        return im, lbl

    def filter_seg(self, fold, label_mask):
        pascal_lbls = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                       'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                       'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']
        ignore_class = pascal_lbls.index(self.ignore)
        class_count = 0
        for c in range(21):
            if c == ignore_class:
                label_mask[label_mask == c] = 250
            else:
                label_mask[label_mask == c] = class_count
                class_count += 1

        if label_mask[label_mask!=250].sum() == 0: # Images with only background and ignored arent used
            label_mask[label_mask != 250] = 250
        return label_mask

    def setup_annotations(self, target_path=None):
        """Sets up Berkley annotations by adding image indices to the
        `train_aug` split and pre-encode all segmentation labels into the
        common label_mask format (if this has not already been done). This
        function also defines the `train_aug` and `train_aug_val` data splits
        according to the description in the class docstring
        """
        if self.ignore is not None:
            target_path = pjoin(self.root, "SegmentationClass/pre_encoded_ignore_"+self.ignore)
            self.fold = 0

        super(pascalVOCIgnoreLoader, self).setup_annotations(target_path=target_path)

if __name__ == "__main__":
    from pascal_voc_loader import pascalVOCLoader
    loader = pascalVOCIgnoreLoader('/home/menna/Datasets/VOCdevkit/VOC2012/', n_classes=20, ignore='bottle',
                                   is_transform=True)
    for img, lbl in loader:
        plt.figure(1); plt.imshow(np.transpose(img, (1,2,0)))
        plt.figure(2); plt.imshow(lbl); plt.show()

