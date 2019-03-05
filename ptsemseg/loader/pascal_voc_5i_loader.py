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
from ptsemseg.loader.oslsm import ss_datalayer
from ptsemseg.loader import pascalVOCLoader
import yaml
import cv2

class pascalVOC5iLoader(pascalVOCLoader):
    """Data loader for the Pascal VOC 5i Few-shot semantic segmentation dataset.
    """

    def __init__(
        self,
        root,
        split="val",
        is_transform=False,
        img_size=512,
        augmentations=None,
        img_norm=True,
        n_classes=15,
        fold=0,
        binary=False,
        k_shot=1,
        hparam_search=False
    ):
        super(pascalVOC5iLoader, self).__init__(root, split=split,
                                          is_transform=is_transform, img_size=img_size,
                                          augmentations=augmentations, img_norm=img_norm,
                                          n_classes=n_classes, hparam_search=hparam_search)

        with open('/home/eren/Work/AdaptiveMaskedImprinting/ptsemseg/loader/oslsm/profile.txt', 'r') as f:
            profile = str(f.read())
            profile = self.convert_d(profile)

        profile['pascal_path'] = self.root
        profile['sbd_path'] = self.sbd_path

        profile['areaRng'][1] = float('Inf')

        pascal_lbls = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                       'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                       'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']
        profile['pascal_cats'] = []

        if split == "val":
            image_sets = ["pascal_val"]
            profile['output_type'] = 'image_pair'
            if hparam_search:
                mapped_folds = {0: 1, 1: 0, 2: 3, 3: 2}
                fold = mapped_folds[fold]

            fold_range = range(fold*5+1, (fold+1)*5+1)
        else:
            image_sets = ["pascal_train"]
            profile['output_type'] = 'image_pair_train'
            excluded_range = range(fold*5+1, (fold+1)*5+1)
            fold_range = list(range(1,21))
            for ex in excluded_range:
                fold_range.remove(ex)

        profile['image_sets'] = image_sets
        for i in fold_range:
            profile['pascal_cats'].append(pascal_lbls[i])

        profile['k_shot'] = k_shot
        profile_copy = profile.copy()
        profile_copy['first_label_params'].append(('original_first_label', 1.0, 0.0))
        profile_copy['deploy_mode'] = True

        dbi = ss_datalayer.DBInterface(profile, fold=fold, binary=binary, hparam_search=hparam_search)
        self.PLP = ss_datalayer.PairLoaderProcess(None, None, dbi, profile_copy)

    def convert_d(self, string):
        s = string.replace("{" ,"")
        finalstring = s.replace("}" , "")
        list = finalstring.split(";")

        dictionary ={}
        for i in list:
            keyvalue = i.split(":")
            m = eval(keyvalue[0])
            dictionary[m] = eval(keyvalue[1])
        return dictionary

    def __len__(self):
        if self.hparam_search:
            return 200
        elif self.split == 'val':
            return 1000
        else:
            return len(self.PLP.db_interface.db_items)

    def __getitem__(self, index):
        self.out = self.PLP.load_next_frame(try_mode=False)
        original_im1 = []
        im1 = []
        lbl1= []

        original_im2 = self.out['second_img'][0]
        original_im2 = cv2.resize(original_im2, self.img_size)

        im2 = np.asarray(self.out['second_img'][0], dtype=np.float32)
        lbl2 = np.asarray(self.out['second_label'][0], dtype=np.int32)
        im2, lbl2 = self.transform(im2, lbl2)

        for j in range(len(self.out['first_img'])):

            img = cv2.resize(self.out['first_img'][j], self.img_size)
            original_im1.append(img)
            im1.append(np.asarray(self.out['first_img'][j], dtype=np.float32))
            lbl1.append(np.asarray(self.out['first_label'][j], dtype=np.int32))

            if self.is_transform:
                im1[j], lbl1[j] = self.transform(im1[j], lbl1[j])

        return im1, lbl1, im2, lbl2, original_im1, original_im2

    def correct_im(self, im):
        im = (np.transpose(im, (0,2,3,1)))/255.
        im += np.array([0.40787055,  0.45752459,  0.4810938])
        return im[:,:,:,::-1]
    # returns the outputs as images and also the first label in original img size

    def get_items_im(self):
        self.out = self.PLP.load_next_frame(try_mode=False)
        return (self.correct_im(self.out['first_img']),
                self.out['original_first_label'],
                self.correct_im(self.out['second_img']),
                self.out['second_label'][0],
                self.out['deploy_info'])
