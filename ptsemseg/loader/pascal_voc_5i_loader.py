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
        k_shot=1
    ):
        super(pascalVOC5iLoader, self).__init__(root, split=split,
                                          is_transform=is_transform, img_size=img_size,
                                          augmentations=augmentations, img_norm=img_norm,
                                          n_classes=n_classes)

        with open('ptsemseg/loader/oslsm/profile.txt', 'r') as f:
            profile = str(f.read())
            profile = self.convert_d(profile)

        profile['pascal_path'] = self.root
        profile['areaRng'][1] = float('Inf')

        pascal_lbls = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                       'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                       'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

        profile['pascal_cats'] = []
        for i in range(fold*5+1, (fold+1)*5+1):
            profile['pascal_cats'].append(pascal_lbls[i])

        profile['k_shot'] = k_shot
        profile_copy = profile.copy()
        profile_copy['first_label_params'].append(('original_first_label', 1.0, 0.0))
        profile_copy['deploy_mode'] = True

        dbi = ss_datalayer.DBInterface(profile, fold=fold, binary=binary)
        self.PLP = ss_datalayer.PairLoaderProcess(None, None, dbi, profile_copy)

        self.oslsm_files = self.parse_file('ptsemseg/loader/imgs_paths_%d_%d.txt'%(fold, k_shot),
                                           k_shot)
        self.prefix_lbl = 'SegmentationClass/pre_encoded/'
        self.current_fold = fold

    def parse_file(self, pth_txt, k_shot):
        files = []
        pair = []
        support = []
        f = open(pth_txt, 'r')
        count = 0
        for line in f:
            if count == (k_shot+1)*2:
                pair.append(line.split(' ')[-1].strip())
                files.append(pair)
                count = -1
            elif count < k_shot:
                support.append(line.strip())
            elif count < k_shot+1:
                pair = [support, line.strip()]
                support = []
            count += 1
        return files

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
        return 1000 #len(self.PLP.db_interface.db_items)

    def map_labels(self, lbl, cls_idx):
        ignore_classes = range(self.current_fold*5+1, (self.current_fold+1)*5+1)
        class_count = 0
        for c in range(21):
            if c not in ignore_classes:
                lbl[lbl == c] = class_count
                class_count += 1
        lbl[lbl==cls_idx] = 16
        return lbl

    def __getitem__(self, index):
        pair = self.oslsm_files[index]
        #self.out = self.PLP.load_next_frame(try_mode=False)
        original_im1 = []
        im1 = []
        lbl1= []

#        original_im2 = self.out['second_img'][0]
        original_im2 = cv2.imread(pair[1])
        original_im2 = cv2.resize(original_im2, self.img_size)

        im2 = np.asarray(cv2.imread(pair[1]), dtype=np.float32)
        lbl2 = cv2.imread(pair[1].replace('JPEGImages', self.prefix_lbl).replace('jpg', 'png') , 0)
        lbl2 = np.asarray(lbl2, dtype=np.int32)
        lbl2 = self.map_labels(lbl2, int(pair[-1]))

        im2, lbl2 = self.transform(im2, lbl2)

        for j in range(len(pair[0])):
            img = cv2.imread(pair[0][j])
            img = cv2.resize(img, self.img_size)
            original_im1.append(img)
            im1.append(np.asarray(cv2.imread(pair[0][j]), dtype=np.float32))
            temp_lbl = cv2.imread(pair[0][j].replace('JPEGImages', self.prefix_lbl).replace('jpg', 'png') , 0)
            temp_lbl = self.map_labels(temp_lbl, int(pair[-1]))
            lbl1.append(np.asarray(temp_lbl, dtype=np.int32))

            if self.is_transform:
                im1[j], lbl1[j] = self.transform(im1[j], lbl1[j])
        return im1, lbl1, im2, lbl2, original_im1, original_im2, int(pair[-1])#self.out['cls_ind']

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
