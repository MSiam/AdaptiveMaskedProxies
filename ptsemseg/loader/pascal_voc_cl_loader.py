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
import pickle
import copy

def get_data_path(name):
    """Extract path to data from config file.

    Args:
        name (str): The name of the dataset.

    Returns:
        (str): The path to the root directory containing the dataset.
    """
    js = open("config.json").read()
    data = json.loads(js)
    return os.path.expanduser(data[name]["data_path"])


class ipascalVOCLoader(pascalVOCLoader):
    """Data loader for the Pascal VOC semantic segmentation dataset.

    Annotations from both the original VOC data (which consist of RGB images
    in which colours map to specific classes) and the SBD (Berkely) dataset
    (where annotations are stored as .mat files) are converted into a common
    `label_mask` format.  Under this format, each mask is an (M,N) array of
    integer values from 0 to 21, where 0 represents the background class.

    The label masks are stored in a new folder, called `pre_encoded`, which
    is added as a subdirectory of the `SegmentationClass` folder in the
    original Pascal VOC data layout.

    A total of five data splits are provided for working with the VOC data:
        train: The original VOC 2012 training data - 1464 images
        val: The original VOC 2012 validation data - 1449 images
        train_aug: The unique images present in both the train split and
                   training images from SBD: - 8829 images (the unique members
                   of the result of combining lists of length 1464 and 8498)
        train_cl: CL mode for training set.
        val_cl: CL mode for validation set.
    """

    def __init__(
        self,
        root,
        split="train_aug",
        is_transform=False,
        img_size=512,
        augmentations=None,
        img_norm=True,
        n_classes=21,
        fold=None,
        n_tasks=5,
        classes_train=None,
        classes_incremental=None
    ):
        super(ipascalVOCLoader, self).__init__(root, split, is_transform,
                                         img_size,augmentations, img_norm,
                                         None, 21)
        self.root = os.path.expanduser(root)
        self.fold = fold
        self.n_tasks = n_tasks
        self.batches = None

        self.current_task = None
        self.current_batch = None

        # During Training mode before CL these splits are randomly generated.
        # During CL mode they are set with the previously generated splits loaded from pickle.
        self.nclasses_inc = (self.n_classes - 1) // 2
        if 'CL' not in self.split:
            if 'train' in self.split:
                self.classes_train = np.arange(1, self.n_classes)
                np.random.shuffle(self.classes_train)
                self.classes_train = self.classes_train[:self.nclasses_inc]
                self.classes_incremental = np.arange(1, self.n_classes)
                remove_inds = []
                for i in range(self.classes_incremental.shape[0]):
                    if self.classes_incremental[i] in self.classes_train:
                        remove_inds.append(i)

                self.classes_incremental = np.delete(self.classes_incremental, remove_inds)
            else:
                self.classes_train = classes_train
                self.classes_incremental = classes_incremental

        self.files_dict = {}
        self.split_data()

        self.batches_path_pre = pjoin(self.root, "SegmentationClass/pre_encoded_" + str(self.fold),
                                      "batches" + self.split.replace('_CL', ''))

        if 'CL' not in self.split:
            self.ignore_classes = self.classes_incremental
            self.setup_annotations()

            self.batches = self.create_batches()

            print('train classes ', self.classes_train)
            print('incremental classes ', self.classes_incremental)

    def __len__(self):
        if 'CL' in self.split:
            if 'train' in self.split:
                return 470
            if self.current_batch is None:
               f = open(self.batches_path_pre+'_'+str(self.current_task)+'.pkl', 'rb')
               self.current_batch = pickle.load(f)
            return self.current_batch[2].shape[0]
        else:
            return len(self.files[self.split])

    def map_labels(self, lbl, class_map):
        lbl = np.array(lbl.copy())
        mapped_lbl = np.zeros_like(lbl)

        for i in range(len(class_map)):
            if class_map[i] in lbl:
                mapped_lbl[lbl==class_map[i]] = i+1
        mapped_lbl[lbl==250] = 250
        return mapped_lbl

    def process_batch(self, im, lbl, classes):
        return im, lbl

    def __getitem__(self, index):
        if 'CL' not in self.split:
            # Regular mode
            im_name = self.files[self.split][index]
            im_path = pjoin(self.root, "JPEGImages", im_name + ".jpg")
            lbl_path = pjoin(self.root, "SegmentationClass/pre_encoded_"+str(self.fold), im_name + ".png")

            #print('getting item ', im_path,' + ' , lbl_path)
            im = Image.open(im_path)
            lbl = Image.open(lbl_path)
            lbl = self.map_labels(lbl, self.classes_train)
            if self.augmentations is not None:
                im, lbl = self.augmentations(im, lbl)
            if self.is_transform:
                im, lbl = self.transform(im, lbl)
            return im, lbl
        else:
            # CL model
            if self.current_batch is None:
                f = open(self.batches_path_pre+'_'+str(self.current_task)+'.pkl', 'rb')
                self.current_batch = pickle.load(f)

            taski, classes, ims, lbls = self.current_batch
            im = ims[index]
            lbl = lbls[index]
            if self.is_transform:
                lbl = self.map_labels(lbl, classes)
                im, lbl = self.transform(im[:, :, ::-1], lbl)
            im = im.unsqueeze(0)
            lbl = lbl.unsqueeze(0)
            return taski, classes, im, lbl

    def split_data(self):
         #Method to create dictionary with classes as keys and filepaths as vals
        for k, v in self.files.items():
            self.files_dict[k] = {}

            for c in range(1, self.n_classes):
                self.files_dict[k][c] = []

            for f in self.files[k]:
                lbl_path = pjoin(self.root, "SegmentationClass/pre_encoded", f + ".png")
                lbl = cv2.imread(lbl_path, 0)

                for c in range(1, self.n_classes):
                    if c in lbl:
                        self.files_dict[k][c].append(f)

    def create_batches(self):
        cpt = self.nclasses_inc//self.n_tasks
        current_classes = self.classes_train
        for t in range(self.n_tasks):
            if os.path.exists(self.batches_path_pre+'_'+str(t)+'.pkl'):
                print('Batch ', t, ' exists')
                continue

            current_classes = np.concatenate((current_classes,
                                              self.classes_incremental[t*cpt:cpt*(t+1)]))
            imgs = []; lbls = []
            for c in current_classes[-2:]:
                for f in self.files_dict[self.split][c]:
                    im_path = pjoin(self.root, "JPEGImages", f + ".jpg")
                    lbl_path = pjoin(self.root, "SegmentationClass/pre_encoded", f + ".png")

                    imgs.append(cv2.imread(im_path))

                    lbl = cv2.imread(lbl_path, 0)
                    lbl = self.filter_seg(self.classes_incremental[cpt*(t+1)+1:], lbl)
                    lbls.append(lbl)

            batch = ( t, current_classes, np.array(imgs), np.array(lbls) )
            print('saved batch ', t)
            f = open(self.batches_path_pre+'_'+str(t)+'.pkl', 'wb')
            pickle.dump(batch, f)
