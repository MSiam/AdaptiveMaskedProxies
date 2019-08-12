import copy
import numpy as np
import scipy.io
import cv2
import os
import random
from torch.utils import data
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
import pickle

class lfwPARTSLoader(object):
    def __init__(self,
                root,
                imgs_root,
                is_transform=True,
                img_size=512,
                k_shot=1,
                base_n_classes=21):

        self.is_transform = is_transform
        self.imgs_root = imgs_root
        self.img_size = img_size
        if type(self.img_size) != tuple:
            self.img_size = (img_size, img_size)
        self.k_shot = k_shot
        self.root = root
        self.base_n_classes = base_n_classes

        self.rand_gen = random.Random()
        self.rand_gen.seed(1385) # For Deterministic Mode
        self.nsamples = 1000
        self.files = os.listdir(self.root)
        self.pairs = self.sample_pairs(self.files)
        self.tf = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    def sample_pairs(self, files):
        pairs = []
        for i in range(self.nsamples):
            sprt_files = []
            for k in range(self.k_shot):
                fname = self.rand_gen.choice(files)
                sprt_files.append(fname)

            qry = self.rand_gen.choice(files)
            pairs.append((sprt_files, qry))
        return pairs

    def transform(self, img, lbl):
        if self.img_size == ('same', 'same'):
            pass
        else:
            img = cv2.resize(img, self.img_size)
            lbl = cv2.resize(lbl, self.img_size,
                             interpolation=cv2.INTER_NEAREST)

        img = self.tf(img)
        lbl = torch.from_numpy(np.array(lbl)).long()
        return img, lbl

    def __len__(self):
        return self.nsamples

    def convert_label(self, label):
        label[label<76] = 0
        label[label==150] = 1
        label[label==76] = 2
        return label

    def read_imgs(self, fname):
        annot_path = self.root + fname

        tkns = fname.split('_')[:-1]
        img_pth = ''
        for t in tkns:
            img_pth += t+'_'

        img_pth = self.imgs_root + img_pth[:-1] + '/' + fname.split('.')[0] + '.jpg'

        img = cv2.imread(img_pth)
        original_img = img.copy()

        annot = cv2.imread(annot_path, 0)
        annot = self.convert_label(annot)

        if self.is_transform:
            img, annot = self.transform(img, annot)

        return img, annot, original_img

    def __getitem__(self, index):
        pair = self.pairs[index]
        sprt_imgs = []
        sprt_lbls = []
        sprt_originals = []

        for fname in pair[0]:
            img, lbl, original_img = self.read_imgs(fname)
            sprt_imgs.append(img)
            sprt_lbls.append(lbl)
            sprt_originals.append(original_img)

        qry_img, qry_lbl, qry_original = self.read_imgs(pair[1])
        return sprt_imgs, sprt_lbls, sprt_originals, \
                    qry_img, qry_lbl, qry_original

def test_loader():
    loader = lfwPARTSLoader('/home/menna/Datasets/LFW/parts_lfw_funneled_gt_images/',
                               '/home/menna/Datasets/LFW/lfw_funneled/',
                               k_shot=1)

    testloader = data.DataLoader(loader, batch_size=1)

    for batch in testloader:
        plt.figure(1);plt.imshow(batch[2][0][0])
        plt.figure(2);plt.imshow(batch[1][0][0])

        plt.figure(4);plt.imshow(batch[5][0])
        plt.figure(5);plt.imshow(batch[4][0])
        plt.show()

if __name__=="__main__":
    test_loader()

