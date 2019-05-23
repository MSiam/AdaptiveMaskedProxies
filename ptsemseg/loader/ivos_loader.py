from torch.utils import data
from torchvision import transforms
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import random

class IVOSLoader(data.Dataset):
    '''
    Creates instance of IVOSLoader
    Args:
        root (str): main directory of dataset path
        split (str): either 'same_trans', 'cross_trans', 'cross_domain'
        n_classes (int)
    '''
    def __init__(self, root, split='same_trans', n_classes=3, kshot=1):
        self.split = split
        self.root = root
        self.n_classes = n_classes
        self.kshot = kshot
        self.nsamples = 1000

        self.rand_gen = random.Random()
        self.rand_gen.seed(1385)

        self.transformations = ['Translation', 'Scale', 'Rotation']
        self.classes = ['bowl', 'bottle', 'mug']

        # create paths files returns dictionary
        # key : transformation, value : dictionary K:category,V:paths
        self.files_path = self.parse_paths()

        # Create support and query pairs randomly sampled
        self.pairs = self.create_pairs(self.rand_gen, self.split,
                                       self.files_path)

    def create_pairs(self, rand_gen, split, paths):
        pairs = []

        for i in range(self.nsamples):
            if split == 'same_trans':
                # 1 - Pick randomly transformation
                rand_gen.shuffle(self.transformations)
                rnd_transf = self.transformations[0]

                # 2 - Pick randomly category
                rand_gen.shuffle(self.classes)
                rnd_category = self.classes[0]

                # 3 - Pick randomly support set poses
                rand_gen.shuffle(paths[rnd_transf][rnd_category])
                support = paths[rnd_transf][rnd_category][:self.kshot]

                # 4 - Pick query set poses
                query = paths[rnd_transf][rnd_category][self.kshot:]

            elif split == 'cross_trans':
                support = None; query = None
            else:
                support = None; query = None
            pairs.append((support, query))

        return pairs

    def parse_paths(self):
        paths = {}

        for transf in self.transformations:

            transf_path = self.root + transf + '/Images/'
            dirs = os.listdir(transf_path)

            category_paths = {}
            for d in dirs:
                if d[:-1] in self.classes:
                    if d[:-1] not in category_paths:
                        category_paths[d[:-1]] = []

                    current_path = transf_path + d

                    for f in os.listdir(current_path):
                        category_paths[d[:-1]].append(current_path + '/' + f)

            paths[transf] = category_paths

        return paths

    def __get_item(self, index):
        pass

if __name__ == "__main__":
    # Testing the ivos loader
    loader = IVOSLoader('/home/menna/Datasets/IVOS_dataset/')


