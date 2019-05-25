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
    def __init__(self, root, split='same_trans', n_classes=3,
                 kshot=1, img_size='same', is_transform=True):
        self.split = split
        self.root = root
        self.n_classes = n_classes
        self.kshot = kshot
        self.img_size = (
            img_size if isinstance(img_size, tuple) else (img_size, img_size)
        )

        self.nsamples = 1000
        self.is_transform = is_transform
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
        self.tf = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])


    def create_pairs(self, rand_gen, split, paths):
        pairs = []

        for i in range(self.nsamples):
            # shuffle categories
            rand_gen.shuffle(self.classes)

            # Pick randomly transformation
            rand_gen.shuffle(self.transformations)
            rnd_transf = self.transformations[0]

            support = []
            query = []
            for cls in self.classes:
                if split == 'same_trans':
                    # Pick randomly support set poses
                    rand_gen.shuffle(paths[rnd_transf][cls])
                    support.append(paths[rnd_transf][cls][:self.kshot])

                    # Pick query set poses
                    query.append(paths[rnd_transf][cls][self.kshot:])

                elif split == 'cross_trans':
                    # Pick randomly support set poses
                    rand_gen.shuffle(paths[rnd_transf][cls])
                    support.append(paths[rnd_transf][cls][:self.kshot])

                    # Pick query set poses
                    cross_transf = self.transformations[1]
                    query.append(paths[cross_transf][cls])
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

    def transform(self, img, lbl, cls_idx):
        if self.img_size == ('same', 'same'):
            pass
        elif hasattr(img, 'dtype'):
            img = cv2.resize(img, self.img_size)
            lbl = cv2.resize(lbl, self.img_size, interpolation=cv2.INTER_NEAREST)
        else:
            img = img.resize((self.img_size[0], self.img_size[1]))  # uint8 with RGB mode
            lbl = lbl.resize((self.img_size[0], self.img_size[1]))

        img = self.tf(img)
        lbl = torch.from_numpy(np.array(lbl)).long()
        lbl[lbl == 255] = cls_idx
        return img, lbl


    def read_imgs_lbls(self, current_set):

        all_imgs = []
        all_lbls = []
        for i in range(len(self.classes)):
            imgs = []
            lbls = []

            for j in range(self.kshot):
                img_path = current_set[i][j]
                img = cv2.imread(img_path)

                lbl_path = img_path.replace('Images', 'Masks')
                lbl = cv2.imread(lbl_path, 0)

                if self.is_transform:
                    img, lbl = self.transform(img, lbl, i+1)

                imgs.append(img)
                lbls.append(lbl)

            all_imgs.append(imgs)
            all_lbls.append(lbls)

        return all_imgs, all_lbls

    def __getitem__(self, index):
        support, query = self.pairs[index]

        sprt_imgs, sprt_lbls = self.read_imgs_lbls(support)
        qry_imgs, qry_lbls = self.read_imgs_lbls(query)

        return sprt_imgs, sprt_lbls, qry_imgs, qry_lbls

if __name__ == "__main__":
    # Testing the ivos loader
    loader = IVOSLoader('/home/menna/Datasets/IVOS_dataset/', split='same_trans')

    for sprt_imgs, sprt_lbls, qry_imgs, qry_lbls in loader:
        plt.figure(0); plt.imshow(np.transpose(sprt_imgs[0][0], (1,2,0)))
        plt.figure(1); plt.imshow(sprt_lbls[0][0]); plt.show()

        plt.figure(0); plt.imshow(np.transpose(qry_imgs[0][0], (1,2,0)))
        plt.figure(1); plt.imshow(qry_lbls[0][0]); plt.show()

