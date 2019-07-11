import collections
import torch
import torchvision
import numpy as np
import scipy.misc as m
import matplotlib.pyplot as plt

from torch.utils import data

from ptsemseg.utils import recursive_glob
import scipy.io

class ADE20KLoader(data.Dataset):
    def __init__(
        self,
        root,
        split="training",
        is_transform=False,
        img_size=512,
        augmentations=None,
        img_norm=True,
        test_mode=False,
        n_classes=151,
        fold=None
    ):
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.test_mode = test_mode
        self.n_classes = n_classes
        self.fold = fold

        ignore_dict = {0: [91, 128, 127, 77, 99], 1:[], 2:[], 3:[]}
        self.ignore_classes = ignore_dict[self.fold]

        self.img_size = img_size if isinstance(img_size, list) else (img_size, img_size)
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.files = collections.defaultdict(list)
        self.class_names = self.parse_classes(self.root+'classes.txt')

        if not self.test_mode:
            for split in ["training", "validation"]:
                file_list = recursive_glob(
                    rootdir=self.root + "images/" + self.split + "/", suffix=".jpg"
                )
                self.files[split] = file_list

    def parse_classes(self, pth):
        classes = ['Background']
        f = open(pth, 'r')
        count = 0
        for line in f:
            if count == 0:
                count += 1
                continue
            classes.append(line.split(',')[-1].strip())
        return classes

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_path = self.files[self.split][index].rstrip()
        lbl_path = img_path.replace("jpg", "png")
        lbl_path = lbl_path.replace("images", "annotations")

        img = m.imread(img_path)
        img = np.array(img, dtype=np.uint8)
        lbl = m.imread(lbl_path)
        lbl = np.array(lbl, dtype=np.int32)

        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl

    def transform(self, img, lbl):
        if self.img_size[0] != 'same':
            if img.shape[0] != img.shape[1]:
                # preserve aspect ratio
                preserve_aspect_ratio = True
                img = self.resize_preserving_aspect(img)
            else:
                preserve_aspect_ratio = False
                img = m.imresize(img, (self.img_size[0], self.img_size[1]))  # uint8 with RGB mode

        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean
        if self.img_norm:
            # Resize scales images from 0 to 255, thus we need
            # to divide by 255.0
            img = img.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        #lbl = self.encode_segmap(lbl)
        lbl = self.filter_seg(lbl)
        classes = np.unique(lbl)
        lbl = lbl.astype(float)
        if self.img_size[0] != 'same':
            if preserve_aspect_ratio:
                lbl = self.resize_preserving_aspect(lbl, lbl_flag=True)
                classes = np.unique(lbl)
            else:
                lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), "nearest", mode="F")
        lbl = lbl.astype(int)
        assert np.all(classes == np.unique(lbl))

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def resize_preserving_aspect(self, x, lbl_flag=False):
        if x.shape[0] < x.shape[1]:
            img_ht = int(x.shape[1] * self.img_size[0]/x.shape[0])
            if lbl_flag:
                x = m.imresize(x, (self.img_size[0], img_ht), "nearest", mode="F")
            else:
                x = m.imresize(x, (self.img_size[0], img_ht))  # uint8 with RGB mode
            x = x[:self.img_size[0], :self.img_size[0]]
        else:
            img_wd = int(x.shape[0] * self.img_size[1]/x.shape[1])
            if lbl_flag:
                x = m.imresize(x, (img_wd, self.img_size[1]), "nearest", mode="F")
            else:
                x = m.imresize(x, (img_wd, self.img_size[1]))  # uint8 with RGB mode
            x = x[:self.img_size[1], :self.img_size[1]]
        return x

    def filter_seg(self, label_mask):
        if self.fold is not None: # Some classes are ignored for OSLSM Training
            class_count = 0
            for c in range(151):
                if c in self.ignore_classes:
                    label_mask[label_mask == c] = 250
                else:
                    label_mask[label_mask == c] = class_count
                    class_count += 1

            if label_mask[label_mask!=250].sum() == 0: # Images with only background and ignored arent used
                label_mask[label_mask != 250] = 250
        return label_mask

    def encode_segmap(self, mask):
        # Refer : http://groups.csail.mit.edu/vision/datasets/ADE20K/code/loadAde20K.m
        mask = mask.astype(np.uint16)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int64)
        label_mask = (mask[:, :, 0] / 10.0) * 256 + mask[:, :, 1]
        return np.array(label_mask, dtype=np.int64)

    def decode_segmap(self, temp, plot=False):
        # TODO:(@meetshah1995)
        # Verify that the color mapping is 1-to-1
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = 10 * (l % 10)
            g[temp == l] = l
            b[temp == l] = 0

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb


if __name__ == "__main__":
    local_path = "/home/menna/Datasets/ADEChallengeData2016/"
    dst = ADE20KLoader(local_path, is_transform=True, fold=0, img_size=[512,512])
    trainloader = data.DataLoader(dst, batch_size=4, num_workers=0)
    for i, data_samples in enumerate(trainloader):
        imgs, labels = data_samples
        img = torchvision.utils.make_grid(imgs).numpy()
        img = np.transpose(img, (1, 2, 0))
        img = img[:, :, ::-1]
        plt.imshow(img)
        plt.show()
        #    for j in range(4):
        #        plt.imshow(labels.numpy()[j])
        #        plt.show()
