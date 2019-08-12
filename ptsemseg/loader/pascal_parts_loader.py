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

class pascalPARTSLoader(object):
    def __init__(self,
                root,
                imgs_root,
                is_transform=False,
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
        self.pmap_dict, _, self.classes_noparts = self.load_parts_map()
        self.root += 'Annotations_Part/'

        self.rand_gen = random.Random()
        self.rand_gen.seed(1385) # For Deterministic Mode
        self.nsamples = 1000
        self.files = self.load_annotation_files()
        self.tf = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    def sample_pairs(self, group_files):
        nsamples_per_cls = len(self) // (self.base_n_classes - 1)
        pairs = []
        for i in range(1, self.base_n_classes):
            if i in self.classes_noparts:
                continue

            for j in range(nsamples_per_cls):
                sprt_files = []
                for k in range(self.k_shot):
                    fname = self.rand_gen.choice(group_files[i])
                    sprt_files.append(fname)

                qry = self.rand_gen.choice(group_files[i])
                pairs.append((sprt_files, qry, i))
        self.nsamples = len(pairs)
        print('Final # of samples ', self.nsamples)
        return pairs

    def load_annotation_files(self):
        files = os.listdir(self.root)
        group_files = [ [] for i in range(self.base_n_classes) ]
        if os.path.exists(self.root + 'group.pkl'):
            pkl_f = open(self.root + 'group.pkl', 'rb')
            group_files = pickle.load(pkl_f)

        else:
            for f in sorted(files):
                print('Loading file ', f)
                annot = scipy.io.loadmat(self.root + f)
                if not os.path.exists(self.imgs_root +
                                      f.split('.')[0] + '.jpg'):
                    continue

                __, clsi, _, parts_dict, ninst = self.parse_annot(annot)
                if ninst == 0:
                    continue
                for cls, part in zip(clsi,parts_dict):
                    if len(part.keys()) != 0:
                        class_ind = cls
                        break
                group_files[class_ind].append(f)

            pkl_f = open(self.root + 'group.pkl', 'wb')
            pickle.dump(group_files, pkl_f)

        pairs = self.sample_pairs(group_files)
        self.rand_gen.shuffle(pairs)
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


    def parse_annot(self, annot):

        small_annot = annot['anno'][0][0][1][0]

        class_labels = []
        class_inds = []
        masks = []
        parts_dicts = []
        ninst = 0
        for inst in small_annot:
            class_labels.append(inst[0][0])
            class_inds.append(inst[1][0][0])

            mask = inst[2]
            mask[mask == 1] = class_inds[-1]
            masks.append(mask)

            parts_dicts.append({})

            if len(inst[3]) == 0:
                continue

            ninst += 1
            parts = inst[3][0]

            for element in parts:
                parts_dicts[-1][str(element[0][0])] = \
                        element[1]

        return class_labels, class_inds, masks, parts_dicts, ninst

    def load_parts_map(self):
        parts_map = [{} for i in range(self.base_n_classes)]
        max_n_classes = -1

        classes_to_remove = list(range(self.base_n_classes))
        f = open(self.root + 'parts_map.txt', 'r')
        for line in f:
            tkns = line.split(', ')

            class_id = int(tkns[0])
            if class_id in classes_to_remove:
                classes_to_remove.remove(class_id)
            part_name = tkns[1].strip()

            if part_name.isdigit():
                parts_map[class_id] = copy.deepcopy(parts_map[int(part_name)])

                # handle if there are exceptions
                for i in range(0, len(tkns[2:]), 2):
                    new_part, new_idx = tkns[2+i], int(tkns[2+i+1])
                    if new_idx == -1:
                        del parts_map[class_id][new_part]
                    else:
                        parts_map[class_id][new_part] = new_idx
            elif '%' in part_name:
                part_max_idx = int(tkns[2])
                part_start_idx = int(tkns[3])
                part_name = part_name.split('%')[0]
                for i in range(1, part_max_idx+1):
                    parts_map[class_id][part_name+'%d'%i] = \
                            part_start_idx + i
            else:
                part_idx = int(tkns[2])
                parts_map[class_id][part_name] = part_idx

            if max_n_classes < len(parts_map[class_id].keys()):
                max_n_classes =  len(parts_map[class_id].keys())

        return parts_map, max_n_classes, classes_to_remove


    def __len__(self):
        return self.nsamples

    def convert_map(self, parts, pmap_dict, class_inds,
                    img, masks, current_cls=0):
        label = np.zeros(img.shape[:2], dtype=np.uint8)

        for part, clsi, m in zip (parts, class_inds, masks):
            if len(pmap_dict[clsi]) == 0:
                label[m!=0] = clsi
            else:
                for k, v in part.items():
                    label[v==1] = pmap_dict[clsi][k]
        mask_all = np.zeros_like(label)
        for m in masks[::-1]:
            m[m!=current_cls] = 0
            mask_all[m!=0] = m[m!=0]
        return label, mask_all

    def read_imgs(self, fname, ncls=None, current_cls=0):
        annot_path = self.root + fname
        img_path = self.imgs_root + fname.split('.')[0] + '.jpg'

        img = cv2.imread(img_path)
        original_img = img.copy()
        annot = scipy.io.loadmat(annot_path)
        cls, clsi, mask, parts, _ = self.parse_annot(annot)
        part_map, mask = self.convert_map(parts, self.pmap_dict,
                                          clsi, img, mask,
                                          current_cls=current_cls)
        part_map[ mask != current_cls] = 0
        if ncls is not None:
            part_map[part_map > ncls] = 0

        if self.is_transform:
            img, part_map = self.transform(img, part_map)

        n_classes = len(self.pmap_dict[current_cls].keys())

        return img, part_map, n_classes, original_img

    def __getitem__(self, index):
        pair = self.files[index]

        sprt_imgs = []
        sprt_lbls = []
        sprt_originals = []

        for fname in pair[0]:
            img, lbl, ncls, original_img = self.read_imgs(fname, current_cls=pair[2])
            sprt_imgs.append(img)
            sprt_lbls.append(lbl)
            sprt_originals.append(original_img)

        qry_img, qry_lbl, _, qry_original = self.read_imgs(pair[1], ncls, current_cls=pair[2])
        return sprt_imgs, sprt_lbls, sprt_originals, \
                    qry_img, qry_lbl, qry_original, pair[2]

def test_loader():
    loader = pascalPARTSLoader('/home/menna/Datasets/PASCALPARTS/',
                               '/home/menna/Datasets/VOCdevkit/VOC2012/JPEGImages/',
                               k_shot=5)

    testloader = data.DataLoader(loader, batch_size=1)

    for batch in testloader:
        plt.figure(1);plt.imshow(np.transpose(batch[0][0][0], (1,2,0)))
        plt.figure(2);plt.imshow(batch[1][0][0])

        plt.figure(4);plt.imshow(np.transpose(batch[0][-1][0], (1,2,0)))
        plt.figure(5);plt.imshow(batch[1][-1][0])

        plt.figure(6);plt.imshow(np.transpose(batch[2][0], (1,2,0)))
        plt.figure(7);plt.imshow(batch[3][0])
        plt.show()

if __name__=="__main__":
    test_loader()

