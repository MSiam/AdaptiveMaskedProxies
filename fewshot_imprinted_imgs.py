import os
import sys
import yaml
import torch
import argparse
import timeit
import numpy as np
import scipy.misc as misc
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.backends import cudnn
from torch.utils import data

from tqdm import tqdm

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.metrics import runningScore
from ptsemseg.utils import convert_state_dict
import matplotlib.pyplot as plt
import copy
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
import cv2
import torch.nn.functional as F

#torch.backends.cudnn.benchmark = True
def save_images(sprt_image, sprt_label, qry_image, iteration, out_dir):
    cv2.imwrite(out_dir+'qry_images/%05d.png'%iteration , qry_image[0].numpy())
    for i in range(len(sprt_image)):
        cv2.imwrite(out_dir+'sprt_images/%05d_shot%01d.png'%(iteration,i) , sprt_image[i][0].numpy())
        cv2.imwrite(out_dir+'sprt_gt/%05d_shot%01d.png'%(iteration,i) , sprt_label[i][0].numpy())

def save_vis(heatmaps, prediction, groundtruth, iteration, out_dir, fg_class=16):
    pred = prediction[0]
    pred[pred != fg_class] = 0

    cv2.imwrite(out_dir+'hmaps_bg/%05d.png'%iteration, heatmaps[0, 0, ...].cpu().numpy())
    cv2.imwrite(out_dir+'hmaps_fg/%05d.png'%iteration , heatmaps[0, -1, ...].cpu().numpy())
    cv2.imwrite(out_dir+'pred/%05d.png'%iteration , pred)
    cv2.imwrite(out_dir+'gt/%05d.png'%iteration , groundtruth[0])

def post_process(gt, pred):
    gt[gt != 16] = 0
    gt[gt == 16] = 1
    if pred is not None:
        pred[pred != 16] = 0
        pred[pred == 16] = 1
    else:
        pred = None
    return gt, pred

def validate(cfg, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.out_dir != "":
        if not os.path.exists(args.out_dir):
            os.mkdir(args.out_dir)
        if not os.path.exists(args.out_dir+'hmaps_bg'):
            os.mkdir(args.out_dir+'hmaps_bg')
        if not os.path.exists(args.out_dir+'hmaps_fg'):
            os.mkdir(args.out_dir+'hmaps_fg')
        if not os.path.exists(args.out_dir+'pred'):
            os.mkdir(args.out_dir+'pred')
        if not os.path.exists(args.out_dir+'gt'):
            os.mkdir(args.out_dir+'gt')
        if not os.path.exists(args.out_dir+'qry_images'):
            os.mkdir(args.out_dir+'qry_images')
        if not os.path.exists(args.out_dir+'sprt_images'):
            os.mkdir(args.out_dir+'sprt_images')
        if not os.path.exists(args.out_dir+'sprt_gt'):
            os.mkdir(args.out_dir+'sprt_gt')

    if args.fold != -1:
        cfg['data']['fold'] = args.fold

    fold = cfg['data']['fold']

    # Setup Dataloader
    data_loader = get_loader(cfg['data']['dataset'])
    data_path = cfg['data']['path']

    loader = data_loader(
        data_path,
        split=cfg['data']['val_split'],
        is_transform=True,
        img_size=[cfg['data']['img_rows'],
                  cfg['data']['img_cols']],
        n_classes=cfg['data']['n_classes'],
        fold=cfg['data']['fold'],
        binary=args.binary,
        k_shot=cfg['data']['k_shot']
    )

    n_classes = loader.n_classes

    # Setup Model
    model = get_model(cfg['model'], n_classes).to(device)
    state = convert_state_dict(torch.load(args.model_path)["model_state"])
    model.load_state_dict(state)
    model.to(device)

    model.save_original_weights()

    alpha = 0.25821

    sprt_fname = '2007_001311'
    qry_fname = '2007_003188'

    sprt_images = cv2.imread(loader.root + 'JPEGImages/'+sprt_fname+'.jpg')
    sprt_labels = cv2.imread(loader.root+'SegmentationClass/pre_encoded/'+sprt_fname+'.png', 0)
    qry_images = cv2.imread(loader.root+'JPEGImages/'+qry_fname+'.jpg')
    qry_labels = cv2.imread(loader.root+'SegmentationClass/pre_encoded/'+qry_fname+'.png')

    orig_sprt = sprt_images.copy()
    orig_qry = qry_images.copy()

    sprt_images, sprt_labels = loader.transform(sprt_images, sprt_labels)
    sprt_images = [sprt_images.unsqueeze(0)]
    sprt_labels = [sprt_labels.unsqueeze(0)]

    qry_images, qry_labels = loader.transform(qry_images, qry_labels)
    qry_images = qry_images.unsqueeze(0)

    for si in range(len(sprt_images)):
        sprt_images[si] = sprt_images[si].to(device)
        sprt_labels[si] = sprt_labels[si].to(device)
    qry_images = qry_images.to(device)

    # 1- Extract embedding and add the imprinted weights
    if args.iterations_imp > 0:
        model.iterative_imprinting(sprt_images, qry_images, sprt_labels,
                                   alpha=alpha, itr=args.iterations_imp)
    else:
        model.imprint(sprt_images, sprt_labels, alpha=alpha)

    # 2- Infer on the query image
    model.eval()
    with torch.no_grad():
        outputs = model(qry_images)
        pred = outputs.data.max(1)[1].cpu().numpy()

    # Reverse the last imprinting (Few shot setting only not Continual Learning setup yet)
    model.reverse_imprinting()

    plt.figure(1); plt.imshow(orig_sprt[:,:,::-1])
    plt.figure(2); plt.imshow(orig_qry[:,:,::-1])
    plt.figure(3); plt.imshow(pred[0]); plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparams")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/fcn8s_pascal.yml",
        help="Config file to be used",
    )
    parser.add_argument(
        "--model_path",
        nargs="?",
        type=str,
        default="fcn8s_pascal_1_26.pkl",
        help="Path to the saved model",
    )
    parser.add_argument(
        "--binary",
        type=int,
        default=0,
        help="Evaluate binary or full nclasses",
    )
    parser.add_argument(
        "--out_dir",
        nargs="?",
        type=str,
        default="",
        help="Config file to be used",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=-1,
        help="fold index for pascal 5i"
    )
    parser.add_argument(
        "--iterations_imp",
        type=int,
        default=0,
        help="iterations used for iterative refinement"
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    validate(cfg, args)
