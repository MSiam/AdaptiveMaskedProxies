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
from one_shot_network import *

#torch.backends.cudnn.benchmark = True
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
        binary=1,
        k_shot=cfg['data']['k_shot']
    )

    n_classes = loader.n_classes

    valloader = data.DataLoader(loader,
                                batch_size=cfg['training']['batch_size'],
                                num_workers=0)
    running_metrics = runningScore(2)
    fp_list = {}
    tp_list = {}
    fn_list = {}

    # Setup Model
    running_metrics = runningScore(2)
    running_metrics_val = runningScore(2)
    iou_list = []

    num_classes = 2
    model = ResNet(Bottleneck,[3, 4, 6, 3], num_classes)
    model=load_resnet50_param(model)

    state = convert_state_dict(torch.load("checkpoint.pth.tar")["model_state"])
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    for i, (support_images, support_labels, query_image, query_label,
            original_sprt_images, original_qry_images, cls_ind) in enumerate(valloader):
        cls_ind = int(cls_ind)
        print('Starting iteration ', i)
        start_time = timeit.default_timer()

        support_images = support_images[0].to(device)
        support_labels = support_labels[0].to(device)
        query_image = query_image.to(device)

        support_labels = support_labels.unsqueeze(1)
        query_label = query_label.unsqueeze(1)

        support_labels, query_label = post_process(support_labels, query_label)

        query_image = query_image.to(device)
        query_label = query_label.to(device)

        out = model(query_image,support_images,support_labels,history_mask=torch.zeros(1,2,50,50))
        out = F.upsample(out, query_label.size()[2:])
        pred = out.data.max(1)[1].cpu().numpy()
        gt = query_label.data.cpu().numpy()

        running_metrics_val.update(gt[0], pred)
        score, class_iou = running_metrics_val.get_scores()

        tp, fp, fn = running_metrics.update_binary_oslsm(gt, pred)

        if cls_ind in fp_list.keys():
            fp_list[cls_ind] += fp
        else:
            fp_list[cls_ind] = fp

        if cls_ind in tp_list.keys():
            tp_list[cls_ind] += tp
        else:
            tp_list[cls_ind] = tp

        if cls_ind in fn_list.keys():
            fn_list[cls_ind] += fn
        else:
            fn_list[cls_ind] = fn

    iou_list = [tp_list[ic]/float(max(tp_list[ic] + fp_list[ic] + fn_list[ic],1)) \
                 for ic in tp_list.keys()]
    print("Binary Mean IoU ", np.mean(iou_list))

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
        "--fold",
        type=int,
        default=-1,
        help="fold index for pascal 5i"
    )
    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    validate(cfg, args)
