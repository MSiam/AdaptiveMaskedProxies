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
from ptsemseg.schedulers import get_scheduler
from ptsemseg.optimizers import get_optimizer
from ptsemseg.loss import get_loss_function
from ptsemseg.augmentations import get_composed_augmentations

#torch.backends.cudnn.benchmark = True
def save_vis(heatmaps, prediction, groundtruth, iteration, out_dir, fg_class=16):
    pred = prediction[0]
    pred[pred != fg_class] = 0

    cv2.imwrite(out_dir+'hmaps_bg/%05d.png'%iteration, heatmaps[0, 0, ...].cpu().numpy())
    cv2.imwrite(out_dir+'hmaps_fg/%05d.png'%iteration , heatmaps[0, -1, ...].cpu().numpy())
    cv2.imwrite(out_dir+'pred/%05d.png'%iteration , pred)
    cv2.imwrite(out_dir+'gt/%05d.png'%iteration , groundtruth[0])

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

    fold = cfg['data']['fold']

    # Setup Dataloader
    data_loader = get_loader(cfg['data']['dataset'])
    data_path = cfg['data']['path']

    augmentations = cfg['training'].get('augmentations', None)
    data_aug = get_composed_augmentations(augmentations)

    t_loader = data_loader(
        data_path,
        is_transform=True,
        split=cfg['data']['train_split'],
        img_size=(cfg['data']['img_rows'], cfg['data']['img_cols']),
        augmentations=data_aug,
        fold=cfg['data']['fold'],
        n_classes=cfg['data']['n_classes'])

    v_loader = data_loader(
            data_path,
            is_transform=True,
            split=cfg['data']['val_split'],
            img_size=(cfg['data']['img_rows'], cfg['data']['img_cols']),
            fold=cfg['data']['fold'],
            n_classes=cfg['data']['n_classes'])

    n_classes = cfg['data']['n_classes']

    # Setup Model
    model = get_model(cfg['model'], n_classes).to(device)
    state = convert_state_dict(torch.load(args.model_path)["model_state"])
    model.load_state_dict(state)
    model.to(device)
    model.freeze_all_except_classifiers()
    original_n_classes = model.n_classes

    # Setup optimizer, lr_scheduler and loss function
    optimizer_cls = get_optimizer(cfg)
    optimizer_params = {k:v for k, v in cfg['training']['optimizer'].items()
                        if k != 'name'}

    cl_log = open(args.cl_log, 'w')
    alpha_ = args.alpha

    for taski in range(t_loader.n_tasks):

        print('Starting Task ', taski)

        # Load batch of current task for training
        t_loader.current_task = taski
        train_loader = data.DataLoader(t_loader,
                                  batch_size=cfg['training']['batch_size'],
                                  num_workers=cfg['training']['n_workers'],
                                  shuffle=True)

        current_n_classes = original_n_classes + (taski+1)*2

        # Add new classes nodes for the current task
        running_metrics = runningScore(current_n_classes)

        # Train on current task
        for i, (task_i, classes, images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            model.imprint(images, labels, alpha=[0.01, alpha_],
                          new_n_classes=current_n_classes)

            if (i + 1) % cfg['training']['print_interval'] == 0:
                fmt_str = "Item [{:d}/{:d}]"
                print_str = fmt_str.format(i+1,
                                       cfg['training']['train_iters']*len(train_loader))
                print(print_str)

        t_loader.current_batch = None

        # Load validation batch for current task
        v_loader.current_task = taski
        val_loader = data.DataLoader(v_loader,
                                     batch_size=cfg['training']['batch_size'],
                                     num_workers=cfg['training']['n_workers'],
                                     shuffle=True)

        # Infer on validation data of current task
        for i, (task_i, classes, images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Infer on validation set for same task
            model.eval()
            with torch.no_grad():
                for b in range(len(images)):
                    outputs = model(images[b])
                    pred = outputs.data.max(1)[1].cpu().numpy()
                    gt = labels[0].cpu().numpy()
                    running_metrics.update(gt, pred)

                    if args.out_dir != "":
                        save_vis(outputs, pred, gt, task_i, args.out_dir)

        v_loader.current_batch = None

        # Evaluate mIoU
        cl_log.write('Task ' + str(taski) + '\n')
        _, class_iou = running_metrics.get_scores()

        avg = 0.0
        count = 0
        for i in range(n_classes, n_classes + (taski+1)*2):
            print(i, class_iou[i])
            cl_log.write(str(i) + ' ' + str(class_iou[i])+'\n')
            avg += class_iou[i]
            count += 1
        cl_log.write('Mean IoU of New Classes '+str(avg/count)+'\n')
        print('Mean IoU of New Classes '+str(avg/count))
        cl_log.flush()

    cl_log.close()

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
        "--measure_time",
        dest="measure_time",
        action="store_true",
        help="Enable evaluation with time (fps) measurement |\
                              True by default",
    )
    parser.add_argument(
        "--no-measure_time",
        dest="measure_time",
        action="store_false",
        help="Disable evaluation with time (fps) measurement |\
                              True by default",
    )
    parser.set_defaults(measure_time=True)

    parser.add_argument(
        "--out_dir",
        nargs="?",
        type=str,
        default="",
        help="Config file to be used",
    )

    parser.add_argument(
        "--cl_log",
        type=str,
        default="cl_log.txt",
        help="log file for continual learning output used for plotting"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.2,
        help="update rate in adaptive imprinting"
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    validate(cfg, args)
