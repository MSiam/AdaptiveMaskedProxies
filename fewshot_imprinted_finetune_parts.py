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

#torch.backends.cudnn.benchmark = True
def save_images(sprt_image, sprt_label, qry_image, iteration, out_dir):
    cv2.imwrite(out_dir+'qry_images/%05d.png'%iteration , qry_image[0].numpy()[:, :, ::-1])
    for i in range(len(sprt_image)):
        cv2.imwrite(out_dir+'sprt_images/%05d_shot%01d.png'%(iteration,i) , sprt_image[i][0].numpy()[:, :, ::-1])
        cv2.imwrite(out_dir+'sprt_gt/%05d_shot%01d.png'%(iteration,i) , sprt_label[i][0].numpy())

def save_vis(heatmaps, prediction, groundtruth, iteration, out_dir, fg_class=16):
    pred = prediction[0]
    pred[pred != fg_class] = 0

    cv2.imwrite(out_dir+'hmaps_bg/%05d.png'%iteration, heatmaps[0, 0, ...].cpu().numpy())
    cv2.imwrite(out_dir+'hmaps_fg/%05d.png'%iteration , heatmaps[0, -1, ...].cpu().numpy())
    cv2.imwrite(out_dir+'pred/%05d.png'%iteration , pred)
    cv2.imwrite(out_dir+'gt/%05d.png'%iteration , groundtruth[0])

def validate(cfg, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(cfg.get('seed', 1385))
    torch.cuda.manual_seed(cfg.get('seed', 1385))

    # Setup Dataloader
    data_loader = get_loader(cfg['data']['dataset'])
    data_path = cfg['data']['path']
    annot_path = cfg['data']['annot_path']

    loader = data_loader(
        annot_path,
        data_path,
        is_transform=True,
        img_size=(cfg['data']['img_rows'],
                  cfg['data']['img_cols']),
        base_n_classes=cfg['data']['base_n_classes'],
        k_shot=cfg['data']['k_shot']
    )

    n_classes = loader.base_n_classes

    valloader = data.DataLoader(loader,
                                batch_size=cfg['training']['batch_size'],
                                num_workers=1)

    # Setup Model
    model = get_model(cfg['model'], n_classes).to(device)
    state = convert_state_dict(torch.load(args.model_path)["model_state"])
    model.load_state_dict(state)
    model.to(device)
    model.freeze_all_except_classifiers()

    # Setup optimizer, lr_scheduler and loss function
    optimizer_cls = get_optimizer(cfg)
    optimizer_params = {k:v for k, v in cfg['training']['optimizer'].items()
                        if k != 'name'}
    print('No Continual Learning of Bg Class')
    model.save_original_weights()

    alpha = 0.14139
    mious = []
    ncls = 3
    for i, (sprt_images, sprt_labels, sprt_originals,
            qry_images, qry_labels, qry_original) in enumerate(valloader):
        print('Starting iteration ', i)
        start_time = timeit.default_timer()

        # Preproc Support and Query Images
        for si in range(len(sprt_images)):
            sprt_images[si] = sprt_images[si].to(device)
            sprt_labels[si] = sprt_labels[si].to(device)
        qry_images = qry_images.to(device)

        # Extract embedding and add the imprinted weights
        model.imprint(sprt_images, sprt_labels, alpha=alpha,
                      ncls=ncls, rnd=args.rnd, decorr=False)

        optimizer = optimizer_cls(model.parameters(), **optimizer_params)
        scheduler = get_scheduler(optimizer, cfg['training']['lr_schedule'])
        loss_fn = get_loss_function(cfg)
        print('Finetuning')
        for j in range(cfg['training']['train_iters']):
            for b in range(len(sprt_images)):
                torch.cuda.empty_cache()
                scheduler.step()
                model.train()
                optimizer.zero_grad()

                outputs = model(sprt_images[b])
                loss = loss_fn(input=outputs, target=sprt_labels[b])
                loss.backward()
                optimizer.step()

                fmt_str = "Iter [{:d}/{:d}]  Loss: {:.4f}"
                print_str = fmt_str.format(j,
                                       cfg['training']['train_iters'],
                                       loss.item())
                print(print_str)


        # Infer on the query image
        model.eval()
        with torch.no_grad():
            outputs = model(qry_images)
            pred = outputs.data.max(1)[1].cpu().numpy()

        # Reverse the last imprinting (Few shot setting only not Continual Learning setup yet)
        model.reverse_imprinting(False)

        gt = qry_labels.numpy()
        running_metrics = runningScore(ncls)

        running_metrics.update_conf(gt, pred)
        score, _ = running_metrics.get_scores()
        for k, v in score.items():
            print(k, v)
        mious.append(score["Mean IoU : \t"])

    print('Overall IoU ', np.mean(mious))

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
        "--cl",
        dest="cl",
        action="store_true",
        help="Evaluate with continual learning mode for background class",
    )
    parser.add_argument(
        "--out_dir",
        nargs="?",
        type=str,
        default="",
        help="Config file to be used",
    )
    parser.add_argument(
        "--rnd",
        default=0,
        help="random flag enable/disable"
    )
    parser.add_argument(
        "--hierarch",
        type=int,
        default=0,
        help="Defines type of hierarch seg if any. 0: None, \
                1: using Prob, 2: using Pred"
    )
    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    validate(cfg, args)
