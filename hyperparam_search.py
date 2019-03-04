import os
import sys
import yaml
import time
import shutil
import torch
import random
import argparse
import datetime
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.utils import data
from tqdm import tqdm

from ptsemseg.models import get_model
from ptsemseg.loss import get_loss_function
from ptsemseg.loader import get_loader
from ptsemseg.metrics import runningScore, averageMeter
from ptsemseg.augmentations import get_composed_augmentations
from ptsemseg.schedulers import get_scheduler
from ptsemseg.optimizers import get_optimizer
from ptsemseg.utils import convert_state_dict, load_my_state_dict

import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler

torch.backends.cudnn.benchmark = True
def post_process(gt, pred):
    gt[gt != 16] = 0
    gt[gt == 16] = 1
    if pred is not None:
        pred[pred != 16] = 0
        pred[pred == 16] = 1
    else:
        pred = None
    return gt, pred

def validate(cfg, args, cfg_hp, rprtr):
    vars(args).update(cfg_hp)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fold = cfg['data']['fold']

    # Setup Dataloader
    data_loader = get_loader(cfg['data']['dataset'])
    data_path = cfg['data']['path']

    loader = data_loader(
        data_path,
        split=cfg['data']['val_split'],
        is_transform=True,
        img_size=(cfg['data']['img_rows'],
                  cfg['data']['img_cols']),
        n_classes=cfg['data']['n_classes'],
        fold=cfg['data']['fold'],
        binary=args.binary,
        k_shot=cfg['data']['k_shot'],
        hparam_search= True
    )

    n_classes = loader.n_classes

    valloader = data.DataLoader(loader,
                                batch_size=cfg['training']['batch_size'],
                                num_workers=1)
    if args.binary:
        running_metrics = runningScore(2)
        iou_list = []
    else:
        running_metrics = runningScore(n_classes+1) #+1 indicate the novel class thats added each time

    # Setup Model
    model = get_model(cfg['model'], n_classes).to(device)
    state = convert_state_dict(torch.load(args.model_path)["model_state"])
    model.load_state_dict(state)
    model.to(device)

    if not args.cl:
        print('No Continual Learning of Bg Class')
        model.save_original_weights()

    alpha = args.alpha
    for i, (sprt_images, sprt_labels, qry_images, qry_labels,
            original_sprt_images, original_qry_images) in enumerate(valloader):
        print('Starting iteration ', i)

        for si in range(len(sprt_images)):
            sprt_images[si] = sprt_images[si].to(device)
            sprt_labels[si] = sprt_labels[si].to(device)
        qry_images = qry_images.to(device)

        # 1- Extract embedding and add the imprinted weights
        model.imprint(sprt_images, sprt_labels, alpha=alpha)

        # 2- Infer on the query image
        model.eval()
        with torch.no_grad():
            outputs = model(qry_images)
            pred = outputs.data.max(1)[1].cpu().numpy()

        # Reverse the last imprinting (Few shot setting only not Continual Learning setup yet)
        model.reverse_imprinting(args.cl)

        gt = qry_labels.numpy()
        if args.binary:
            gt, pred = post_process(gt, pred)

        if args.binary:
            if args.binary == 1:
                iou_list.append(running_metrics.update_binary(gt, pred))
            else:
                running_metrics.update(gt, pred)
        else:
            running_metrics.update(gt, pred)

    if args.binary:
        if args.binary == 1:
            print("Binary Mean IoU ", np.mean(iou_list))
            rprtr(mean_accuracy=np.mean(iou_list))
        else:
            score, class_iou = running_metrics.get_scores()
            for k, v in score.items():
                print(k, v)
            rprtr(mean_accuracy=score["Mean IoU : \t"])
    else:
        score, class_iou = running_metrics.get_scores()
        for k, v in score.items():
            print(k, v)
        val_nclasses = model.n_classes + 1
        for i in range(val_nclasses):
            print(i, class_iou[i])
        rprtr(mean_accuracy=score["Mean IoU : \t"])

def start_hyperopt(args, cfg):
    # Define Scheduler
    ray.init()
    sched = AsyncHyperBandScheduler(
        time_attr="training_iteration",
        reward_attr="mean_accuracy",
        max_t=400,
        grace_period=20)


    # Start trials
    tune.register_trainable("validate",
                            lambda cfg_hp, rprtr: validate(cfg, args, cfg_hp, rprtr))

    tune.run_experiments(
                {
                    "exp": {
                        "stop": {
                            "mean_accuracy": 0.98,
                            "training_iteration": 1000
                        },
                        "resources_per_trial": {
                            "cpu": 3,
                            "gpu": 1
                        },
                        "run": "validate",
                        "num_samples": args.n_samples,
                        "config": {
                            "alpha": tune.sample_from(
                                lambda spec: np.random.uniform(0.001, 0.9)),
                        }
                    }
                },
                verbose=1,
                scheduler=sched)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/fcn8s_pascal.yml",
        help="Configuration file to use"
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
        "--cl",
        dest="cl",
        action="store_true",
        help="Evaluate with continual learning mode for background class",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=10,
        help="number of samples to use in the hyperparameter search")
    args = parser.parse_args()

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="update rate for the adaptive imprinting")

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    start_hyperopt(args, cfg)
