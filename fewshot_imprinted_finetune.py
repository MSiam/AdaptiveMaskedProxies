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

    valloader = data.DataLoader(loader,
                                batch_size=cfg['training']['batch_size'],
                                num_workers=0)
    if args.binary:
        running_metrics = runningScore(2)
        fp_list = {}
        tp_list = {}
        fn_list = {}
    else:
        running_metrics = runningScore(n_classes+1) #+1 indicate the novel class thats added each time

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
    model.save_original_weights()

    alpha = 0.14139
    for i, (sprt_images, sprt_labels, qry_images, qry_labels,
            original_sprt_images, original_qry_images, cls_ind) in enumerate(valloader):

        cls_ind = int(cls_ind)
        print('Starting iteration ', i)
        start_time = timeit.default_timer()
        if args.out_dir != "":
            save_images(original_sprt_images, sprt_labels,
                        original_qry_images, i, args.out_dir)

        for si in range(len(sprt_images)):
            sprt_images[si] = sprt_images[si].to(device)
            sprt_labels[si] = sprt_labels[si].to(device)
        qry_images = qry_images.to(device)

        # 1- Extract embedding and add the imprinted weights
        model.imprint(sprt_images, sprt_labels, alpha=alpha)

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


        # 2- Infer on the query image
        model.eval()
        with torch.no_grad():
            outputs = model(qry_images)
            pred = outputs.data.max(1)[1].cpu().numpy()

        # Reverse the last imprinting (Few shot setting only not Continual Learning setup yet)
        model.reverse_imprinting()

        gt = qry_labels.numpy()
        if args.binary:
            gt,pred = post_process(gt, pred)

        if args.binary:
            if args.binary == 1:
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
            else:
                running_metrics.update(gt, pred)
        else:
            running_metrics.update(gt, pred)

        if args.out_dir != "":
            if args.binary:
                save_vis(outputs, pred, gt, i, args.out_dir, fg_class=1)
            else:
                save_vis(outputs, pred, gt, i, args.out_dir)

    if args.binary:
        if args.binary == 1:
            iou_list = [tp_list[ic]/float(max(tp_list[ic] + fp_list[ic] + fn_list[ic],1)) \
                         for ic in tp_list.keys()]
            print("Binary Mean IoU ", np.mean(iou_list))
        else:
            score, class_iou = running_metrics.get_scores()
            for k, v in score.items():
                print(k, v)
    else:
        score, class_iou = running_metrics.get_scores()
        for k, v in score.items():
            print(k, v)
        val_nclasses = model.n_classes + 1
        for i in range(val_nclasses):
            print(i, class_iou[i])


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
        "--fold",
        type=int,
        default=-1,
        help="fold index for pascal 5i"
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    validate(cfg, args)
