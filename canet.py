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
from ptsemseg.utils import get_logger
from ptsemseg.metrics import runningScore, averageMeter
from ptsemseg.augmentations import get_composed_augmentations
from ptsemseg.schedulers import get_scheduler
from ptsemseg.optimizers import get_optimizer
from ptsemseg.utils import convert_state_dict, load_my_state_dict
from ptsemseg.models.utils import freeze_weights
import pdb
#from tensorboardX import SummaryWriter

from one_shot_network import *

torch.backends.cudnn.benchmark = True

def post_process(support_mask, query_mask):
    support_mask[support_mask != 16] = 0
    support_mask[support_mask == 16] = 1
    if query_mask is not None:
        query_mask[query_mask != 16] = 0
        query_mask[query_mask == 16] = 1
    else:
        query_mask = None
    return support_mask, query_mask

 # freeze the initial resnet layers
def freeze_weights_extractor(self):
    freeze_weights(self.conv1)
    freeze_weights(self.layer1)
    freeze_weights(self.layer2)
    freeze_weights(self.layer3)

    #freeze_weights(self.conv_block5)

def train(cfg):

    # Setup seeds
    torch.manual_seed(cfg.get('seed', 1337))
    torch.cuda.manual_seed(cfg.get('seed', 1337))
    np.random.seed(cfg.get('seed', 1337))
    random.seed(cfg.get('seed', 1337))

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_loader = get_loader(cfg['data']['dataset'])
    data_path = cfg['data']['path']

    t_loader = data_loader(
        data_path,
        split=cfg['data']['train_split'],
        is_transform=True,
        img_size=[cfg['data']['img_rows'],
                  cfg['data']['img_cols']],
        n_classes=cfg['data']['n_classes'],
        fold=cfg['data']['fold'],
        binary=args.binary,
        k_shot=cfg['data']['k_shot']
    )

    v_loader = data_loader(
        data_path,
        is_transform=True,
        split=cfg['data']['train_split'],
        img_size=[cfg['data']['img_rows'], cfg['data']['img_cols']],
        n_classes=cfg['data']['n_classes'],
        fold=cfg['data']['fold'],
        binary=args.binary,
        k_shot=cfg['data']['k_shot']
    )

    trainloader = data.DataLoader(t_loader,
                                batch_size=cfg['training']['batch_size'],
                                num_workers= 16)

    valloader = data.DataLoader(v_loader,
                                batch_size=1,
                                num_workers=8)

    running_metrics = runningScore(2)
    running_metrics_val = runningScore(2)
    iou_list = []

    num_classes = 2
    model = ResNet(Bottleneck,[3, 4, 6, 3], num_classes)
    model=load_resnet50_param(model)
    model.freeze_weights_extractor()

    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))


    # Setup optimizer, lr_scheduler and loss function
    optimizer_cls = get_optimizer(cfg)
    optimizer_params = {k:v for k, v in cfg['training']['optimizer'].items()
                        if k != 'name'}

    optimizer = optimizer_cls(model.parameters(), **optimizer_params)

    scheduler = get_scheduler(optimizer, cfg['training']['lr_schedule'])

    loss_fn = get_loss_function(cfg)

    best_iou = -100.0
    i = 0
    flag = True
    #pdb.set_trace()
    while i <= cfg['training']['train_iters'] and flag:


      for j, (support_rgb, support_mask, query_rgb, query_mask, \
              original_sprt_images, original_qry_images, _) in enumerate(trainloader):
        #pdb.set_trace()

        print('Starting iteration ', i)
        start_ts = time.time()
        i += 1

        support_rgb = support_rgb[0].to(device)
        support_mask = support_mask[0].to(device)

        support_mask = support_mask.unsqueeze(1)
        query_mask = query_mask.unsqueeze(1)

        support_mask,query_mask = post_process(support_mask, query_mask)

        query_rgb = query_rgb.to(device)
        query_mask = query_mask.to(device)

        scheduler.step()
        model.train()
        optimizer.zero_grad()

        prediction = model(query_rgb,support_rgb,support_mask,history_mask=torch.zeros(1,2,50,50))
        loss = loss_fn(input=prediction, target=query_mask[0])

        loss.backward()
        optimizer.step()

        #time_meter.update(time.time() - start_ts)

        if (i + 1) % cfg['training']['print_interval'] == 0:
           fmt_str = "Iter [{:d}/{:d}]  Loss: {:.4f}"
           print_str = fmt_str.format(i + 1,
                                           cfg['training']['train_iters'],
                                           loss.item())

           print(print_str)
                #logger.info(print_str)

                #writer.add_scalar('loss/train_loss', loss.item(), i+1)
           #time_meter.reset()

        if (i + 1) % cfg['training']['val_interval'] == 0 or \
           (i + 1) == cfg['training']['train_iters']:

           model.eval()
           with torch.no_grad():
               for k, (support_image, support_label, query_image, query_label, \
                       original_sprt_images, original_qry_images, _) in enumerate(valloader):

                   support_image = support_image[0].to(device)
                   support_label = support_label[0].to(device)

                   support_label = support_label.unsqueeze(1)
                   query_label = query_label.unsqueeze(1)

                   support_label,query_label = post_process(support_label, query_label)

                   query_image = query_image.to(device)
                   query_label = query_label.to(device)

                   out = model(query_image,support_image,support_label,history_mask=torch.zeros(1,2,50,50))
                   val_loss = loss_fn(input=out, target=query_label[0])

                   out = F.upsample(out, query_label.size()[2:])
                   pred = out.data.max(1)[1].cpu().numpy()
                   gt = query_label.data.cpu().numpy()

                   running_metrics_val.update(gt[0], pred)
           score, class_iou = running_metrics_val.get_scores()


           running_metrics_val.reset()

           if score["Mean IoU : \t"] >= best_iou:
              best_iou = score["Mean IoU : \t"]
              state = {
                "epoch": i + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_iou": best_iou,
              }

              model_file_name = args.savedir + os.sep + 'model_best.pth'

              filename='checkpoint.pth.tar'
              torch.save(state, filename)
        if (i + 1) == cfg['training']['train_iters']:
           flag = False
           break

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
        "--savedir",
        type=str,
        default="./checkpoints",
        help="Checkpoint save directory",
    )

    parser.add_argument(
        "--binary",
        type=int,
        default=1,
        help="Evaluate binary or full nclasses",
    )

    args = parser.parse_args()


    with open(args.config) as fp:
        cfg = yaml.load(fp)

    run_id = random.randint(1,100000)
    logdir = os.path.join('runs', os.path.basename(args.config)[:-4] , str(run_id))
    #writer = SummaryWriter(log_dir=logdir)
    #writer = SummaryWriter()

    print('RUNDIR: {}'.format(logdir))
    #shutil.copy(args.config, logdir)

    #logger = get_logger(logdir)
    #logger.info('Let the games begin')

    #train(cfg, writer, logger)
    train(cfg)
