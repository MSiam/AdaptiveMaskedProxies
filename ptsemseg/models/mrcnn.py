import torch
import json
from torch import nn
from torch import optim

import numpy as np
import torch.nn.functional as F
from ptsemseg.models import ann_utils as au

from maskrcnn_benchmark.structures.image_list import to_image_list

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.rpn import rpn
from maskrcnn_benchmark.modeling import backbone
from maskrcnn_benchmark.modeling.roi_heads import roi_heads
from maskrcnn_benchmark.utils.model_serialization import load_state_dict
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.model_zoo import cache_url
from maskrcnn_benchmark.utils.c2_model_loading import load_c2_format
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from ptsemseg.models import mrcnn_utils as ut


class MRCNN(nn.Module):
    def __init__(self, exp_dict):
        super().__init__()
        self.option_dict = exp_dict["option"]
        cfg_base_path = "./models/configs/"

        self.n_classes = 21
        cfg_path = cfg_base_path + "e2e_mask_rcnn_R_50_FPN_1x.yaml"

        self.cfg = cfg
        self.cfg.merge_from_file(cfg_path)

        self.cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES = self.n_classes

        # ---------------
        # build model
        self.backbone_fpn = backbone.build_backbone(self.cfg)
        self.rpn = rpn.build_rpn(self.cfg, self.backbone_fpn.out_channels)
        self.roi_heads = roi_heads.build_roi_heads(
            self.cfg, self.backbone_fpn.out_channels)

        # ---------------
        # load checkpoint
        checkpoint = _load_file(self.cfg)
        load_state_dict(self, checkpoint.pop("model"))

        #--------
        # Opt stage
        self.cfg.SOLVER.BASE_LR = ((0.0025 * 8) /
                     (16 / float(exp_dict["option"]["batch_size"])))

        optimizer = make_optimizer(self.cfg, self)
        scheduler = make_lr_scheduler(self.cfg, optimizer)
        self.opt = optimizer
        self.scheduler = scheduler

    def get_input(self, batch):
        # Image
        imageList = ensure_image_list(
            batch["images"],
            SIZE_DIVISIBILITY=self.cfg.DATALOADER.SIZE_DIVISIBILITY)

        imageList = imageList.to(0)
        targets = [target.to(0) for target in batch["targets"]]

        batch_dict = {"imageList":imageList,
                      "targets":targets}

        return batch_dict

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        images = to_image_list(images)

        # features = self.backbone_fpn(images.tensors)
        features = self.backbone_fpn(images.tensors)
        proposals, proposal_losses = self.rpn(images, features, targets)

        try:
            _, result, detector_losses = self.roi_heads(
                features, proposals, targets)
        except RuntimeError as e:
            if  e.message == 'cuDNN error: CUDNN_STATUS_BAD_PARAM':
                import ipdb; ipdb.set_trace()  # breakpoint 9825e6cd //


        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)

            return losses

        else:
            return result

    def train_step(self, batch):
        for annList in batch["annList"]:
            if len(annList) == 0:
                return 0.

        batch_dict = self.get_input(batch)

        # Make step
        self.train()
        if hasattr(self, "scheduler"):
            self.scheduler.step()
        self.opt.zero_grad()

        loss_dict = self(batch_dict["imageList"], batch_dict["targets"])
        label_loss = sum(loss for loss in loss_dict.values())
        label_loss.backward()

        self.opt.step()

        return float(label_loss)

    @torch.no_grad()
    def predict(self, batch, method="annList"):
        self.eval()
        for annList in batch["annList"]:
            if len(annList) == 0:
                return []

        batch_dict = self.get_input(batch)
        if len(batch_dict) == 0:
            return []

        preds = self(batch_dict["imageList"])[0]
        image_id = batch["meta"]["image_id"][0]
        _, _, H, W = map(int, batch["meta"]["shape"])

        if method == "annList":
            maskVoid_flag = True
            if self.option_dict.get("apply_void") == "True":
                maskVoid = batch["maskVoid"][0]
            else:
                maskVoid = None
            annList = au.targets2annList(
                preds, shape=(H, W), image_id=image_id,
                maskVoid=maskVoid)

        if method == "annList_noVoid":
            maskVoid_flag = False
            annList = au.targets2annList(
                preds, shape=(H, W), image_id=image_id,
                maskVoid=None)

        if self.option_dict.get("refine") == "bo":
            annList = au.annList2BestDice(annList, batch,
                    maskVoid_flag=maskVoid_flag)["annList"]

        return annList

    @torch.no_grad()
    def visualize(self, batch, return_image=False):
        self.eval()
        _,_,H_org, W_org = batch["meta"]["shape"]
        batch_dict = self.get_input(batch)
        targets_preds = self(batch_dict["imageList"])[0]
        annList_gt = batch["annList"][0]
        annList_pred = au.targets2annList(
                targets_preds, shape=(H_org, W_org),
                image_id=batch["meta"]["image_id"][0])
        # Resize
        W_img, H_img = targets_preds.size
        images_given = batch["images"].tensors[:,:,:H_img, :W_img]

        images = F.interpolate(
            images_given,
            size=(H_org, W_org), mode="nearest")

        pred_image = ut.get_image(images, annList=annList_pred, denorm="bgr")
        pred_image_blacked = ut.get_image(images*0, annList=annList_pred)
        gt_image = ut.get_image(images, annList=annList_gt, denorm="bgr")

        result = np.concatenate([pred_image, gt_image], axis=2)

        return {"gt_image":gt_image,
                "pred_image":pred_image,
                "pred_image_blacked":pred_image_blacked}

        # ut.images(pred_image, win="pred")
        # ut.images(gt_image, win="gt")
        # ut.images(result)

    def extract_features(self, batch):
        feats = self.main_model.extract_backbone(batch["images"].to(0))
        feats = feats["backbone"][-1].mean(1)
        return feats.view(-1)


#-----------------------------
# misc
def _load_file(cfg):
    f = cfg.MODEL.WEIGHT
    # catalog lookup
    if f.startswith("catalog://"):
        paths_catalog = import_file("maskrcnn_benchmark.config.paths_catalog",
                                    cfg.PATHS_CATALOG, True)
        catalog_f = paths_catalog.ModelCatalog.get(f[len("catalog://"):])

        f = catalog_f
    # download url files
    if f.startswith("http"):
        # if the file is a url path, download it and cache it
        cached_f = cache_url(f)

        f = cached_f
    # convert Caffe2 checkpoint from pkl
    if f.endswith(".pkl"):
        return load_c2_format(cfg, f)


def ensure_image_list(T, SIZE_DIVISIBILITY=1):
    if isinstance(T, torch.Tensor):
        image_list = to_image_list([t for t in T],
                                   size_divisible=SIZE_DIVISIBILITY)
    else:
        image_list = T

    return image_list
