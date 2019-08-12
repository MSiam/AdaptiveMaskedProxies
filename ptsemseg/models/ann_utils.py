import os
from skimage import morphology as morph
from torch.utils import data
import pandas as pd
import pycocotools.mask as mask_util
from skimage import measure
from ptsemseg.models import mrcnn_utils as ut
import numpy as np
from skimage.transform import resize

from torch.nn.functional import interpolate
import pycocotools.mask as mask_utils
import torch

from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark.structures.bounding_box import BoxList
from pycocotools import mask as maskUtils


# ------------------------------
# proposals
def annList2BestDice(annList, batch, maskVoid_flag=True):
    if annList == []:
        return {"annList": annList}

    assert len(batch["proposals"]) == 1
    proposals = batch["proposals"][0]
    # dt = [ann["segmentation"] for ann in batch["annList"]]
    # dt = [{'size': [333, 500], 'counts': 'Ya\\18S:3N2M2M3O1O1N2M3K5N2O1N2jNV1L4N2O1N2\\Od0L4N2M3K5N2O1O1N2N2N2O1O1N2N2O1O1N2N2O1O1O1N2O1O1O1O1O1N2O1O1O1N2N2O1O1O1N2O1O1O100O10000O100O100N2O100O10O0H9N1O100O1O101O1O0N3N2O1N2N2N2N3N1N2N3J6K5N2M3L3N3N1O2N1N3M4N1N3N2N1O2M3M4L4K4M2N5J5K4L3N3L6D;I6L4L5J`U^2'}, {'size': [333, 500], 'counts': '[ea36Q:;J3hNHUH=\\OBf7>eHc0S7LaH9\\7^1M2L4I7O1O2L3N2N2O1O2L3O1O2M3L6LR1mN4M2M3N9G101O1O000000000000001O00000000000000000000aNfKkLZ4e2XLYMh3T2jKfLg0U1_3P2RMoMn2P2SMPNm2P2SMPNm2o1TMQNl2o1TMQNl2o1TMQNl2o1UMPNk2P2RLlLa0T1]3P2mKRMe0n0^3P2lKTMe0l0_3Q2iKVMh0g0`3T2eKXMj0d0a3i2^LWMb3i2_LVMa3j2`LUM`3k2aLTM_3l2bLSM^3m2dLQM\\3o2fLoLZ3Q3hLmLY3R3jLkLV3U3VM_Lk2`3RMcLQ3Y3hLoLY3P3fLQM[3n2dLSM^3k2^LZMc3d2ZL_Mh3_2ULdMl3[2RLgMP4W2TLSMPO0o4k2iLnLY3P3jLlLY3S3iLiLY3V3U2J:F`0@4L6J8H5K4L8H4L2M2N1O2M2M6Iah?'}]
    annList_segs = [ann["segmentation"] for ann in annList]
    props_segs = [prop["segmentation"]
                  for prop in proposals if prop["score"] > 0.75]

    ious = maskUtils.iou(annList_segs,
                         props_segs,
                         np.zeros(len(annList_segs)))
    if len(ious) == 0:
        return {"annList": annList}

    # return {"annList":annList}
    indices = ious.argmax(axis=1)
    annList_new = []
    # print("best ious", ious[:, indices])

    for i, ind in enumerate(indices):
        ann_org = annList[i]
        ann_prop_seg = props_segs[ind]
        # Apply Void
        binmask = ann2mask({"segmentation": ann_prop_seg})["mask"]
        maskVoid = batch["maskVoid"]
        if (maskVoid is not None) and maskVoid_flag:
            binmask = binmask * (ut.t2n(maskVoid).squeeze())

        ann_prop_seg = maskUtils.encode(
            np.asfortranarray(ut.t2n(binmask)).astype("uint8"))
        ann_prop_seg["counts"] = ann_prop_seg["counts"].decode("utf-8")

        ann_org["segmentation"] = ann_prop_seg
        annList_new += [ann_org]

    return {"annList": annList_new}
    # # ious = maskUtils.iou(dt, dt, 0)
    # return {"annList": annList}

    # proposals = batch["proposals"]
    # new_annList = []

    # if "maskVoid" in batch and batch["maskVoid"] is not None:
    #     maskVoid = batch["maskVoid"]
    # else:
    #     maskVoid = None

    # dt_list = []
    # for i, dt in enumerate(copy.deepcopy(annList)):
    #     # key = (dt['image_id'], dt['category_id'])

    #     bb = dt["bbox"]
    #     dt['area'] = bb[2]*bb[3]

    #     dt["id"] = i + 1
    #     dt['iscrowd'] = 0
    #     dt_list += [dt]
    # #     dt_dict[key] += [dt]
    # import ipdb; ipdb.set_trace()  # breakpoint 436e4cb4 //
    # for ann_prop in proposals:
    #     if ann_prop["score"] < 0.5:
    #         continue

    #     for ann in annList:
    #         binmask = ann2mask(ann)["mask"]
    #         best_dice = 0.
    #         best_mask = None
    #     for ann_prop in proposals:
    #         import ipdb; ipdb.set_trace()  # breakpoint 81db1ecd //

    #         if ann["score"] < 0.5:
    #             continue
    #         score = dice(ann["mask"], binmask)
    #         if score > best_dice:
    #             best_dice = score
    #             best_mask = ann["mask"]
    #             # prop_score = ann["score"]

    #     if best_mask is None:
    #         best_mask = binmask

    #     if maskVoid is not None:
    #         binmask = best_mask * (ut.t2n(maskVoid).squeeze())
    #     else:
    #         binmask = best_mask

    #     seg = maskUtils.encode(np.asfortranarray(ut.t2n(binmask)).astype("uint8"))
    #     seg["counts"] = seg["counts"].decode("utf-8")
    #     ann["score"] = best_dice
    #     ann["segmentation"] = seg

    #     new_annList += [ann]


@torch.no_grad()
def annList2best_objectness(annList, points,
                            single_point=True, maskVoid=None):
    for ann in annList:
        pass
    return {"annList": annList, "blobs": blobs, "categoryDict": categoryDict}


@torch.no_grad()
def pointList2BestObjectness(pointList, image_id,
                             single_point=True, maskVoid=None):
    import ipdb;
    ipdb.set_trace()  # breakpoint 2a55a8ba //

    propDict = pointList2propDict(
        pointList, batch, thresh=0.5,
        single_point=single_point)

    h, w = propDict["background"].squeeze().shape
    blobs = np.zeros((h, w), int)
    categoryDict = {}

    annList = []
    for i, prop in enumerate(propDict['propDict']):
        if len(prop["annList"]) == 0:
            continue
        blobs[prop["annList"][0]["mask"] != 0] = i + 1

        categoryDict[i + 1] = prop["category_id"]

        if maskVoid is not None:
            binmask = prop["annList"][0]["mask"] * (ut.t2n(maskVoid).squeeze())
        else:
            binmask = prop["annList"][0]["mask"]

        seg = maskUtils.encode(
            np.asfortranarray(ut.t2n(binmask)).astype("uint8"))
        seg["counts"] = seg["counts"].decode("utf-8")
        score = prop["annList"][0]["score"]

        annList += [{
            "segmentation": seg,
            "iscrowd": 0,
            "area": int(maskUtils.area(seg)),
            "image_id": image_id,
            "category_id": int(prop['category_id']),
            "height": h,
            "width": w,
            "score": score
        }]

    return {"annList": annList, "blobs": blobs, "categoryDict": categoryDict}


class SharpProposals:
    def __init__(self, fname):
        # if dataset_name == "pascal":
        self.proposals_path = batch["proposals_path"][0]

        if "SharpProposals_name" in batch:
            batch_name = batch["SharpProposals_name"][0]
        else:
            batch_name = batch["name"][0]
        name_jpg = self.proposals_path + "{}.jpg.json".format(batch_name)
        name_png = self.proposals_path + "{}.json".format(batch_name)

        if os.path.exists(name_jpg):
            name = name_jpg
        else:
            name = name_png

        _, _, self.h, self.w = batch["images"].shape

        if "resized" in batch and batch["resized"].item() == 1:
            name_resized = self.proposals_path + "{}_{}_{}.json".format(batch["name"][0],
                                                                        self.h, self.w)

            if not os.path.exists(name_resized):
                proposals = ut.load_json(name)
                json_file = loop_and_resize(self.h, self.w, proposals)
                ut.save_json(name_resized, json_file)
        else:
            name_resized = name
        # name_resized = name
        proposals = ut.load_json(name_resized)
        self.proposals = sorted(proposals, key=lambda x: x["score"],
                                reverse=True)

    def __getitem__(self, i):
        encoded = self.proposals[i]["segmentation"]
        proposal_mask = maskUtils.decode(encoded)

        return {"mask": proposal_mask,
                "score": self.proposals[i]["score"]}

    def __len__(self):
        return len(self.proposals)

    def sharpmask2psfcn_proposals(self):
        import ipdb;
        ipdb.set_trace()  # breakpoint 102ed333 //

        pass


def loop_and_resize(h, w, proposals):
    proposals_resized = []
    n_proposals = len(proposals)
    for i in range(n_proposals):
        print("{}/{}".format(i, n_proposals))
        prop = proposals[i]
        seg = prop["segmentation"]
        proposal_mask = maskUtils.decode(seg)
        # proposal_mask = resize(proposal_mask*255, (h, w), order=0).astype("uint8")

        if not proposal_mask.shape == (h, w):
            proposal_mask = (resize(proposal_mask * 255, (h, w), order=0) > 0).astype(int)
            seg = maskUtils.encode(np.asfortranarray(proposal_mask).astype("uint8"))
            seg["counts"] = seg["counts"].decode("utf-8")

            prop["segmentation"] = seg
            proposals_resized += [proposals[i]]

        else:
            proposals_resized += [proposals[i]]

    return proposals_resized


@torch.no_grad()
def pointList2propDict(pointList, batch, single_point=False, thresh=0.5):
    sharp_proposals = SharpProposals(batch)
    propDict = []
    shape = pointList[0]["shape"]
    foreground = np.zeros(shape, int)

    if single_point:
        points = pointList2mask(pointList)["mask"]

    idDict = {}
    annDict = {}
    for i, p in enumerate(pointList):
        annDict[i] = []
        idDict[i] = []

    for k in range(len(sharp_proposals)):
        proposal_ann = sharp_proposals[k]
        if not (proposal_ann["score"] > thresh):
            continue
        proposal_mask = proposal_ann["mask"]

        for i, p in enumerate(pointList):
            if proposal_mask[p["y"], p["x"]] == 0:
                continue

            if single_point and (points * proposal_mask).sum() > 1:
                continue

            # score = proposal_ann["score"]

            annDict[i] += [proposal_ann]
            idDict[i] += [k]

    for i in annDict:
        annList = annDict[i]
        idList = idDict[i]
        p = pointList[i]

        mask = annList2mask(annList)["mask"]
        if mask is not None:
            foreground = foreground + mask

        # foreground[foreground<2]=0
        propDict += [{
            "annList": annList,
            "point": p,
            "idList": idList,
            "category_id": int(p["category_id"])
        }]
        #########

    return {
        "propDict": propDict,
        "foreground": foreground,
        "background": (foreground == 0).astype(int)
    }


def mask2annList(maskClass, maskObjects, image_id, classes=None,
                 maskVoid=-1):
    annList = []
    maskClass = np.array(maskClass)
    maskObjects = np.array(maskObjects)

    objects = np.setdiff1d(np.unique(maskObjects), [0, 255])
    for i, obj in enumerate(objects):
        binmask = maskObjects == obj
        category_id = np.unique(maskClass * binmask)[1]

        if classes is not None and category_id not in classes:
            continue

        ann = mask2ann(binmask, category_id=category_id,
                       image_id=image_id,
                       maskVoid=maskVoid,
                       score=1,
                       point=-1)
        # "bbox" - [nx4] Bounding box(es) stored as [x y w h]
        ann["bbox"] = maskUtils.toBbox(ann["segmentation"])
        # ann["id"] = i + 1
        annList += [ann]

    return annList


def annList2targets(annList):
    if len(annList) == 0:
        return []

    ann = annList[0]
    H = ann["height"]
    W = ann["width"]
    img_size = (W, H)

    annList = [obj for obj in annList if (("iscrowd" not in obj) or
                                          (obj["iscrowd"] == 0))]

    boxes = [obj["bbox"] for obj in annList]
    boxes = torch.as_tensor(boxes).reshape(len(annList), 4)  # guard against no boxes
    target = BoxList(boxes, img_size, mode="xywh").convert("xyxy")

    classes = [obj["category_id"] for obj in annList]
    classes = torch.tensor(classes)
    target.add_field("labels", classes)

    if "segmentation" in annList[0]:
        masks = [obj["segmentation"] for obj in annList]
        # from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
        masks = SegmentationMask(masks, img_size, mode="mask")

        target.add_field("masks", masks)

    target = target.clip_to_image(remove_empty=True)
    # ut.images(torch.ones(1,3,500,500), annList=target2annList(target, "1"))
    return target


def targets2annList_bbox(preds, image_id=-1, maskVoid=None):
    bbox = preds.bbox
    W, H = preds.size
    annList = bbox2annList(
        bbox,
        scoreList=preds.get_field("scores"),
        categoryList=preds.get_field("labels"),
        H=H,
        W=W,
        image_id=image_id,
        mode="xyxy",
        mask=None)
    return annList


def targets2annList(targets, shape, image_id=-1, maskVoid=None, score_threshold=0.5):
    # preds.to("cpu")
    H, W = shape
    preds = targets.resize((W, H))

    labels = targets.get_field("labels")
    if "masks" in targets.extra_fields:
        masks = targets.get_field("masks")
        masks = masks.get_mask_tensor()[:, None]
    else:
        masks = targets.get_field("mask")

    if "scores" in targets.extra_fields:
        scores = targets.get_field("scores")
    else:
        scores = labels * 0 + 1

    masker = Masker(threshold=0.5, padding=1)

    if list(masks.shape[-2:]) != [H, W]:
        masks = masker(masks.expand(1, -1, -1, -1, -1), targets)
        masks = masks[0]

    # apply mask void
    if maskVoid is not None and masks.shape[0] > 0:
        masks = masks * maskVoid.byte()

    rles = [
        mask_util.encode(np.array(mask[0, :, :, np.newaxis], order="F"))[0]
        for mask in masks
    ]
    for rle in rles:
        rle["counts"] = rle["counts"].decode("utf-8")

    annList = segm2annList(
        segm=rles,
        boxes=preds.bbox.cpu(),
        scoreList=scores,
        categoryList=labels,
        H=H,
        W=W,
        image_id=image_id,
        mode="xyxy",
        mask=None,
        score_threshold=score_threshold)

    return annList


def proposals2bbox(bbox, scoreList, W, H):
    annList = bbox2annList(
        bbox,
        scoreList,
        np.ones(bbox.shape[0]),
        H,
        W,
        image_id="proposal",
        mode="xyxy",
        mask=None)

    return annList


def proposals2annList(proposals):
    bbox = proposals.bbox
    W, H = proposals.size
    annList = bbox2annList(
        bbox,
        proposals.get_field("objectness"),
        np.ones(bbox.shape[0]),
        H,
        W,
        image_id="proposal",
        mode="xyxy",
        mask=None)

    return annList


def target2annList(target, image_id):
    bbox = target.bbox
    W, H = target.size
    annList = bbox2annList(
        bbox,
        np.zeros(bbox.shape[0]),
        target.get_field("labels"),
        H,
        W,
        image_id,
        mode="xyxy",
        mask=None)

    return annList


def load_ann_json(fname, image_shape):
    name = ut.extract_fname(fname)
    fname_new = fname.replace(name, name + "_%s.json" % str(image_shape))

    if os.path.exists(fname_new):
        return ut.load_json(fname_new)
    else:
        annList = ut.load_json(fname)

        annList_new = []
        for ann in annList:
            binmask = ann2mask(ann)["mask"]
            if binmask.shape != image_shape:
                binmask = resize(
                    ann2mask(ann)["mask"],
                    output_shape=image_shape,
                    order=0,
                    anti_aliasing=False,
                    mode="constant",
                    preserve_range=True).astype(int)

            seg = maskUtils.encode(
                np.asfortranarray(ut.t2n(binmask)).astype("uint8"))
            seg["counts"] = seg["counts"].decode("utf-8")
            ann["score"] = ann["score"]
            ann["segmentation"] = seg
            ann["bbox"] = maskUtils.toBbox(seg).astype(int).tolist()

            annList_new += [ann]

        ut.save_json(fname_new, annList_new)

    return annList_new


def intersect_bbox(b1, b2):
    xs_1, ys_1, w1, h1 = np.array(b1)
    xs_2, ys_2, w2, h2 = np.array(b2)

    xe_1 = xs_1 + w1
    ye_1 = ys_1 + h1

    xe_2 = xs_2 + w2
    ye_2 = ys_2 + h2

    # if(y1<(y2+h2) or x1<(x2+w2)):
    #     flag = False

    # elif((x1 + w1) > x2 or (y1 + h1) > y2):
    #     flag = False

    # else:
    #     flag = True

    return not (xe_1 < xs_2 or xs_1 > xe_2 or ye_1 < ys_2 or ys_1 > ye_2)


def annList2propList(annList, sharpmask_annList):
    new_annList = []

    for ann in annList:
        binmask = ann2mask(ann)["mask"]
        best_score = 0.
        best_mask = binmask
        # ut.images(sharp_mask, win="resized")
        # ut.images(ann2mask(sharp_ann)["mask"], win="original")
        for sharp_ann in sharpmask_annList:
            if ann["score"] < 0.5:
                continue

            if not intersect_bbox(ann["bbox"], sharp_ann["bbox"]):
                continue

            sharp_mask = ann2mask(sharp_ann)["mask"]
            score = dice(sharp_mask, binmask)

            # assert score > 0
            if score > best_score:
                best_mask = sharp_mask
                best_score = score

                # ut.images(sharp_mask, win="sharpmask")
                # ut.images(binmask, win="predmask")
                # break

            # if score > best_dice:
            #     best_dice = score
            #     best_mask = sharp_mask

        seg = maskUtils.encode(
            np.asfortranarray(ut.t2n(best_mask)).astype("uint8"))
        seg["counts"] = seg["counts"].decode("utf-8")
        ann["score"] = best_score
        ann["segmentation"] = seg

        new_annList += [ann]

    return new_annList


def maskList2annList(maskList, categoryList, image_id, scoreList=None):
    annList = []
    _, h, w = maskList.shape

    for i in range(maskList.shape[0]):
        binmask = maskList[i]

        seg = maskUtils.encode(
            np.asfortranarray(ut.t2n(binmask)).astype("uint8"))
        seg["counts"] = seg["counts"].decode("utf-8")
        if scoreList is not None:
            score = scoreList[i]

        annList += [{
            "segmentation": seg,
            "iscrowd": 0,
            "bbox": maskUtils.toBbox(seg).astype(int).tolist(),
            "area": int(maskUtils.area(seg)),
            "image_id": image_id,
            "category_id": int(categoryList[i]),
            "height": h,
            "width": w,
            "score": score
        }]

    return annList


# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1


class Mask(object):
    """
    This class is unfinished and not meant for use yet
    It is supposed to contain the mask for an object as
    a 2d tensor
    """

    def __init__(self, segm, size, mode):
        width, height = size
        if isinstance(segm, Mask):
            mask = segm.mask
        else:
            if type(segm) == list:
                # polygons
                mask = Polygons(
                    segm, size,
                    'polygon').convert('mask').to(dtype=torch.float32)
            elif type(segm) == dict and 'counts' in segm:
                if type(segm['counts']) == list:
                    # uncompressed RLE
                    h, w = segm['size']

                    rle = mask_utils.frPyObjects(segm, h, w)
                    mask = mask_utils.decode(rle)
                    mask = torch.from_numpy(mask).to(dtype=torch.float32)
                else:
                    # compressed RLE
                    mask = mask_utils.decode(segm)
                    mask = torch.from_numpy(mask).to(dtype=torch.float32)
            else:
                # binary mask
                if type(segm) == np.ndarray:
                    mask = torch.from_numpy(segm).to(dtype=torch.float32)
                else:  # torch.Tensor
                    mask = segm.to(dtype=torch.float32)
        self.mask = mask
        self.size = size
        self.mode = mode

    def transpose(self, method):
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented")

        width, height = self.size
        if method == FLIP_LEFT_RIGHT:
            max_idx = width
            dim = 1
        elif method == FLIP_TOP_BOTTOM:
            max_idx = height
            dim = 0

        flip_idx = torch.tensor(list(range(max_idx)[::-1]))
        flipped_mask = self.mask.index_select(dim, flip_idx)
        return Mask(flipped_mask, self.size, self.mode)

    def crop(self, box):
        box = [int(b) for b in box]
        # w, h = box[2] - box[0], box[3] - box[1]
        w, h = box[2] - box[0] + 1, box[3] - box[1] + 1

        # if w == 0:
        #     box[2] = box[0] + 1

        # if h == 0:
        #     box[3] = box[1] + 1

        w = max(w, 1)
        h = max(h, 1)
        # cropped_mask = self.mask[box[1]: box[3], box[0]: box[2]]
        cropped_mask = self.mask[box[1]:box[3] + 1, box[0]:box[2] + 1]
        return Mask(cropped_mask, size=(w, h), mode=self.mode)

    def resize(self, size, *args, **kwargs):
        width, height = size
        scaled_mask = interpolate(
            self.mask[None, None, :, :], (height, width), mode='nearest')[0, 0]
        return Mask(scaled_mask, size=size, mode=self.mode)

    def convert(self, mode):
        mask = self.mask.to(dtype=torch.uint8)
        return mask

    def __iter__(self):
        return iter(self.mask)

    def __repr__(self):
        s = self.__class__.__name__ + "("
        # s += "num_mask={}, ".format(len(self.mask))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={}, ".format(self.size[1])
        s += "mode={})".format(self.mode)
        return s


class Polygons(object):
    """
    This class holds a set of polygons that represents a single instance
    of an object mask. The object can be represented as a set of
    polygons
    """

    def __init__(self, polygons, size, mode):
        # assert isinstance(polygons, list), '{}'.format(polygons)
        if isinstance(polygons, list):
            polygons = [
                torch.as_tensor(p, dtype=torch.float32) for p in polygons
            ]
        elif isinstance(polygons, Polygons):
            polygons = polygons.polygons

        self.polygons = polygons
        self.size = size
        self.mode = mode

    def transpose(self, method):
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented")

        flipped_polygons = []
        width, height = self.size
        if method == FLIP_LEFT_RIGHT:
            dim = width
            idx = 0
        elif method == FLIP_TOP_BOTTOM:
            dim = height
            idx = 1

        for poly in self.polygons:
            p = poly.clone()
            TO_REMOVE = 1
            p[idx::2] = dim - poly[idx::2] - TO_REMOVE
            flipped_polygons.append(p)

        return Polygons(flipped_polygons, size=self.size, mode=self.mode)

    def crop(self, box):
        w, h = box[2] - box[0], box[3] - box[1]

        # TODO chck if necessary
        w = max(w, 1)
        h = max(h, 1)

        cropped_polygons = []
        for poly in self.polygons:
            p = poly.clone()
            p[0::2] = p[0::2] - box[0]  # .clamp(min=0, max=w)
            p[1::2] = p[1::2] - box[1]  # .clamp(min=0, max=h)
            cropped_polygons.append(p)

        return Polygons(cropped_polygons, size=(w, h), mode=self.mode)

    def resize(self, size, *args, **kwargs):
        ratios = tuple(
            float(s) / float(s_orig) for s, s_orig in zip(size, self.size))
        if ratios[0] == ratios[1]:
            ratio = ratios[0]
            scaled_polys = [p * ratio for p in self.polygons]
            return Polygons(scaled_polys, size, mode=self.mode)

        ratio_w, ratio_h = ratios
        scaled_polygons = []
        for poly in self.polygons:
            p = poly.clone()
            p[0::2] *= ratio_w
            p[1::2] *= ratio_h
            scaled_polygons.append(p)

        return Polygons(scaled_polygons, size=size, mode=self.mode)

    def convert(self, mode):
        width, height = self.size
        if mode == "mask":
            rles = mask_utils.frPyObjects([p.numpy() for p in self.polygons],
                                          height, width)
            rle = mask_utils.merge(rles)
            mask = mask_utils.decode(rle)
            mask = torch.from_numpy(mask)
            # TODO add squeeze?
            return mask

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_polygons={}, ".format(len(self.polygons))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={}, ".format(self.size[1])
        s += "mode={})".format(self.mode)
        return s


class SegmentationMask(object):
    """
    This class stores the segmentations for all objects in the image
    """

    def __init__(self, segms, size, mode=None):
        """
        Arguments:
            segms: three types
                (1) polygons: a list of list of lists of numbers. The first
                level of the list correspond to individual instances,
                the second level to all the polygons that compose the
                object, and the third level to the polygon coordinates.
                (2) rles: COCO's run length encoding format, uncompressed or compressed
                (3) binary masks
            size: (width, height)
            mode: 'polygon', 'mask'. if mode is 'mask', convert mask of any format to binary mask
        """
        assert isinstance(segms, list)
        if len(segms) == 0:
            self.masks = []
            mode = 'mask'
        else:
            if type(segms[0]) != list:
                mode = 'mask'

            if mode == 'mask':
                self.masks = [Mask(m, size, mode) for m in segms]
            else:  # polygons
                self.masks = [Polygons(p, size, mode) for p in segms]
        self.size = size
        self.mode = mode

    def transpose(self, method):
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented")

        flipped = []
        for mask in self.masks:
            flipped.append(mask.transpose(method))
        return SegmentationMask(flipped, size=self.size, mode=self.mode)

    def crop(self, box):
        w, h = box[2] - box[0], box[3] - box[1]
        cropped = []
        for mask in self.masks:
            cropped.append(mask.crop(box))
        return SegmentationMask(cropped, size=(w, h), mode=self.mode)

    def resize(self, size, *args, **kwargs):
        scaled = []
        for mask in self.masks:
            scaled.append(mask.resize(size, *args, **kwargs))
        return SegmentationMask(scaled, size=size, mode=self.mode)

    def to(self, *args, **kwargs):
        return self

    def __getitem__(self, item):
        if isinstance(item, (int, slice)):
            selected_masks = [self.masks[item]]
        else:
            # advanced indexing on a single dimension
            selected_masks = []
            if isinstance(item, torch.Tensor) and item.dtype == torch.uint8:
                item = item.nonzero()
                item = item.squeeze(1) if item.numel() > 0 else item
                item = item.tolist()
            for i in item:
                selected_masks.append(self.masks[i])
        return SegmentationMask(selected_masks, size=self.size, mode=self.mode)

    def __iter__(self):
        return iter(self.masks)

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_instances={}, ".format(len(self.masks))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={})".format(self.size[1])
        return s


def annList2maskList(annList, box=False, color=False):
    n_anns = len(annList)
    if n_anns == 0:
        return {"mask": None}

    ann = annList[0]
    try:
        h, w = ann["mask"].shape
    except:
        h, w = ann["height"], ann["width"]
    maskList = np.zeros((h, w, n_anns), int)
    categoryList = np.zeros(n_anns, int)
    for i in range(n_anns):
        ann = annList[i]

        if "mask" in ann:
            ann_mask = ann["mask"]
        else:
            ann_mask = maskUtils.decode(ann["segmentation"])

        assert ann_mask.max() <= 1
        maskList[:, :, i] = ann_mask

        categoryList[i] = ann["category_id"]
    # mask[mask==1] = ann["category_id"]
    return {"maskList": maskList, "categoryList": categoryList}


def batch2annList(batch):
    annList = []
    image_id = int(batch["name"][0].replace("_", ""))
    # image_id = batch["image_id"][0]
    height, width = batch["images"].shape[-2:]

    maskObjects = batch["maskObjects"]
    maskClasses = batch["maskClasses"]
    n_objects = maskObjects[maskObjects != 255].max()

    object_uniques = np.unique(maskObjects)
    object_uniques = object_uniques[object_uniques != 0]
    id = 1
    for obj_id in range(1, n_objects + 1):
        if obj_id == 0:
            continue

        binmask = (maskObjects == obj_id)

        segmentation = maskUtils.encode(
            np.asfortranarray(ut.t2n(binmask).squeeze()))

        segmentation["counts"] = segmentation["counts"].decode("utf-8")
        uniques = (binmask.long() * maskClasses).unique()
        uniques = uniques[uniques != 0]
        assert len(uniques) == 1

        category_id = uniques[0].item()

        annList += [{
            "segmentation": segmentation,
            "iscrowd": 0,
            # "bbox":maskUtils.toBbox(segmentation).tolist(),
            "area": int(maskUtils.area(segmentation)),
            "id": id,
            "height": height,
            "width": width,
            "image_id": image_id,
            "category_id": category_id
        }]
        id += 1

    return annList


def test(model, val_set, metric="bbox"):
    pass


@torch.no_grad()
def validate(model, val_set, method="annList_box"):
    n_batches = len(val_set)

    pred_annList = []
    gt_annList = []
    for i in range(n_batches):
        batch = ut.get_batch(val_set, [i])
        print(i, "/", n_batches)
        pred_dict = model.predict(batch, method=method)
        assert batch["name"][0] not in model.trained_batch_names

        pred_annList += pred_dict["annList"]

    results = compare_annList(gt_annList, pred_annList, val_set)
    results_dict = results["result_dict"]

    return results_dict


@torch.no_grad()
def valBatch(model, batch, method="annList_box"):
    pred_annList = []
    gt_annList = []

    pred_annList += model.predict(batch, method=method)
    gt_annList += batch["annList"]

    result_dict = compare_annList(gt_annList, pred_annList)

    return result_dict


def pred2annList(boxes_yxyx, scoreList, categoryList, batch, mask=None):
    image_shape = batch["images"].shape
    _, _, h, w = image_shape

    boxes_yxyx_denorm = bbox_yxyx_denormalize(boxes_yxyx, image_shape)
    boxes_xyhw = ut.t2n(yxyx2xywh(boxes_yxyx_denorm))

    annList = []
    for i in range(len(boxes_xyhw)):
        ann = {
            "bbox": list(map(int, boxes_xyhw[i])),
            "image_id": batch["meta"]["image_id"][0],
            "category_id": int(categoryList[i]),
            "height": h,
            "width": w,
            "score": float(scoreList[i])
        }

        annList += [ann]

    return annList


def segm2annList(segm,
                 boxes,
                 scoreList,
                 categoryList,
                 H,
                 W,
                 image_id,
                 mode="yxyx",
                 mask=None,
                 score_threshold=None):
    if len(boxes) == 0:
        return []
    if boxes.max() < 1:
        boxes_denorm = bbox_yxyx_denormalize(boxes, (1, 3, H, W))
    else:
        boxes_denorm = boxes

    if mode == "yxyx":
        boxes_xywh = ut.t2n(yxyx2xywh(boxes_denorm))
    else:
        boxes_xywh = ut.t2n(xyxy2xywh(boxes_denorm))

    annList = []

    for i in range(len(boxes_xywh)):
        score = float(scoreList[i])
        if score_threshold is not None and score < score_threshold:
            continue
        ann = {
            "segmentation": segm[i],
            "bbox": list(map(int, boxes_xywh[i])),
            "image_id": image_id,
            "category_id": int(categoryList[i]),
            "height": H,
            "width": W,
            "score": score
        }

        annList += [ann]

    return annList


def bbox2annList(boxes,
                 scoreList,
                 categoryList,
                 H,
                 W,
                 image_id,
                 mode="yxyx",
                 mask=None):
    if len(boxes) == 0:
        return []
    if boxes.max() < 1:
        boxes_denorm = bbox_yxyx_denormalize(boxes, (1, 3, H, W))
    else:
        boxes_denorm = boxes

    if mode == "yxyx":
        boxes_xywh = ut.t2n(yxyx2xywh(boxes_denorm))
    else:
        boxes_xywh = ut.t2n(xyxy2xywh(boxes_denorm))

    annList = []
    for i in range(len(boxes_xywh)):
        ann = {
            "bbox": list(map(int, boxes_xywh[i])),
            "image_id": image_id,
            "category_id": int(categoryList[i]),
            "height": H,
            "width": W,
            "score": float(scoreList[i])
        }

        annList += [ann]

    return annList


# def annList2annDict(annList, type="instances"):
#     # type = "instances or bbox"
#     if isinstance(annList[0], list):
#         annList = annList[0]

#     annDict = {}

#     annDict["categories"] = [{"id":category_id} for category_id in
#                               np.unique([a["category_id"] for a in annList])]
#     try:
#         annDict["images"] = [{"file_name":a["image_id"],
#                         "id":a["image_id"],
#                         "width":a["segmentation"]["size"][1],
#                         "height":a["segmentation"]["size"][0]} for a in annList]
#     except:
#         annDict["images"] = [{"file_name":a["image_id"],
#                         "id":a["image_id"],
#                         "width":a["width"],
#                         "height":a["height"]} for a in annList]
#     annDict["type"] = type
#     if "id" not in annList[0]:
#         for i, ann in enumerate(annList):
#             ann["id"] = i

#     annDict["annotations"] = annList

#     return annDict


def bbox2mask(bbox, image_shape, window_box=None, mode="yxyx"):
    # bbox = ut.t2n(bbox)
    _, _, h, w = image_shape

    if bbox.max() <= 1.:
        bbox = ut.t2n(
            bbox_yxyx_denormalize(bbox.cpu(), image_shape, window_box))
    mask = np.zeros((h, w), int)

    for i in range(bbox.shape[0]):
        if mode == "xyxy":
            x1, y1, x2, y2 = map(int, bbox[i])
        else:
            y1, x1, y2, x2 = map(int, bbox[i])
        # print(y1,x1,y2,x2)
        mask[y1:y2, x1] = 1
        mask[y1:y2, x2] = 1

        mask[y1, x1:x2] = 1
        mask[y2, x1:x2] = 1

    return mask


def clamp_boxes_yxyx(boxes, image_shape):
    _, _, H, W = image_shape
    # Height
    boxes[:, 0] = boxes[:, 0].clamp(0, H - 1)
    boxes[:, 2] = boxes[:, 2].clamp(0, H - 1)

    # Width
    boxes[:, 1] = boxes[:, 1].clamp(0, W - 1)
    boxes[:, 3] = boxes[:, 3].clamp(0, W - 1)

    return boxes


def apply_delta_on_bbox(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, 4] where each row is y1, x1, y2, x2
    deltas: [N, 4] where each row is [dy, dx, log(dh), log(dw)]
    """
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width

    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= torch.exp(deltas[:, 2])
    width *= torch.exp(deltas[:, 3])

    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = torch.stack([y1, x1, y2, x2], dim=1)

    return result


def compute_bbox_delta(b1_yxyx, b2_yxyx):
    """Applies the given deltas to the given boxes.
    boxes 1: [N, 4] where each row is y1, x1, y2, x2
    boxes 2: [N, 4] where each row is y1, x1, y2, x2
    """

    b1_dict = bbox_yxyx_dict(ut.t2n(b1_yxyx))
    b2_dict = bbox_yxyx_dict(ut.t2n(b2_yxyx))

    y1 = (b1_dict["yc"] - b2_dict["yc"]) / b2_dict["h"]
    x1 = (b1_dict["xc"] - b2_dict["xc"]) / b2_dict["w"]
    y2 = np.log(b1_dict["h"] / b2_dict["h"])
    x2 = np.log(b1_dict["w"] / b2_dict["w"])

    return torch.FloatTensor([y1, x1, y2, x2]).t()


def compute_overlaps_yxyx(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].

    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou_yxyx(box2, boxes1, area2[i], area1)
    return overlaps


def compute_iou_yxyx(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
          efficency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def yxyx2xywh(boxes_yxyx):
    y1, x1, y2, x2 = torch.chunk(boxes_yxyx, chunks=4, dim=1)
    h = y2 - y1
    w = x2 - x1

    return torch.cat([x1, y1, w, h], dim=1)


def xyxy2xywh(boxes_xyxy):
    x1, y1, x2, y2 = torch.chunk(boxes_xyxy, chunks=4, dim=1)
    h = y2 - y1
    w = x2 - x1

    return torch.cat([x1, y1, w, h], dim=1)


def bbox_yxyx_normalize(bbox, image_shape):
    _, _, H, W = image_shape
    scale = torch.FloatTensor([H, W, H, W])
    return bbox / scale


def bbox_yxyx_denormalize(bbox, image_shape, window_box=None, clamp=True):
    _, _, H, W = image_shape

    if window_box is None:
        window_box = [0, 0, H, W]
    else:
        window_box = list(map(int, window_box))

    H_window_box = window_box[2] - window_box[0]
    W_window_box = window_box[3] - window_box[1]

    scales = torch.FloatTensor(
        [H_window_box, W_window_box, H_window_box, W_window_box])

    shift = torch.FloatTensor(
        [window_box[0], window_box[1], window_box[0], window_box[1]])
    # Translate bounding boxes to image domain
    bbox = bbox * scales + shift
    if clamp:
        bbox = clamp_boxes_yxyx(bbox, image_shape)

    return bbox


def bbox_yxyx_dict(bbox):
    h = bbox[:, 2] - bbox[:, 0]
    w = bbox[:, 3] - bbox[:, 1]
    yc = bbox[:, 0] + 0.5 * h
    xc = bbox[:, 1] + 0.5 * w

    return {"xc": xc, "yc": yc, "h": h, "w": w}


def bbox_xywh_dict(bbox, H, W):
    pass


def bbox_yxyx_shape2shape(bbox, shape1, shape2):
    bbox = ut.n2t(bbox).float()
    bbox_norm = bbox_yxyx_normalize(bbox, shape1)
    return bbox_yxyx_denormalize(bbox_norm, shape2)


def annList2bbox(annList, mode="yxyx"):
    n_objs = len(annList)

    bbox_yxyx = torch.zeros((n_objs, 4))
    bbox_yxhw = torch.zeros((n_objs, 4))
    gt_category_ids = torch.zeros((n_objs))
    seg_areas = torch.zeros((n_objs))

    # Load object bounding boxes into a data frame.
    for i, ann in enumerate(annList):
        try:
            W, H = ann["width"], ann["height"]
            x, y, w, h = ann["bbox"].flatten()
        except:
            H, W = ann["segmentation"]["size"]
            x, y, w, h = maskUtils.toBbox(ann["segmentation"])

        x1 = x
        y1 = y

        x2 = min(x + w, W - 1)
        y2 = min(y + h, H - 1)

        bbox_yxyx[i] = torch.FloatTensor((y1 / H, x1 / W, y2 / H, x2 / W))

        bbox_yxhw[i] = torch.FloatTensor((x / W, y / H, w / W, h / H))
        seg_areas[i] = h * w

        gt_category_ids[i] = ann["category_id"]

    return {
        'bbox_yxyx': bbox_yxyx,
        'bbox_yxhw': bbox_yxhw,
        'category_ids': gt_category_ids,
        'seg_areas': seg_areas
    }


def ann2poly(ann):
    mask = ann2mask(ann)["mask"]
    # f_mask = np.asfortranarray(mask)
    # e_mask = maskUtils.encode(f_mask)
    # area = maskUtils.area(e_mask)
    # bbox = maskUtils.toBbox(e_mask)
    contours = measure.find_contours(mask, 0.5)

    polyList = []
    for contour in contours:
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().astype(int)
        polyList.append(segmentation)

    return polyList


def poly2mask(poly, bbox_new):
    Rs = maskUtils.frPoly
    mo = maskUtils.decode(Rs)

    return polyList


def load_annList(exp_dict, predict_method, reset=None):
    dataset_name = exp_dict["dataset_name"]
    base = "/mnt/projects/counting/Saves/main/"

    fname = base + "lcfcn_points/{}_{}_annList.json".format(
        dataset_name, predict_method)

    if os.path.exists(fname) and reset != "reset":
        return ut.load_json(fname)

    else:
        _, val_set = load_trainval(exp_dict)

        loader = data.DataLoader(
            val_set, batch_size=1, num_workers=0, drop_last=False)

        pointDict = load_LCFCNPoints(exp_dict)

        annList = []
        for i, batch in enumerate(loader):
            print(i, "/", len(loader), " - annList")

            pointList = pointDict[batch["name"][0]]
            if len(pointList) == 0:
                continue

            if predict_method == "BestObjectness":
                pred_dict = pointList2BestObjectness(pointList, batch)
            elif predict_method == "UpperBound":
                pred_dict = pointList2UpperBound(pointList, batch)

            annList += pred_dict["annList"]

        ut.save_json(fname, annList)
        return annList


def load_BestObjectness(exp_dict, reset=None):
    return load_annList(
        exp_dict, predict_method="BestObjectness", reset=reset)


def load_UpperBound(exp_dict, reset=None):
    return load_annList(exp_dict, predict_method="UpperBound", reset=reset)


def get_perSizeResults(gt_annDict, pred_annList):
    cocoGt = pycocotools.coco.COCO(gt_annDict)
    # pred_annList2 = []

    cocoDt = cocoGt.loadRes(pred_annList)

    cocoEval = COCOeval(cocoGt, cocoDt, "segm")
    cocoEval.params.iouThrs = np.array([.25, .5, .75])

    cocoEval.evaluate()
    cocoEval.accumulate()

    results = cocoEval.summarize()

    result_dict = {}
    for i in ["0.25", "0.5", "0.75"]:
        score = results["{}_all".format(i)]
        result_dict[i] = score

    return {"results": results, "result_dict": result_dict}


def get_perCategoryResults(gt_annDict, pred_annDict):
    cocoGt = pycocotools.coco.COCO(gt_annDict)
    cocoDt = cocoGt.loadRes(pred_annDict)

    cocoEval = COCOeval(cocoGt, cocoDt, "segm")
    results = {}
    for i in cocoEval.params.catIds:
        cocoEval = COCOeval(cocoGt, cocoDt, "segm")
        cocoEval.params.iouThrs = np.array([.5])
        cocoEval.params.catIds = [i]
        cocoEval.params.areaRngLbl = ["all"]

        cocoEval.evaluate()
        cocoEval.accumulate()

        stat = list(cocoEval.summarize().values())
        assert len(stat) == 1
        results[i] = stat[0]

    return results


def get_image_ids(pred_annList):
    idList = set()
    for p in pred_annList:
        idList.add(p["image_id"])

    return list(idList)


# def pred_for_coco2014(exp_dict, pred_annList):
#     if exp_dict["dataset_name"] == "CocoDetection2014":
#         train_set,_ = ut.load_trainval(exp_dict)
#         for p in pred_annList:
#             p["image_id"] = int(p["image_id"])
#             p["category_id"] = train_set.label2category[p["category_id"]]

#     return pred_annList


def test_baselines(exp_dict, reset=None):
    #### Best Objectness

    # pointDict = load_LCFCNPoints(exp_dict)
    pred_annList = load_UpperBound(exp_dict, reset=reset)
    if os.path.exists(exp_dict["path_baselines"]) and reset != "reset":
        result_list = ut.load_pkl(exp_dict["path_baselines"])
        return result_list

    else:
        gt_annDict = load_gtAnnDict(exp_dict)
        pred_annList = load_BestObjectness(exp_dict, reset=reset)

        # idList1 = get_image_ids(pred_annList)
        # idList2 = get_image_ids(gt_annDict["annotations"])

        results = get_perSizeResults(gt_annDict, pred_annList)
        result_dict = results["result_dict"]

        result_dict["predict_method"] = "BestObjectness"
        result_list = [result_dict]

        #### Upper bound

        pred_annList = load_UpperBound(exp_dict, reset=reset)
        results = get_perSizeResults(gt_annDict, pred_annList)

        result_dict = results["result_dict"]

        result_dict["predict_method"] = "UpperBound"
        result_list += [result_dict]
        ut.save_pkl(exp_dict["path_baselines"], result_list)

    print(pd.DataFrame(result_list))
    return result_list


def validate(model, dataset, predict_method, n_val=None, return_annList=False):
    pred_annList = dataset2annList(
        model, dataset, predict_method=predict_method, n_val=n_val)

    gt_annDict = load_gtAnnDict({"dataset_name": type(dataset).__name__})
    results = get_perSizeResults(gt_annDict, pred_annList)

    result_dict = results["result_dict"]
    result_dict["predict_method"] = predict_method

    if return_annList:
        return result_dict, pred_annList

    return result_dict


def test_best(exp_dict, reset=None):
    _, val_set = load_trainval(exp_dict)

    history = ut.load_history(exp_dict)

    # if reset == "reset":
    try:
        pred_annList = ut.load_best_annList(exp_dict)
    except:
        model = ut.load_best_model(exp_dict)
        pred_annList = dataset2annList(
            model, val_set, predict_method="BestDice", n_val=None)
        ut.save_pkl(exp_dict["path_best_annList"], pred_annList)
    # else:
    # pred_annList = ut.load_best_annList(exp_dict)

    gt_annDict = load_gtAnnDict(exp_dict)
    results = get_perSizeResults(gt_annDict, pred_annList)

    result_dict = results["result_dict"]
    result_dict["predict_method"] = "BestDice"
    result_dict["epoch"] = history["best_model"]["epoch"]
    result_list = test_baselines(exp_dict)
    result_list += [result_dict]

    print(pd.DataFrame(result_list))


def get_random_indices(mask, n_indices=10):
    mask_ind = np.where(mask.squeeze())
    n_pixels = mask_ind[0].shape[0]
    P_ind = np.random.randint(0, n_pixels, n_indices)
    yList = mask_ind[0][P_ind]
    xList = mask_ind[1][P_ind]

    return {"yList": yList, "xList": xList}


def propDict2seedList(propDict, n_neighbors=100, random_proposal=False):
    seedList = []
    for prop in propDict["propDict"]:
        if len(prop["annList"]) == 0:
            seedList += [{
                "category_id": [prop["point"]["category_id"]],
                "yList": [prop["point"]["y"]],
                "xList": [prop["point"]["x"]],
                "neigh": {
                    "yList": [prop["point"]["y"]],
                    "xList": [prop["point"]["x"]]
                }
            }]

        else:
            if random_proposal:
                i = np.random.randint(0, len(prop["annList"]))
                mask = prop["annList"][i]["mask"]
            else:
                mask = prop["annList"][0]["mask"]

            seedList += [{
                "category_id": [prop["point"]["category_id"]],
                "yList": [prop["point"]["y"]],
                "xList": [prop["point"]["x"]],
                "neigh": get_random_indices(mask, n_indices=100)
            }]

    # Background
    background = propDict["background"]
    if background.sum() == 0:
        y_axis = np.random.randint(0, background.shape[1], 100)
        x_axis = np.random.randint(0, background.shape[2], 100)
        background[0, y_axis, x_axis] = 1
    bg_seeds = get_random_indices(
        background, n_indices=len(propDict["propDict"]))
    seedList += [{
        "category_id": [0] * len(bg_seeds["yList"]),
        "yList": bg_seeds["yList"].tolist(),
        "xList": bg_seeds["xList"].tolist(),
        "neigh": get_random_indices(background, n_indices=100)
    }]

    return seedList


def CombineSeeds(seedList, ind=None):
    yList = []
    xList = []
    categoryList = []

    if ind is None:
        ind = range(len(seedList))

    for i in ind:
        yList += seedList[i]["yList"]
        xList += seedList[i]["xList"]
        categoryList += seedList[i]["category_id"]

    assert len(categoryList) == len(yList)
    return {"yList": yList, "xList": xList, "categoryList": categoryList}


# 0. load val
def load_trainval(exp_dict):
    path_datasets = "datasets"
    path_transforms = 'addons/transforms.py'
    dataset_dict = ut.get_module_classes(path_datasets)
    transform_dict = ut.get_functions(path_transforms)
    dataset_name = exp_dict["dataset_name"]
    train_set, val_set = ut.load_trainval({
        "dataset_name": dataset_name,
        "path_datasets": path_datasets,
        "trainTransformer": "Tr_WTP_NoFlip",
        "testTransformer": "Te_WTP",
        "dataset_options": {},
        "dataset_dict": dataset_dict,
        "transform_dict": transform_dict
    })

    annList_path = val_set.path + "/annotations/{}_gt_annList.json".format(
        val_set.split)
    val_set.annList_path = annList_path

    return train_set, val_set


# 1. Load gtAnnDict
def load_gtAnnDict(exp_dict, reset=None):
    reset = None
    _, val_set = load_trainval(exp_dict)
    annList_path = val_set.annList_path

    if os.path.exists(annList_path) and reset != "reset":
        return ut.load_json(annList_path)

    else:
        ann_json = {}
        ann_json["categories"] = val_set.categories
        ann_json["type"] = "instances"

        # Images
        imageList = []
        annList = []
        id = 1

        for i in range(len(val_set)):
            print("{}/{}".format(i, len(val_set)))
            batch = val_set[i]

            image_id = batch["name"]

            height, width = batch["images"].shape[-2:]
            imageList += [{
                "file_name": batch["name"],
                "height": height,
                "width": width,
                "id": batch["name"]
            }]

            maskObjects = batch["maskObjects"]
            maskClasses = batch["maskClasses"]
            n_objects = maskObjects[maskObjects != 255].max().item()

            for obj_id in range(1, n_objects + 1):
                if obj_id == 0:
                    continue

                binmask = (maskObjects == obj_id)
                segmentation = maskUtils.encode(
                    np.asfortranarray(ms.t2n(binmask)))
                segmentation["counts"] = segmentation["counts"].decode("utf-8")
                uniques = (binmask.long() * maskClasses).unique()
                uniques = uniques[uniques != 0]
                assert len(uniques) == 1

                category_id = uniques[0].item()

                annList += [{
                    "segmentation": segmentation,
                    "iscrowd": 0,
                    # "bbox":maskUtils.toBbox(segmentation).tolist(),
                    "area": int(maskUtils.area(segmentation)),
                    "id": id,
                    "image_id": image_id,
                    "category_id": category_id
                }]
                id += 1

        ann_json["annotations"] = annList
        ann_json["images"] = imageList

        ut.save_json(annList_path, ann_json)

        # Save dummy results
        anns = ut.load_json(annList_path)
        fname_dummy = annList_path.replace(".json", "_best.json")
        annList = anns["annotations"]
        for a in annList:
            a["score"] = 1

        ut.save_json(fname_dummy, annList)


# 1. Load dummyAnnDict
def assert_gtAnnDict(exp_dict, reset=None):
    _, val_set = load_trainval(exp_dict)
    annList_path = val_set.annList_path

    fname_dummy = annList_path.replace(".json", "_best.json")

    # Test should be 100
    cocoGt = pycocotools.coco.COCO(annList_path)

    imgIds = sorted(cocoGt.getImgIds())
    assert len(imgIds) == len(val_set)
    assert len(ms.load_json(fname_dummy)) == len(
        ut.load_json(annList_path)["annotations"])

    assert len(ms.load_json(fname_dummy)) == len(cocoGt.anns)
    imgIds = imgIds[0:100]
    imgIds = np.random.choice(imgIds, min(100, len(imgIds)), replace=False)
    cocoDt = cocoGt.loadRes(fname_dummy)

    cocoEval = COCOeval(cocoGt, cocoDt, "segm")
    # cocoEval.params.imgIds  = imgIds.tolist()
    cocoEval.params.iouThrs = np.array([.25, .5, .75])

    cocoEval.evaluate()
    cocoEval.accumulate()
    stats = cocoEval.summarize()

    assert stats["0.25_all"] == 1
    assert stats["0.5_all"] == 1
    assert stats["0.75_all"] == 1


def load_LCFCNPoints(exp_dict, reset=None):
    dataset_name = exp_dict["dataset_name"]
    base = "/mnt/projects/counting/Saves/main/"

    if "Pascal" in dataset_name:
        path = base + "dataset:Pascal2007_model:Res50FCN_metric:mRMSE_loss:water_loss_B_config:basic/"

    elif "CityScapes" in dataset_name:
        path = base + "dataset:CityScapes_model:Res50FCN_metric:mRMSE_loss:water_loss_B_config:basic/"

    elif "CocoDetection2014" in dataset_name:
        path = base + "dataset:CocoDetection2014_model:Res50FCN_metric:mRMSE_loss:water_loss_B_config:sample3000/"

    elif "Kitti" in dataset_name:
        path = base + "dataset:Kitti_model:Res50FCN_metric:mRMSE_loss:water_loss_B_config:basic/"

    elif "Plants" in dataset_name:
        path = base + "dataset:Plants_model:Res50FCN_metric:mRMSE_loss:water_loss_B_config:basic/"

    else:
        raise

    fname = base + "lcfcn_points/{}.pkl".format(dataset_name)

    if os.path.exists(fname):
        history = ut.load_pkl(path + "history.pkl")
        pointDict = ut.load_pkl(fname)

        if pointDict["best_model"]["epoch"] != history["best_model"]["epoch"]:
            reset = "reset"

    if os.path.exists(fname) and reset != "reset":
        return pointDict

    else:
        train_set, val_set = load_trainval(exp_dict)

        # Create Model
        model = exp_dict["model_dict"]["Res50FCN"](train_set)
        model.load_state_dict(torch.load(path + "/State_Dicts/best_model.pth"))
        history = ut.load_pkl(path + "history.pkl")
        model.cuda()

        loader = data.DataLoader(
            val_set, batch_size=1, num_workers=1, drop_last=False)
        pointDict = {}
        model.eval()
        for i, batch in enumerate(loader):
            print(i, "/", len(loader), " - pointDict")
            pointList = model.predict(
                batch, predict_method="points")["pointList"]
            pointDict[batch["name"][0]] = pointList

        pointDict["best_model"] = history['best_model']
        pointDict['exp_dict'] = history['exp_dict']

        ut.save_pkl(fname, pointDict)

        return pointDict


def blobs2annList(blobs, image_id):
    n_classes, h, w = blobs.shape
    annList = []

    for i in range(n_classes):

        blobs_class = blobs[i]
        for u in np.unique(blobs_class):
            if u == 0:
                continue

            binmask = (blobs_class == u).astype(int)

            seg = maskUtils.encode(
                np.asfortranarray(ms.t2n(binmask.squeeze())).astype("uint8"))
            seg["counts"] = seg["counts"].decode("utf-8")
            score = 1.0

            annList += [{
                "segmentation": seg,
                "bbox": maskUtils.toBbox(seg).astype(int).tolist(),
                "iscrowd": 0,
                "area": int(maskUtils.area(seg)),
                "image_id": image_id,
                "category_id": i + 1,
                "height": h,
                "width": w,
                "score": score
            }]
    return annList


def blobs2BestDice(blobs, categoryDict, propDict, batch):
    h, w = blobs.shape
    annList = []
    blobs_copy = np.zeros(blobs.shape, int)

    if "maskVoid" in batch:
        maskVoid = batch["maskVoid"]
    else:
        maskVoid = None

    for u in np.unique(blobs):
        if u == 0:
            continue
        binmask = (blobs == u)
        best_dice = 0.
        best_mask = None
        for ann in propDict['propDict'][u - 1]["annList"]:

            score = dice(ann["mask"], binmask)
            if score > best_dice:
                best_dice = score
                best_mask = ann["mask"]
                prop_score = ann["score"]

        if best_mask is None:
            best_mask = (blobs == u).astype(int)

        if maskVoid is not None:
            binmask = best_mask * (ms.t2n(maskVoid).squeeze())
        else:
            binmask = best_mask

        if best_mask is None:
            blobs_copy[blobs == u] = u
        else:
            blobs_copy[best_mask == 1] = u

        seg = maskUtils.encode(
            np.asfortranarray(ms.t2n(binmask)).astype("uint8"))
        seg["counts"] = seg["counts"].decode("utf-8")
        score = best_dice

        # if batch["dataset"] == "coco2014":
        #     image_id = int(batch["name"][0])
        # else:
        image_id = batch["name"][0]

        annList += [{
            "segmentation": seg,
            "iscrowd": 0,
            "area": int(maskUtils.area(seg)),
            "image_id": image_id,
            "category_id": int(categoryDict[u]),
            "height": h,
            "width": w,
            "score": score
        }]

    return {"blobs": blobs_copy, "annList": annList}


@torch.no_grad()
def dataset2annList(model,
                    dataset,
                    predict_method="BestObjectness",
                    n_val=None):
    loader = data.DataLoader(
        dataset, batch_size=1, num_workers=1, drop_last=False)

    annList = []
    for i, batch in enumerate(loader):
        print(i, "/", len(loader))
        pred_dict = model.predict(batch, predict_method="BestDice")

        annList += pred_dict["annList"]
    return annList


def pointList2mask(pointList):
    if "shape" in pointList[0]:
        shape = pointList[0]
    else:
        shape = (1, pointList[0]["h"], pointList[0]["w"])

    _, h, w = shape
    mask = np.zeros(shape, "uint8")
    for p in pointList:
        if "category_id" in p:
            category_id = p["category_id"]
        else:
            category_id = p["cls"]
        if p["y"] < 1:
            y = int(p["y"] * h)
            x = int(p["x"] * w)
        else:
            y = int(p["y"])
            x = int(p["x"])
        mask[:, y, x] = int(category_id)

    return {"mask": mask}


def pointList2annList(pointList):
    annList = []
    for p in pointList:
        binmask = np.zeros((p["h"], p["w"]), "uint8")
        binmask[int(p["y"] * p["h"]),
                int(p["x"] * p["w"])] = 1.
        annList += [
            mask2ann(
                binmask,
                p["cls"],
                image_id=-1,
                maskVoid=None,
                score=None,
            )]

    return annList


def pointList2points(pointList):
    return pointList2mask(pointList)


def print_results(results):
    pass


def probs2blobs(probs):
    annList = []

    probs = ut.t2n(probs)
    n, n_classes, h, w = probs.shape

    counts = np.zeros((n, n_classes - 1))

    # Binary case
    pred_mask = ut.t2n(probs.argmax(1))
    blobs = np.zeros(pred_mask.shape)
    points = np.zeros(pred_mask.shape)

    max_id = 0
    for i in range(n):
        for category_id in np.unique(pred_mask[i]):
            if category_id == 0:
                continue

            ind = pred_mask == category_id

            connected_components = morph.label(ind)

            uniques = np.unique(connected_components)

            blobs[ind] = connected_components[ind] + max_id
            max_id = uniques.max() + max_id

            n_blobs = (uniques != 0).sum()

            counts[i, category_id - 1] = n_blobs

            for j in range(1, n_blobs + 1):
                binmask = connected_components == j
                blob_probs = probs[i, category_id] * binmask
                y, x = np.unravel_index(blob_probs.squeeze().argmax(),
                                        blob_probs.squeeze().shape)

                points[i, y, x] = category_id
                annList += [
                    mask2ann(
                        binmask,
                        category_id,
                        image_id=-1,
                        height=binmask.shape[1],
                        width=binmask.shape[2],
                        maskVoid=None,
                        score=None,
                        point={
                            "y": y,
                            "x": x,
                            "prob": float(blob_probs[blob_probs != 0].max()),
                            "category_id": int(category_id)
                        })
                ]

    blobs = blobs.astype(int)

    # Get points

    return {
        "blobs": blobs,
        "annList": annList,
        "probs": probs,
        "counts": counts,
        "points": points,
        "pointList": mask2pointList(points)["pointList"],
        "pred_mask": pred_mask,
        "n_blobs": len(annList)
    }


def mask2pointList(mask):
    pointList = []
    mask = ut.t2n(mask)
    pointInd = np.where(mask.squeeze())
    n_points = pointInd[0].size

    for p in range(n_points):
        p_y, p_x = pointInd[0][p], pointInd[1][p]
        point_category = mask[0, p_y, p_x]

        pointList += [{
            "y": p_y,
            "x": p_x,
            "category_id": int(point_category),
            "shape": mask.shape
        }]

    return {"pointList": pointList}


def pointList2UpperBound(pointList, batch):
    propDict = pointList2propDict(pointList, batch, thresh=0.5)

    n, c = batch["counts"].shape
    n, _, h, w = batch["images"].shape

    n_objects = batch["maskObjects"].max()
    if "maskVoid" in batch:
        maskVoid = ut.t2n(batch["maskVoid"].squeeze())
    else:
        maskVoid = None

    ###########
    annList = []
    for p_index, p in enumerate(pointList):
        category_id = p["category_id"]
        best_score = 0
        best_mask = None

        gt_object_found = False
        cls_mask = (batch["maskClasses"] == category_id).long().squeeze()

        for k in range(n_objects):
            gt_mask = (batch["maskObjects"] == (k + 1)).long().squeeze()

            if (gt_mask[p["y"], p["x"]].item() != 0
                    and cls_mask[p["y"], p["x"]].item() == 1):
                gt_object_found = True

                break

        if gt_object_found == False:
            continue

        gt_mask = ut.t2n(gt_mask)
        #########################################
        best_score = 0
        best_mask = None

        for proposal_ann in propDict["propDict"][p_index]["annList"]:
            proposal_mask = proposal_ann["mask"]
            #########
            # proposal_mask = resize(proposal_mask, (h, w), order=0)
            score = dice(gt_mask, proposal_mask)

            if score > best_score:
                best_mask = proposal_mask
                best_score = score

        # ut.images(batch["images"], best_mask, denorm=1)
        if best_mask is not None:
            ann = mask2ann(
                best_mask,
                p["category_id"],
                image_id=batch["name"][0],
                height=batch["images"].shape[2],
                width=batch["images"].shape[3],
                maskVoid=maskVoid,
                score=best_score)

            annList += [ann]

    return {"annList": annList}


def maskBatch2ann(maskClasses, maskObjects, image_id, maskVoid=None):
    binmask = None
    binmask = binmask.squeeze().astype("uint8")
    height, width = binmask.shape

    if maskVoid is not None:
        binmask = binmask * maskVoid

    segmentation = maskUtils.encode(
        np.asfortranarray(ms.t2n(binmask)).astype("uint8"))
    segmentation["counts"] = segmentation["counts"].decode("utf-8")

    ann = {
        "segmentation": segmentation,
        "iscrowd": 0,
        "area": int(maskUtils.area(segmentation)),
        "image_id": image_id,
        "category_id": int(category_id),
        "height": height,
        "width": width,
        "score": score,
        "point": point
    }
    return ann


def create_ann(
        category_id,
        image_id,
        height,
        width,
        maskVoid=None,
        xywh=None,
        binmask=None,
        score=None,
        point=None):
    # bbox: x,y,w,h
    if xywh is not None:
        x, y, w, h = xywh
        area = w * h
        bbox = (x, y, w, h)

    if binmask is not None:
        binmask = binmask.squeeze().astype("uint8")
        height, width = binmask.shape

        if maskVoid is not None:
            binmask = binmask * maskVoid

        segmentation = maskUtils.encode(
            np.asfortranarray(ms.t2n(binmask)).astype("uint8"))
        segmentation["counts"] = segmentation["counts"].decode("utf-8")
        area = int(maskUtils.area(segmentation))
        bbox = maskUtils.toBbox(segmentation)

    ann = {

        "iscrowd": 0,
        "bbox": bbox,
        "area": area,
        "image_id": image_id,
        "category_id": int(category_id),
        "height": height,
        "width": width,
        "score": score,
        "point": point
    }

    if binmask:
        ann["segmentation"] = segmentation

    return ann


def mask2ann(binmask,
             category_id,
             image_id,
             maskVoid=None,
             score=None,
             point=None):
    binmask = binmask.squeeze().astype("uint8")
    height, width = binmask.shape

    if maskVoid is not None:
        binmask = binmask * maskVoid

    segmentation = maskUtils.encode(
        np.asfortranarray(ut.t2n(binmask)).astype("uint8"))
    segmentation["counts"] = segmentation["counts"].decode("utf-8")

    ann = {
        "segmentation": segmentation,
        "iscrowd": 0,
        "bbox": maskUtils.toBbox(segmentation),
        "area": int(maskUtils.area(segmentation)),
        "image_id": image_id,
        "category_id": int(category_id),
        "height": height,
        "width": width,
        "score": score,
        "point": point
    }
    return ann

    # for i, prop in enumerate(propDict['propDict']):
    #     if len(prop["annList"]) == 0:
    #         continue
    #     blobs[prop["annList"][0]["mask"] !=0] = i+1

    #     categoryDict[i+1] = prop["category_id"]

    #     if maskVoid is not None:
    #         binmask = prop["annList"][0]["mask"] * (ms.t2n(maskVoid).squeeze())
    #     else:
    #         binmask = prop["annList"][0]["mask"]

    #     seg = maskUtils.encode(np.asfortranarray(ms.t2n(binmask)).astype("uint8"))
    #     seg["counts"] = seg["counts"].decode("utf-8")
    #     score = prop["annList"][0]["score"]

    #     annList += [{"segmentation":seg,
    #           "iscrowd":0,
    #           "area":int(maskUtils.area(seg)),
    #          "image_id":batch["name"][0],
    #          "category_id":int(prop['category_id']),
    #          "height":h,
    #          "width":w,
    #          "score":score}]

    #############

    # for p in blob_dict["pointList"]:

    #     category_id = p["category_id"]
    #     best_score = 0
    #     best_mask = None

    #     gt_object_found = False
    #     cls_mask = (batch["maskClasses"] == category_id).long().squeeze()

    #     for k in range(n_objects):
    #         gt_mask = (batch["maskObjects"] == (k+1)).long().squeeze()

    #         if (gt_mask[p["y"],p["x"]].item() != 0 and
    #             cls_mask[p["y"],p["x"]].item()==1):
    #             gt_object_found = True

    #             break

    #     if gt_object_found == False:
    #         continue

    #         # label_class = (pred_mask*batch["maskClasses"]).max().item()

    #     gt_mask = ut.t2n(gt_mask)
    #     #########################################

    #     best_score = 0
    #     best_mask = None

    #     for k in range(len(sharp_proposals)):
    #         proposal_ann = sharp_proposals[k]
    #         if proposal_ann["score"] < 0.5:
    #             continue

    #         proposal_mask =  proposal_ann["mask"]

    #         #########
    #         # proposal_mask = resize(proposal_mask, (h, w), order=0)
    #         score = sf.dice(gt_mask, proposal_mask)

    #         #########

    #         if score > best_score:
    #             best_mask = proposal_mask
    #             best_score = score

    #     # ut.images(batch["images"], best_mask, denorm=1)
    #     if best_mask is not None:
    #         ann = bu.mask2ann(best_mask, p["category_id"],
    #                         image_id=batch["name"][0],
    #                         height=batch["images"].shape[2],
    #                         width=batch["images"].shape[3],
    #                         maskVoid=maskVoid, score=best_score)
    #         annList += [ann]


def naive(pred_mask, gt_mask):
    return (pred_mask * gt_mask).mean()


def dice(pred_mask, gt_mask, smooth=1.):
    iflat = pred_mask.ravel()
    tflat = gt_mask.ravel()
    intersection = (iflat * tflat).sum()

    score = ((2. * intersection) / (iflat.sum() + tflat.sum() + smooth))
    return score


def cosine_similarity(pred_mask, true_mask):
    scale = np.linalg.norm(pred_mask) * np.linalg.norm(true_mask)
    return pred_mask.ravel().dot(true_mask.ravel()) / scale


# @torch.no_grad()
# def pointList2propDict(pointList, batch, single_point=False, thresh=0.5):
#     sharp_proposals = base_dataset.SharpProposals(batch)
#     propDict = []
#     shape = pointList[0]["shape"]
#     foreground = np.zeros(shape, int)

#     idDict= {}
#     annDict = {}
#     for i, p in enumerate(pointList):
#         annDict[i] = []
#         idDict[i] = []

#     for k in range(len(sharp_proposals)):
#         proposal_ann = sharp_proposals[k]
#         if not (proposal_ann["score"] > thresh):
#             continue
#         proposal_mask =  proposal_ann["mask"]

#         for i, p in enumerate(pointList):
#             if proposal_mask[p["y"], p["x"]]==0:
#                 continue

#             # score = proposal_ann["score"]

#             annDict[i] += [proposal_ann]
#             idDict[i] += [k]

#     for i in annDict:
#         annList = annDict[i]
#         idList = idDict[i]
#         p = pointList[i]

#         mask = annList2mask(annList)["mask"]
#         if mask is not None:
#             foreground = foreground + mask

#         #foreground[foreground<2]=0
#         propDict += [{"annList":annList,"point":p, "idList":idList,
#                       "category_id":int(p["category_id"])}]
#         #########

#     return {"propDict":propDict,"foreground":foreground, "background":(foreground==0).astype(int)}


def annList2mask(annList, box=False, binary=False):
    n_anns = len(annList)
    if n_anns == 0:
        return {"mask": None}

    mask = None

    for i in range(n_anns):
        ann = annList[i]
        tmp = ann2mask(ann)["mask"]
        if mask is None:
            mask = np.zeros(tmp.shape)

        mask[tmp != 0] = ann["category_id"]
        # print(i, ann["category_id"])

    if binary:
        mask = (torch.LongTensor(mask) != 0).long()
    else:
        mask = torch.LongTensor(mask)
    # mask[mask==1] = ann["category_id"]
    return {"mask": mask}


def ann2mask(ann):
    if "mask" in ann:
        mask = ann["mask"]
    elif "segmentation" in ann:
        # TODO: fix this monkey patch
        if isinstance(ann["segmentation"]["counts"], list):
            ann["segmentation"]["counts"] = ann["segmentation"]["counts"][0]
        mask = maskUtils.decode(ann["segmentation"])
    else:
        x, y, w, h = ann["bbox"]
        img_h, img_w = ann["height"], ann["width"]
        mask = np.zeros((img_h, img_w))
        mask[y:y + h, x:x + w] = 1
    # mask[mask==1] = ann["category_id"]
    return {"mask": mask}


def annList2points(annList, points=None):
    n_anns = len(annList)
    if n_anns == 0:
        return {"mask": None}

    ann = annList[0]
    img_h, img_w = ann["height"], ann["width"]
    mask = np.zeros((img_h, img_w))

    for i in range(n_anns):
        ann = annList[i]
        x, y, w, h = ann["bbox"]
        mask[(y + y + h) // 2,
             (x + x + w) // 2] = ann["category_id"]

    # mask[mask==1] = ann["category_id"]
    return mask


def annList2scores_points(annList, img_h, img_w, points=None):
    mask = np.zeros((img_h, img_w))
    n_anns = len(annList)
    if n_anns == 0:
        return mask

    ann = annList[0]
    img_h, img_w = ann["height"], ann["width"]

    for i in range(n_anns):
        ann = annList[i]
        x, y, w, h = ann["bbox"]
        mask[(y + y + h) // 2,
             (x + x + w) // 2] = ann["score"]

    # mask[mask==1] = ann["category_id"]
    return mask


def annList2scores_mask(annList, img_h, img_w, points=None):
    mask = np.zeros((img_h, img_w))
    n_anns = len(annList)
    if n_anns == 0:
        return mask

    for i in range(n_anns):
        ann = annList[i]
        x, y, w, h = ann["bbox"]
        mask[y:y + h, x:x + w] = ann["score"]

    return mask
