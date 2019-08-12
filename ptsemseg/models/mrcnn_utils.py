import numpy as np
from torchvision import transforms
from pycocotools import mask as maskUtils
import pickle, os, json
import torch
import re
import collections
from torch._six import string_classes, int_classes
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.image_list import to_image_list
import pylab as plt
import pycocotools.mask as mask_utils
from matplotlib.patches import Polygon
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def f2l(X):
    if X.ndim == 3 and (X.shape[2] == 3 or X.shape[2] == 1):
        return X
    if X.ndim == 4 and (X.shape[3] == 3 or X.shape[3] == 1):
        return X

    # CHANNELS FIRST
    if X.ndim == 3:
        return np.transpose(X, (1, 2, 0))
    if X.ndim == 4:
        return np.transpose(X, (0, 2, 3, 1))

    return X

def t2n(x):
    if isinstance(x, (int, float)):
        return x
    if isinstance(x, torch.autograd.Variable):
        x = x.cpu().data.numpy()

    if isinstance(x, (torch.cuda.FloatTensor, torch.cuda.IntTensor,
                      torch.cuda.LongTensor, torch.cuda.DoubleTensor)):
        x = x.cpu().numpy()

    if isinstance(x, (torch.FloatTensor, torch.IntTensor, torch.LongTensor,
                      torch.DoubleTensor)):
        x = x.numpy()

    return x

import os

def load_pkl(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)

def save_json(fname, data):
    create_dirs(fname)
    with open(fname, "w") as json_file:
        json.dump(data, json_file, indent=4, sort_keys=True)


def load_json(fname, decode=None):
    with open(fname, "r") as json_file:
        d = json.load(json_file)

    return d

def create_dirs(fname):
    if "/" not in fname:
        return

    if not os.path.exists(os.path.dirname(fname)):
        os.makedirs(os.path.dirname(fname))

def extract_fname(directory):
    import ntpath
    return ntpath.basename(directory)

# -=====================================
# images
import cv2
from PIL import Image

def save_image(fname, image):
    image = Image.fromarray(image.astype('uint8'))
    image.save(fname)

def get_image(image, annList, dpi=100, **options):
    image = f2l(np.array(image)).squeeze().clip(0, 255)
    if image.max() > 1:
        image /= 255.

    # box_alpha = 0.5
    # print(image.clip(0, 255).max())
    color_list = colormap(rgb=True) / 255.

    # fig = Figure()
    fig = plt.figure(frameon=False)
    canvas = FigureCanvas(fig)
    fig.set_size_inches(image.shape[1] / dpi, image.shape[0] / dpi)
    # ax = fig.gca()

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.axis('off')
    fig.add_axes(ax)
    # im = im.clip(0, 1)
    # print(image)
    ax.imshow(image)


    mask_color_id = 0
    for i in range(len(annList)):
        ann = annList[i]

        if "bbox" in ann:
            bbox = ann["bbox"]
            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2],
                              bbox[3],
                              fill=False,
                              edgecolor='r',
                              linewidth=3.0,
                              alpha=0.5))

        # if show_class:
        if options.get("show_text") == True or options.get("show_text") is None:
            score = ann["score"] or -1
            ax.text(
                bbox[0], bbox[1] - 2,
                "%.1f" % score,
                fontsize=14,
                family='serif',
                bbox=dict(facecolor='g', alpha=1.0, pad=0, edgecolor='none'),
                color='white')

        # show mask
        if "segmentation" in ann:
            mask = ann2mask(ann)["mask"]
            img = np.ones(image.shape)
            # category_id = ann["category_id"]
            # mask_color_id = category_id - 1
            # color_list = ["r", "g", "b","y", "w","orange","purple"]
            # color_mask = color_list[mask_color_id % len(color_list)]
            color_mask = color_list[mask_color_id % len(color_list), 0:3]
            mask_color_id += 1
            # print("color id: %d - category_id: %d - color mask: %s"
                        # %(mask_color_id, category_id, str(color_mask)))
            w_ratio = .4
            for c in range(3):
                color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio
            for c in range(3):
                img[:, :, c] = color_mask[c]
            e = mask

            contour, hier = cv2.findContours(e.copy(),
                                    cv2.RETR_CCOMP,
                                    cv2.CHAIN_APPROX_NONE)

            for c in contour:
                polygon = Polygon(
                    c.reshape((-1, 2)),
                    fill=True,
                    facecolor=color_mask,
                    edgecolor="white",
                    linewidth=1.5,
                    alpha=0.7
                    )
                ax.add_patch(polygon)

    canvas.draw()  # draw the canvas, cache the renderer
    width, height = fig.get_size_inches() * fig.get_dpi()
    # image = np.fromstring(canvas.tostring_rgb(), dtype='uint8')

    fig_image = np.fromstring(
        canvas.tostring_rgb(), dtype='uint8').reshape(
            int(height), int(width), 3)
    plt.close()
    # print(fig_image)
    return fig_image


def colormap(rgb=False):
    color_list = np.array([
        0.000, 0.447, 0.741, 0.850, 0.325, 0.098, 0.929, 0.694, 0.125, 0.494,
        0.184, 0.556, 0.466, 0.674, 0.188, 0.301, 0.745, 0.933, 0.635, 0.078,
        0.184, 0.300, 0.300, 0.300, 0.600, 0.600, 0.600, 1.000, 0.000, 0.000,
        1.000, 0.500, 0.000, 0.749, 0.749, 0.000, 0.000, 1.000, 0.000, 0.000,
        0.000, 1.000, 0.667, 0.000, 1.000, 0.333, 0.333, 0.000, 0.333, 0.667,
        0.000, 0.333, 1.000, 0.000, 0.667, 0.333, 0.000, 0.667, 0.667, 0.000,
        0.667, 1.000, 0.000, 1.000, 0.333, 0.000, 1.000, 0.667, 0.000, 1.000,
        1.000, 0.000, 0.000, 0.333, 0.500, 0.000, 0.667, 0.500, 0.000, 1.000,
        0.500, 0.333, 0.000, 0.500, 0.333, 0.333, 0.500, 0.333, 0.667, 0.500,
        0.333, 1.000, 0.500, 0.667, 0.000, 0.500, 0.667, 0.333, 0.500, 0.667,
        0.667, 0.500, 0.667, 1.000, 0.500, 1.000, 0.000, 0.500, 1.000, 0.333,
        0.500, 1.000, 0.667, 0.500, 1.000, 1.000, 0.500, 0.000, 0.333, 1.000,
        0.000, 0.667, 1.000, 0.000, 1.000, 1.000, 0.333, 0.000, 1.000, 0.333,
        0.333, 1.000, 0.333, 0.667, 1.000, 0.333, 1.000, 1.000, 0.667, 0.000,
        1.000, 0.667, 0.333, 1.000, 0.667, 0.667, 1.000, 0.667, 1.000, 1.000,
        1.000, 0.000, 1.000, 1.000, 0.333, 1.000, 1.000, 0.667, 1.000, 0.167,
        0.000, 0.000, 0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000,
        0.000, 0.833, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.167, 0.000,
        0.000, 0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000,
        0.833, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.167, 0.000, 0.000,
        0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833,
        0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.143, 0.143, 0.143, 0.286,
        0.286, 0.286, 0.429, 0.429, 0.429, 0.571, 0.571, 0.571, 0.714, 0.714,
        0.714, 0.857, 0.857, 0.857, 1.000, 1.000, 1.000
    ]).astype(np.float32)
    color_list = color_list.reshape((-1, 3)) * 255
    if not rgb:
        color_list = color_list[:, ::-1]
    return color_list

def ann2mask(ann):
    if "mask" in ann:
        mask = ann["mask"]
    elif "segmentation" in ann:
        # TODO: fix this monkey patch
        if isinstance(ann["segmentation"]["counts"], list):
            ann["segmentation"]["counts"] = ann["segmentation"]["counts"][0]
        mask = mask_utils.decode(ann["segmentation"])
    else:
        x,y,w,h = ann["bbox"]
        img_h, img_w = ann["height"], ann["width"]
        mask = np.zeros((img_h, img_w))
        mask[y:y+h, x:x+w] = 1
    # mask[mask==1] = ann["category_id"]
    return {"mask": mask}

# -=====================================
# collate
def collate_fn_0_4(batch, level=0):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if (isinstance(batch[0], torch.Tensor) and
            batch[0].ndimension() == 3 and
            batch[0].dtype == torch.float32):

        batch = to_image_list(batch, 32)
        return batch

    elif isinstance(batch[0], torch.Tensor):
        out = None

        return torch.stack(batch, 0, out=out)


    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)

    elif batch[0] is None:
        return batch

    elif isinstance(batch[0], list) and level == 1:
        return batch
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], BoxList):
        return batch
    elif isinstance(batch[0], collections.Mapping):
        return {key: collate_fn_0_4([d[key] for d in batch], level=level + 1) for key in batch[0]}
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [collate_fn_0_4(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))

# =============================================
# dataset utils
def random_proposal(proposals, image_id, pointList):
    if pointList is None or len(pointList) == 0:
        return []

    props_segs = [prop["segmentation"] for prop in proposals]
    point_segs = [p["segmentation"] for p in au.pointList2annList(pointList)]

    ious = maskUtils.iou(point_segs,
                         props_segs,
                         np.zeros(len(props_segs)))
    annList = []
    for i, point in enumerate(pointList):
        propList = np.array(proposals)[ious[i] != 0]
        scoreList = np.array([pr["score"] for pr in propList])
        if len(propList) == 0:
            continue
        prop = np.random.choice(propList, p=scoreList / scoreList.sum())

        mask = au.ann2mask(prop)["mask"]
        ann = au.mask2ann(mask,
                          category_id=point["cls"],
                          image_id=image_id,
                          maskVoid=None,
                          score=prop["score"],
                          point=None)
        annList += [ann]

    return annList


def bo_proposal(proposals, image_id, pointList):
    if pointList is None or len(pointList) == 0:
        return []
    props_segs = [prop["segmentation"] for prop in proposals]
    point_segs = [p["segmentation"] for p in au.pointList2annList(pointList)]

    ious = maskUtils.iou(point_segs,
                         props_segs,
                         np.zeros(len(props_segs)))

    if 1:
        annList = []
        for i, point in enumerate(pointList):
            propList = np.array(proposals)[ious[i] != 0]
            scoreList = np.array([pr["score"] for pr in propList])
            if len(propList) == 0:
                continue

            prop = propList[scoreList.argmax(0)]

            mask = au.ann2mask(prop)["mask"]
            ann = au.mask2ann(mask,
                              category_id=point["cls"],
                              image_id=image_id,
                              maskVoid=None,
                              score=prop["score"],
                              point=None)
            annList += [ann]

    return annList

class BGR_Transform(object):
    def __init__(self):
        pass
    def __call__(self, x):
        return (x * 255)[[2, 1, 0]]

def bgrNormalize():
    PIXEL_MEAN = [102.9801, 115.9465, 122.7717]
    PIXEL_STD = [1., 1., 1.]

    normalize_transform = transforms.Normalize(mean=PIXEL_MEAN, std=PIXEL_STD)

    return transforms.Compose(
        [transforms.ToTensor(),
         BGR_Transform(),
         normalize_transform])

