import copy
import torchvision.models as models

from ptsemseg.models.fcn import *
from ptsemseg.models.vgg_osvos import *
from ptsemseg.models.dilated_fcn import *
from ptsemseg.models.dilated_fcn_highskip import *
from ptsemseg.models.reduced_fcn import *
import torch

if int(torch.__version__.split('.')[0]) == 0:
    from ptsemseg.models.seg_hrnet import *
    from ptsemseg.models.default import _C as cfg

if int(torch.__version__.split('.')[0]) > 0:
    from ptsemseg.models.mrcnn import *

def get_model(model_dict, n_classes, version=None):
    name = model_dict['arch']
    model = _get_model_instance(name)
    param_dict = copy.deepcopy(model_dict)
    param_dict.pop('arch')

    if name == "fcn8s" or name == "dilated_fcn8s" or name == "reduced_fcn8s":
        model = model(n_classes=n_classes, **param_dict)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)
    elif name == "dilated_fcn8s_highskip":
        model = model(n_classes=n_classes, **param_dict)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)
    elif name == "osvos":
        model = model(n_classes=n_classes, **param_dict)
    elif name == "hrnet" and int(torch.__version__.split('.')[0]) == 0:
        cfg.merge_from_file('/home/eren/Work/AdaptiveMaskedImprinting/ptsemseg/models/seg_hrnet_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml')
        model = HighResolutionNet(cfg)
        model.init_weights(cfg.MODEL.PRETRAINED)
    elif name == "mrcnn" and int(torch.__version__.split('.')[0]) > 0:
        exp_dict = {"model":"MRCNN",
                    "option":{"batch_size":1,
                    "pointList":"prm",
                    "backbone":"FPN_R50",
                    "proposal_name":"sm",
                    "apply_void":"True",
                    "lr":"mrcnnn"}}
        model = MRCNN(exp_dict)
    return model


def _get_model_instance(name):
    try:
        models_dict = {
            "fcn8s": fcn8s,
            "dilated_fcn8s": dilated_fcn8s,
            "reduced_fcn8s": reduced_fcn8s,
            "dilated_fcn8s_highskip": dilated_fcn8s_highskip,
            "osvos": OSVOS,
        }
        if int(torch.__version__.split('.')[0]) == 0:
            models_dict["hrnet"] = HighResolutionNet
        else:
            models_dict["mrcnn"] = MRCNN

        return models_dict[name]
    except:
        raise("Model {} not available".format(name))
