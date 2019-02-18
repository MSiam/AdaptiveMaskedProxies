import copy
import torchvision.models as models

from ptsemseg.models.fcn import *
from ptsemseg.models.vgg_osvos import *
from ptsemseg.models.dilated_fcn import *
from ptsemseg.models.dilated_fcn_highskip import *
from ptsemseg.models.reduced_fcn import *
from ptsemseg.models.light_resnet import *

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

    elif name == "lrefinenet":
        model = model(Bottleneck, [3, 4, 23, 3], n_classes=n_classes, **param_dict)
        key = '101_imagenet'
        url = models_urls[key]
        model.load_state_dict(maybe_download(key, url), strict=False)

    else:
        model = model(n_classes=n_classes, **param_dict)

    return model


def _get_model_instance(name):
    try:
        return {
            "fcn8s": fcn8s,
            "dilated_fcn8s": dilated_fcn8s,
            "reduced_fcn8s": reduced_fcn8s,
            "dilated_fcn8s_highskip": dilated_fcn8s_highskip,
            "osvos": OSVOS,
            "lrefinenet": ResNetLW,
        }[name]
    except:
        raise("Model {} not available".format(name))
