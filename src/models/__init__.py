import torch
import torchvision

import timm

from .transformers_models import *
from .cnn_models import *
from .resnet import *

def build_model(cfg, input_size, num_classes):
    modelname = cfg['class']
    if "myresnet" in modelname.lower() or "smallresnet" in modelname.lower():
        return eval(f"{modelname}(cfg, num_classes)")
    elif "loadresnet" == modelname.lower():
        model = torchvision.models.resnet34(pretrained=False, num_classes=num_classes)
        model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = torch.nn.Identity()
        return model
    elif "loadvit" == modelname.lower():
        return eval(f"{modelname}")
    else:
        return eval(f"{modelname}(cfg, input_size, num_classes)")



