import torch
import torchvision

import timm

import transformers
from transformers import ViTConfig, ViTModel, ViTForImageClassification

from .transformers_models import *
from .cnn_models import *
from .resnet import *

class ViTWrapper(nn.Module):
    def __init__(self, vit_model):
        super().__init__()
        self.model = vit_model

    def forward(self, x):
        return self.model(x).logits   # retourne directement le Tensor


def build_model(cfg, input_size, num_classes):
    modelname = cfg['class']
    if "smallresnet" == modelname.lower() or "mycifar" in modelname.lower() :
        return eval(f"{modelname}(cfg, num_classes)")
    elif "loadresnet" == modelname.lower():
        model = torchvision.models.resnet34(pretrained=False, num_classes=num_classes)
        model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = torch.nn.Identity()
        return model
    elif "timmresnet" == modelname.lower():
        return timm.create_model("resnet34",pretrained=False,num_classes=100)
    elif "loadvit" == modelname.lower():
        config = ViTConfig(hidden_size=192,
                            num_hidden_layers=9,
                            num_attention_heads=3,
                            intermediate_size=768, # intermediate_size = hidden_size * 4
                            hidden_dropout_prob = 0.1,
                            attention_dropout = 0.1,
                            image_size=32,      # CIFAR100
                            num_labels=100,
                            patch_size=4,       
                            num_channels=3,
                            qkv_bias=True,
                            classifier="token"
                            )
        model = ViTForImageClassification(config)
        return ViTWrapper(model)
    else:
        return eval(f"{modelname}(cfg, input_size, num_classes)")



