import torch

from .transformers_models import *
from .cnn_models import *


def build_model(cfg, input_size, num_classes):
    return eval(f"{cfg['class']}(cfg, input_size, num_classes)")