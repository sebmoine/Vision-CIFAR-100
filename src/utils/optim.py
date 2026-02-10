import torch
import torch.nn as nn

def get_optimizer(cfg, params):
    params_dict = {k: float(v) for k, v in cfg["params"].items()}  # convert float le "1e-3"
    optim_class = getattr(torch.optim, cfg["algo"])
    return optim_class(params, **params_dict)
