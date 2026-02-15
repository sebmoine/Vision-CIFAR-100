import torch

def get_optimizer(cfg, params):
    params_dict = {k: float(v) for k, v in cfg["params"].items() if k=="lr" or k=="weight_decay"}  # convert float le "1e-3"
    optim_class = getattr(torch.optim, cfg["algo"])
    return optim_class(params, **params_dict)
