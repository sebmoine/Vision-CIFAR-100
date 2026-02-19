import torch

def get_optimizer(cfg, params):
    opti_name = cfg["algo"]
    if cfg["SAM"]:
        return getattr(torch.optim, opti_name) # return the class AdamW not initialized
    else:
        params_dict = {k: float(v) for k, v in cfg["params"].items() if isinstance(v, str)}  # convert float le "1e-3"
        optim_class = getattr(torch.optim, opti_name)
        return optim_class(params, **params_dict)
