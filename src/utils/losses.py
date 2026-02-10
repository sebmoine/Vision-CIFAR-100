import torch.nn as nn

def get_loss(lossname):
    try:
        return getattr(nn, lossname)()
    except AttributeError:
        raise ValueError(f"La fonction de loss '{lossname}' n'existe pas dans torch.nn")