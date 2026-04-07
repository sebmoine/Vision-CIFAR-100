import torch.nn as nn

def get_loss(lossname):
    try:
        return getattr(nn, lossname)() #label_smoothing=0.05
    except AttributeError:
        raise ValueError(f"La fonction de loss '{lossname}' n'existe pas dans torch.nn")
