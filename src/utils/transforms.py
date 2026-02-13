import torch
import numpy as np

#mixup: BEYOND EMPIRICAL RISK MINIMIZATION (https://arxiv.org/pdf/1710.09412)
# Inspired from the official repo : https://github.com/facebookresearch/mixup-cifar10/blob/main/train.py
class Mixup:
    def __init__(self, criterion, alpha):
        self.criterion = criterion
        self.alpha = alpha
        self.lam = None
        self.y_a = None
        self.y_b = None

    def mixup_data(self, x, y):
        if self.alpha > 0:
            self.lam = np.random.beta(self.alpha, self.alpha)   # Tire un random dans la Beta distribution
                                                                # Distib. in sharped 'U' shape, with very high probability at the extremums, so lambda=0 and lambda=1
                                                                # if alpha = 1, then lam is picked out from a Uniform Distrib, so the mix% is equally random.
                                                                # if alpha < 1, for example 0.1, we have f(x) prop. to λ^(-0.9) * (1 - λ)^(-0.9)
                                                                # so, the lower is alpha, the poorest is the mix
        else:
            self.lam = 1                                        # if lam=1, no mix!

        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(x.device)         # pick out a random input in the batch

        mixed_x = self.lam * x + (1 - self.lam) * x[index, :]   # x' = λx_1 + (1 - λ)x_2
        self.y_a, self.y_b = y, y[index]                        # Corresponding targets of the mixed inputs (nb : 2)
        return mixed_x

    def mixup_criterion(self, pred):
        return self.lam * self.criterion(pred, self.y_a) + (1 - self.lam) * self.criterion(pred, self.y_b)