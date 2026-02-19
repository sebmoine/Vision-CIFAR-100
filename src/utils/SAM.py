import torch
from torch.nn.modules.batchnorm import _BatchNorm

# @inproceedings{foret2021sharpnessaware,
#   title={Sharpness-aware Minimization for Efficiently Improving Generalization},
#   author={Pierre Foret and Ariel Kleiner and Hossein Mobahi and Behnam Neyshabur},
#   booktitle={International Conference on Learning Representations},
#   year={2021},
#   url={https://openreview.net/forum?id=6Tm1mposlrM}
# }

# source : https://github.com/davda54/sam/blob/main/sam.py
class SAM(torch.optim.Optimizer):
    """
    Purpose : improve out-of-distribution (OOD) generalization by seeking "out parameter values whose entire neighborhoods have uniformly low training loss value (equivalently, neighborhoods having both low loss and low curvature).".
    Operation : 
        MinMax Loss :
        - Looking for the maximum value loss in the neighborhood of the current model's parameters (weights),
        - then minimize this area.
        - A direct solution to the maximization problem is very computationnally expensive, so SAM aproximate it with a single gradient ascent step (more gradietn, more sharpness) to find the perturbation (\eps).
    Limitations :
        - Double the training time : compute two gradient steps : one ascent and one descent.
        - The theoretical convergence properties of SAM are still under investigation.
        - Tuning of \rho, a new parameter for neighboorhood size.
        p=2 optimal preuve dans papier
        MAIS RHO = 0.1 ou 0.05 cifar paper
    """
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs): # default good rho in the paper, error-test based-on
                                                                                    # adaptative is better but x10 larger according to the repo
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)   # defaults params of SAM
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)   # we will use AdamW (SGD is fine)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)                  # add the base optimizer params to the SAM's params

    @torch.no_grad()
    def first_step(self, zero_grad=False):  # one gradient ascent step to approximate the maximization problem and find \eps^(^), the perturbation
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)" ==> new (modified) weights stocked in p.grad, inplace=True (add"_")

        if zero_grad: self.zero_grad() # clean the grad

    @torch.no_grad()
    def second_step(self, zero_grad=False): # The gradient descent
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update by applying the modified weights from "w" the initial position in the loss landscape

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2 # optimal value cf. paper
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups



def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)