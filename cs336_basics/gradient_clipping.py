import torch.nn as nn
import torch
from typing import Iterable

def gradient_clip(parameters:Iterable[nn.Parameter],max_l2_norm:float):
    eps=1e-6
    parameters=[p for p in parameters if p.grad is not None]

    total_norm=torch.sqrt(sum(torch.sum(p.grad**2) for p in parameters))
    if total_norm<max_l2_norm:
        return
    for p in parameters:
        p.grad.mul_(max_l2_norm/(total_norm+eps))