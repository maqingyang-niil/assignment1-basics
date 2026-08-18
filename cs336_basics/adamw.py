import torch
from typing import Optional
from collections.abc import Callable, Iterable
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self,params,lr=1e-3,betas:tuple[float,float]=(0.9,0.999),eps=1e-3,weight_decay=0.0):
        if lr<0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults={"lr":lr,
                  "betas":betas,
                  "eps":eps,
                  "weight_decay":weight_decay}
        super().__init__(params,defaults)

    def step(self,closure:Optional[callable]=None):
        loss=None if closure is None else closure()
        for group in self.param_groups:
            lr=group["lr"]
            beta1,beta2=group["betas"]
            eps=group["eps"]
            weight_decay=group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad=p.grad.data
                state=self.state[p]
                if len(state)==0:
                    state["t"]=0
                    state["m"]=torch.zeros_like(p.data)
                    state["v"]=torch.zeros_like(p.data)
                t=state["t"]+1
                m=state["m"]
                v=state["v"]
                m=beta1*m+(1-beta1)*grad
                v=beta2*v+(1-beta2)*grad**2
                new_lr=lr*math.sqrt(1-beta2**t)/(1-beta1**t)
                p.data-=lr*weight_decay*p.data
                p.data-=new_lr*m/(torch.sqrt(v)+eps)
                state["t"]=t
                state["m"]=m
                state["v"]=v
        return loss



