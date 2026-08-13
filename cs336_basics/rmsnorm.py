import torch.nn as nn
import torch

class Rmsnorm(nn.Module):
    def __init__(self,d_model:int,eps:float=1e-5,
                 device:torch.device|None=None,
                 dtype:torch.dtype|None=None):
        super().__init__()
        self.d_model=d_model
        self.eps=eps
        self.g=nn.Parameter(torch.ones(d_model,device=device,dtype=dtype))

    def forward(self,x:torch.Tensor)->torch.Tensor:
        in_dtype=x.dtype
        x=x.to(torch.float32)
        rms=torch.sqrt(x.pow(2).mean(dim=-1,keepdim=True)+self.eps)
        result=x/rms*self.g
        return result.to(in_dtype)
