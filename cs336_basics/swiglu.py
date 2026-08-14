import torch.nn as nn
import torch
from cs336_basics.linear import Linear

class SwiGLU(nn.Module):
    def __init__(self,d_model:int,d_ff:int|None=None,
                 device:torch.device|None=None,
                 dtype:torch.dtype|None=None):
        super().__init__()
        if d_ff is None:
            d_ff=int(round((8/3)*d_model/64)*64)

        self.w1=Linear(d_model,d_ff,device=device,dtype=dtype)
        self.w2=Linear(d_ff,d_model,device=device,dtype=dtype)
        self.w3=Linear(d_model,d_ff,device=device,dtype=dtype)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        SiLU_w1x=self.w1(x)*torch.sigmoid(self.w1(x))
        gated=SiLU_w1x*self.w3(x)
        return self.w2(gated)