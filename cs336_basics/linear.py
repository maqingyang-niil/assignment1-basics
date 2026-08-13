import torch
import torch.nn as nn
from einops import einsum

class Linear(nn.Module):
    def __init__(self,in_features,out_features,device:torch.device|None=None,dtype:torch.dtype|None=None):
        super().__init__()
        self.in_features=in_features
        self.out_features=out_features
        self.W=nn.Parameter(torch.empty(out_features,in_features,device=device,dtype=dtype))
        std=(2/(in_features+out_features))**0.5
        nn.init.trunc_normal_(self.W,mean=0.0,std=std,a=-3*std,b=3*std)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        return einsum(x,self.W,"... in_features,out_features in_features->... out_features")
        