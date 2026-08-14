import torch.nn as nn
import torch

class RoPE(nn.Module):
    def __init__(self,theta:float,d_k:int,max_seq_len:int,
                 device:torch.device|None=None):
        super().__init__()
        assert d_k%2==0, "d_k must be even"

        k=torch.arange(0,d_k,2,dtype=torch.float32,device=device)

        freq_pre=1.0/(theta**(k/d_k))
        positions=torch.arange(max_seq_len,dtype=torch.float32,device=device)
        freqs=torch.outer(positions,freq_pre)

        cos=torch.cos(freqs)
        sin=torch.sin(freqs)
        self.register_buffer("cos_cached",cos,persistent=False)
        self.register_buffer("sin_cached",sin,persistent=False)

    def forward(self,x:torch.Tensor,token_positions:torch.Tensor)->torch.Tensor:
        cos=self.cos_cached[token_positions]
        sin=self.sin_cached[token_positions]

        x1=x[...,0::2]
        x2=x[...,1::2]

        rotated_x1=x1*cos-x2*sin
        rotated_x2=x1*sin+x2*cos

        ## 在最后面新开一个维度，配对元素放在一起，
        ## 然后把他们按照行先序的规则压平
        out=torch.stack([rotated_x1,rotated_x2],dim=-1)
        out=out.flatten(-2)

        return out.to(x.dtype)