import torch.nn as nn
import torch
from cs336_basics.multihead_self_attention import MultiHeadSelfAttentionRoPE
from cs336_basics.rmsnorm import Rmsnorm
from cs336_basics.swiglu import SwiGLU

class TransformerBlock(nn.Module):
    def __init__(self,d_model:int,num_heads:int,d_ff:int,
                 theta:float,max_seq_len:int,
                 device:torch.device|None=None,
                 dtype:torch.dtype|None=None):
        super().__init__()
        assert d_model%num_heads==0,"d_model must be divisible by num_heads"
        self.d_model=d_model
        self.num_heads=num_heads
        self.d_k=d_model//num_heads
        self.d_v=d_model//num_heads
        self.theta=theta
        self.max_seq_len=max_seq_len

        self.ln1=Rmsnorm(d_model,device=device,dtype=dtype)
        self.ln2=Rmsnorm(d_model,device=device,dtype=dtype)
        self.attn=MultiHeadSelfAttentionRoPE(d_model,num_heads,theta,max_seq_len,device=device,dtype=dtype)
        self.ffn=SwiGLU(d_model,d_ff,device=device,dtype=dtype)

    def forward(self,x:torch.Tensor,token_positions:torch.Tensor|None=None)->torch.Tensor:
        x=x+self.attn(self.ln1(x),token_positions)
        x=x+self.ffn(self.ln2(x))
        return x
