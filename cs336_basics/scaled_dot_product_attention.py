import torch.nn as nn
import torch
from einops import einsum
import math
from cs336_basics.softmax import softmax

def scaled_dot_product_attention(Q:torch.Tensor,
                                 K:torch.Tensor,
                                 V:torch.Tensor,
                                 mask:torch.Tensor|None=None)->torch.Tensor:
    d_k=Q.shape[-1]

    c=einsum(Q,K,"... seq_q d_k,... seq_k d_k->... seq_q seq_k")/math.sqrt(d_k)

    if mask is not None:
        c=c.masked_fill(mask==False,float("-inf"))

    attn=softmax(c,dim=-1)
    out=einsum(attn,V,"... seq_q seq_k, ... seq_k d_v->... seq_q d_v")
    return out