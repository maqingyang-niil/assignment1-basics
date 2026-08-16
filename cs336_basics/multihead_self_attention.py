import torch.nn as nn
import torch
from cs336_basics.scaled_dot_product_attention import scaled_dot_product_attention
from cs336_basics.linear import Linear
from cs336_basics.rope import RoPE
from einops import rearrange

class MultiHeadSelfAttention(nn.Module):
    def __init__(self,d_model:int,num_heads:int,
                 device:torch.device|None=None,
                 dtype:torch.dtype|None=None):
        super().__init__()
        assert d_model%num_heads==0, "d_model must be divisible by num_heads"
        self.d_model=d_model
        self.num_heads=num_heads
        self.d_v=d_model//num_heads
        self.d_k=d_model//num_heads

        self.qkv_proj=Linear(d_model,3*d_model,device=device,dtype=dtype)
        self.output_proj=Linear(d_model,d_model,device=device,dtype=dtype)


    def forward(self,x:torch.Tensor)->torch.Tensor:
        *batch,seq_len,_=x.shape
        qkv=self.qkv_proj(x)
        Q,K,V=qkv.split(self.d_model,dim=-1)

        Q=rearrange(Q,"... seq (h d)->... h seq d",h=self.num_heads)
        K=rearrange(K,"... seq (h d)->... h seq d",h=self.num_heads)
        V=rearrange(V,"... seq (h d)->... h seq d",h=self.num_heads)

        causal_mask=torch.tril(torch.ones(seq_len,seq_len,dtype=torch.bool,device=x.device))

        attn_output=scaled_dot_product_attention(Q,K,V,mask=causal_mask)
        attn_output=rearrange(attn_output,"... h seq d->... seq (h d)")
        return self.output_proj(attn_output)


class MultiHeadSelfAttentionRoPE(nn.Module):
    def __init__(self,d_model:int,num_heads:int,theta:float,max_seq_len:int,
                 device:torch.device|None,dtype:torch.dtype|None=None):
        super().__init__()
        assert d_model%num_heads==0, "d_model must be divisible by num_heads"
        self.d_model=d_model
        self.num_heads=num_heads
        self.d_k=d_model//num_heads
        self.d_v=d_model//num_heads

        self.qkv_proj=Linear(d_model,3*d_model,device=device,dtype=dtype)
        self.output_proj=Linear(d_model,d_model,device=device,dtype=dtype)
        self.rope=RoPE(theta=theta,d_k=self.d_k,max_seq_len=max_seq_len,device=device)

    def forward(self,x:torch.Tensor,token_positions:torch.Tensor|None=None)->torch.Tensor:
        *batch_dims,seq_len,_=x.shape
        if token_positions is None:
            token_positions=torch.arange(seq_len,device=x.device)

        qkv=self.qkv_proj(x)

        Q,K,V=qkv.split(self.d_model,dim=-1)

        Q = rearrange(Q, "... seq (h d) -> ... h seq d", h=self.num_heads)
        K = rearrange(K, "... seq (h d) -> ... h seq d", h=self.num_heads)
        V = rearrange(V, "... seq (h d) -> ... h seq d", h=self.num_heads)

        Q = self.rope(Q, token_positions.unsqueeze(-2))
        K = self.rope(K, token_positions.unsqueeze(-2))

        causal_mask=torch.tril(torch.ones(seq_len,seq_len,dtype=torch.bool,device=x.device))
        
        attn_output=scaled_dot_product_attention(Q,K,V,mask=causal_mask)
        attn_output=rearrange(attn_output,"... h seq d->... seq (h d)")
        return self.output_proj(attn_output)
        


