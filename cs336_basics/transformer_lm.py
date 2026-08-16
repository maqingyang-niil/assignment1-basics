import torch.nn as nn
import torch
from cs336_basics.transformer_block import TransformerBlock
from cs336_basics.embedding import Embedding
from cs336_basics.rmsnorm import Rmsnorm
from cs336_basics.linear import Linear

class TransformerLM(nn.Module):
    def __init__(self,vocab_size:int,
                 context_length:int,
                 d_model:int,
                 num_heads:int,
                 num_layers:int,
                 d_ff:int,
                 rope_theta:float,
                 device:torch.device|None=None,
                 dtype:torch.dtype|None=None
                 ):
        super().__init__()
        self.vocab_size=vocab_size
        self.context_length=context_length
        self.d_model=d_model
        self.num_layers=num_layers
        self.d_ff=d_ff

        self.token_embedding=Embedding(vocab_size,d_model,device=device,dtype=dtype)
        self.layers=nn.ModuleList([
            TransformerBlock(
                d_model,
                num_heads,
                d_ff,
                rope_theta,
                max_seq_len=context_length,
                device=device,
                dtype=dtype
            )
            for _ in range(num_layers)
        ])
        self.ln_final=Rmsnorm(d_model=d_model,device=device,dtype=dtype)
        self.lm_head=Linear(in_features=d_model,out_features=vocab_size,device=device,dtype=dtype)

    def forward(self,in_indices:torch.Tensor)->torch.Tensor:
        ## in_indices是文字被tokenizer转化成的数字
        x=self.token_embedding(in_indices)
        for layer in self.layers:
            x=layer(x)
        x=self.ln_final(x)
        logits=self.lm_head(x)
        return logits