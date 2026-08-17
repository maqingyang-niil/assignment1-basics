import torch

def cross_entropy(logits:torch.Tensor,targets:torch.Tensor)->torch.Tensor:
    max_logits=torch.max(logits,dim=-1,keepdim=True).values
    stabilized=logits-max_logits
    log_sum_exp=torch.log(torch.sum(torch.exp(stabilized),dim=-1))
    target_logits=torch.gather(stabilized,dim=-1,index=targets.unsqueeze(-1)).squeeze(-1)
    losses=log_sum_exp-target_logits
    return losses.mean()
