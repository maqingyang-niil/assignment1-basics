import numpy as np
import torch

def load_memmap_datasets(path:str,dtype:np.dtype=np.uint16)->np.memmap:
    if path.endswith(".npy"):
        data=np.load(path,mmap_mode="r")
    else:
        data=np.memmap(path,dtype,mode="r")
    return data

def tokens_sanity_check(data:np.memmap,vocab_size:int,num_sample:int=1e5):
    indices=np.random.randint(0,len(data),size=min(len(data),num_sample))
    samples=data[indices]
    if samples.min()<0 or samples.max()>=vocab_size:
        raise ValueError("Detected token_id overflow. Please check the dtype used when loading the data")

def get_batch(x:np.ndarray,batch_size:int,context_length:int,device:str)->tuple[torch.Tensor,torch.Tensor]:
    n=len(x)
    if n<context_length:
        raise ValueError("input length must longer than the context lenght")
    last_start=n-context_length-1
    starts=np.random.randint(0,last_start+1,size=batch_size)
    offsets=np.arange(context_length)
    input_indices=starts[:,None]+offsets[None,:]
    target_indices=input_indices+1

    inputs=x[input_indices]
    targets=x[target_indices]

    inputs = torch.from_numpy(inputs.astype(np.int64))
    targets = torch.from_numpy(targets.astype(np.int64))

    if device.startswith("cuda") and torch.cuda.is_available():
        inputs = inputs.pin_memory().to(device, non_blocking=True)
        targets = targets.pin_memory().to(device, non_blocking=True)
    else:
        inputs=inputs.to(device)
        targets=targets.to(device)

    return inputs,targets