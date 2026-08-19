import math

def CosLRSchedule(t:int,Tw:int,Tc:int,max_lr:float,min_lr:float)->float:
    assert Tc>Tw,"Tc must bigger than Tw"
    if t<Tw:
        lr_t=max_lr*t/Tw
    elif t<=Tc:
        lr_t=min_lr+0.5*(1+math.cos(math.pi*(t-Tw)/(Tc-Tw)))*(max_lr-min_lr)
    else:
        lr_t=min_lr
    return lr_t