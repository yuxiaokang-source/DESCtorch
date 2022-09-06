import numpy as np
import random
import os 
import torch

######################## set device and random seed ###########################
def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.badatahmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
###############################################################################


def getdims(x=(10000,200)):
    """
    This function will give the suggested nodes for each encoder layer
    return the dims for network
    """
    assert len(x)==2
    n_sample=x[0]
    if n_sample>20000:# may be need complex network
        dims=[x[-1],128,32]
    elif n_sample>10000:#10000
        dims=[x[-1],64,32]
    elif n_sample>5000: #5000
        dims=[x[-1],32,16] #16
    elif n_sample>2000:
        dims=[x[-1],128]
    elif n_sample>500:
        dims=[x[-1],64]
    else:
        dims=[x[-1],16]
    return dims