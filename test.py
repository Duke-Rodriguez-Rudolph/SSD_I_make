import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter


x=torch.randn((20))
y=torch.randn((20))
a=[x,y]
a=torch.Tensor(a)
print(a.shape)
print(torch.cat(a,dim=1).sum(dim=1).shape)
