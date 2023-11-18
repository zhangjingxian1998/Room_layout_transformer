import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
class Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,output, target,):
        loss = F.l1_loss(output, target, reduction='sum')
        return loss / output.shape[0]