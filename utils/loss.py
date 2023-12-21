import torch
import torch.nn as nn
import torch.nn.functional as F
class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.smooth_loss = nn.SmoothL1Loss(reduction='sum')
        self.l1_loss = nn.L1Loss(reduction='sum')
    def forward(self,output, target,mask):
        count = torch.sum(mask)
        output_normal = output[:,:,:-1]
        output_distance = output[:,:,-1:]
        target_normal = target[:,:,:-1]
        target_distance = target[:,:,-1:]

        # loss_normal = self.l1_loss(output_normal, target_normal) / (count*3)
        # loss_distance = self.l1_loss(output_distance, target_distance) / count

        loss_normal = self.smooth_loss(output_normal, target_normal) / (count*3)
        loss_distance = self.smooth_loss(output_distance, target_distance) / count
        return loss_normal + 0.5 * loss_distance
    
class Loss_c(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()