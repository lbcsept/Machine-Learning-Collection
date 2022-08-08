
import torch
from torch import nn


class YoloLoss(nn.Module):
    
    def __init__(self, nclass = 20, nbox =2, s_grid =7):
        super(YoloLoss, self).__init__()
        self.C = nclass
        self.B = nbox
        self.S = s_grid
        
    

    def forward(self, x):
        
        loss = None
        
        return loss