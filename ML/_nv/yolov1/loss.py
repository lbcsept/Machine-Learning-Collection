
import torch
from torch import nn

from utils_to_recode import intersection_over_union


class YoloLoss(nn.Module):
    
    def __init__(self, nclass = 20, nbox =2, s_grid =7, ncol_coords = 4, lambda_c = 4.0, lambda_no=1.0, **kwargs):
        super(YoloLoss, self).__init__()
        self.C = nclass
        self.B = nbox
        self.S = s_grid
        self.ncol_coords = ncol_coords
        self.lambda_c = lambda_c # coords
        self.lambda_no = lambda_no # no object
        

    def forward(self, x, target):
        
        ## # x  shape (s, s, c + b * ncol_coords) == > concat [s, s c + b1 * ncol_coords, s, s c + b2 * ncol_coords] ...
        pred_bboxes = torch.reshape(x,  shape = (-1, self.S, self.S, self.C + self.B*self.ncol_coords))
        objectness_cols = [ self.C + self.ncol_coords * i for i in range (self.B)]
        
        bious = []
        b_startc = (self.B * self.ncol_coords)
        for bi in range(self.B):
            b_endc = -(b_startc - self.ncol_coords)
            b_endc =  self.C + self.B*self.ncol_coords + 1 if b_endc<=0 else b_endc
            iou = intersection_over_union(pred_bboxes[..., -b_startc:b_endc], target[..., -self.ncol_coords:])
            bious.append(iou)
            b_startc -= self.ncol_coords

        pred_best_bbxi = torch.cat(bious, axis=-1)
        _, best_bx_per_s = torch.max(pred_best_bbxi)


        ## where are the objects in target
        exists_box = target[..., 20].unsqueeze(3) # shape : batch, s, s, 1

        ##  COORDINATES 
                 
        # batch, s, s, 1 x batch, s, s, ncol_coors ==> 1 x batch, s, s, ncol_coors
        box_tagets = exists_box * box_tagets[..., -self.ncol_coords] 

        # 
        
        box_predictions_ =  pred_bboxes[:,:,:,self.C: ].reshape(-1, self.S, self.S, self.B, self.ncol_coords)
        

        ## 

        ## no object loss
        self.lamdba_no

        ## class loss


        


        return loss