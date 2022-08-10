
import torch
from torch import nn

from utils_to_recode import intersection_over_union


class YoloLoss(nn.Module):
    
    def __init__(self, nclass = 20, nbox =2, s_grid =7, ncol_coords = 4, lambda_c = 4.0, lambda_no=1.0, **kwargs):
        super(YoloLoss, self).__init__()
        self.C = nclass
        self.B = nbox
        self.S = s_grid
        self.ncol_coords = ncol_coords # 5 objectness, x0, y0, w, h
        self.lambda_c = lambda_c # coords
        self.lambda_no = lambda_no # no object
        self.coord_loss_fn = nn.MSELoss(reduction="sum")
        self.class_loss_fn = nn.MSELoss(reduction="sum")
        self.objs_losses_fn = nn.MSELoss(reduction="sum") 

    def forward(self, x, target):
        
        ## # x  shape (s, s, c + b * ncol_coords) == > concat [s, s c + b1 * ncol_coords, s, s c + b2 * ncol_coords] ...
        pred_bboxes = torch.reshape(x,  shape = (-1, self.S, self.S, self.C + self.B * self.ncol_coords))
        #objectness_cols = [ self.C + self.ncol_coords * i for i in range (self.B)]

        ## ##################################################################################
        ##  COORDINATES LOSS 
        ## ##################################################################################

        ## extract coordinated from pred_bboxes and reshape so all boxes are in "parallel" on last dimension (ready for final sum)
        pred_coords = pred_bboxes[:,:,:,-(self.B*self.ncol_coords):].reshape(pred_bboxes.shape[0], self.S, self.S, self.B, self.ncol_coords)

        ## compute iou of all boxes vs target
        bious = []
        for bi in range(self.B):
            iou = intersection_over_union(pred_coords[..., bi, 1:], target[..., -(self.ncol_coords-1):])
            bious.append(iou)

        # get best box per cell (max iou) ==> best_bx_per_s
        pred_best_bbxi = torch.cat([t.unsqueeze(0).type(torch.int64) for t in bious])
        _, best_bx_per_s = torch.max(pred_best_bbxi, dim=0)

        ## where are actually the objects in target
        exists_box = target[..., self.C].unsqueeze(3) # => shape : batch, s, s, 1

        ## build a filter having 1 for all coords of best boxes, 0 for the rest
        best_box_filter = torch.zeros((pred_bboxes.shape[0], self.S, self.S, self.B, self.ncol_coords)).to(pred_best_bbxi.device)

        ## ugly gugly !! 
        ## TODO: find a torch way to do this
        for bai in range(best_bx_per_s.shape[0]):
            for si1 in range(best_bx_per_s.shape[1]):
                for si2 in range(best_bx_per_s.shape[2]):
                    bbi = best_bx_per_s[bai, si1, si2]
                    for ci in range(self.ncol_coords):
                        best_box_filter[bai, si1, si2, bbi, ci] = 1.0

        ## pred_bboxes_t1 mutiplied by filter will have coords values only on best boxes, rest will be 0
        ## final sum on boxes axis (-2) reduce to dim (batch, S, S, coords)
        box_predictions = torch.sum((best_box_filter * pred_coords), axis=-2) 

        # Only compute loss where there are actually objects in GT
        box_predictions *= exists_box
        box_targets = exists_box * target[..., -(self.ncol_coords-1):]

        # compute sqrt of h and w for predictions and target (keeping sign of the coords)
        box_predictions[..., -2:] = torch.sign(box_predictions[..., -2:]) * \
            torch.sqrt(torch.abs(box_predictions[..., -2:] + 1e-6))
        box_targets[..., -2:] =  torch.sqrt(box_targets[..., -2:] )

        # compute Mean square Error loss 
        box_loss = self.coord_loss_fn(
            torch.flatten(box_predictions[..., -(self.ncol_coords-1):], end_dim=-2), 
            torch.flatten( box_targets[..., -(self.ncol_coords-1):] , end_dim=-2)
        )
        print(f">box_loss : {box_loss}")

        ## ##################################################################################
        ##  Classes LOSS 
        ## ##################################################################################
        # mse on classes
        class_loss =  self.class_loss_fn(
             ## torch.flatten(, end_dim=-2) : flattent first cols batch, s, s
             torch.flatten(exists_box * pred_bboxes[..., :self.C], end_dim=-2),  
             torch.flatten(exists_box * target[..., :self.C], end_dim=-2)
        )

        print(f">class loss : {class_loss}")

        ## ##################################################################################
        ##  No Obj LOSS 
        ## ##################################################################################


        ## ##################################################################################
        ##  Obj LOSS 
        ## ##################################################################################


        ## class loss

        print(f"RC box_loss : {box_loss}")
        print(f"RC class loss : {class_loss}")

        
        loss = box_loss

        return loss