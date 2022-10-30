
import torch
import numpy as np


def nms(boxes_list, threshoold):
    pass


def yolo_out_to_boxes_and_classes(yolo_out, conf_score=0.5, S=7, C=20, B=2, ncol_coords=5):
    """Takes the torch tensor resulting from yolo model prediction
        Returns [[x1, y1, h1, w1],[x2, y2, h2, w2],..., [xn, yn, hn, wn]]


    Args:
        yolo_out ([Tensor]): tensor (batch, S, S, C + B*nbcol_coords)
        S (int, optional): [description]. Defaults to 7.
        C (int, optional): [description]. Defaults to 20.
        B (int, optional): [description]. Defaults to 2.
    """

    yolo_out = yolo_out.reshape(-1, S , S, C + B * ncol_coords) 
    
    ## extract highest prob classes for each bbox
    # bbox_class ==> [batch, S, S, (classix, classprob)]
    classes_i = torch.tensor([i for i in range(C)])
    all_classes_prob = torch.index_select(yolo_out, -1, classes_i)
    bbxclass_prob, bbx_class_ix = torch.max(all_classes_prob, dim=-1)
    bbox_class = torch.cat([bbx_class_ix.unsqueeze(-1), bbxclass_prob.unsqueeze(-1)], -1)

    # find bboxes that have highest objectness prob per cell
    bboxes = []
    bbx_i = torch.tensor([C+ bi*ncol_coords for  bi in range(B)])
    boxes_objectness = torch.index_select(yolo_out, -1, bbx_i)
    bestb_objectness, bestb_ix = torch.max(boxes_objectness, dim=-1)
    ## [batch, s, s (starbestbx)]=> [batch, s, s, (startbestbx, ..., end best bxt)]
    bbbx_i = (ncol_coords * bestb_ix).unsqueeze(-1) + torch.tensor([ii for ii in range(ncol_coords)])
    bbox_coords = torch.index_select(yolo_out, -1, bbbx_i)
    #bbox_coords = bestb_ix.unsqueeze(-1) + torch.tensor([ii for ii in range(ncol_coords)])
    #bbox_coords = bestb_ix.unsqueeze(-1) + torch.tensor([ii for ii in range(ncol_coords)]) 
    # + torch.tensor([ii for ii in range(ncol_coords)])
    return torch.cat([bbox_class, bbox_coords], -1)
    # ## shift box
    # torch.index_select(yolo_out[]
    # # get coords of highest prob box ==> [batch, S, S, (best box coords)]

    # # iterate through samples in batch
    # for spli in range(yolo_out.shape[0]):
    #     #pred_sample = yolo_out[si]
    #     boxes, classes = [], []
    #     #start = C
    #     for sxi in range (S):
    #         for syi in range (S):
    #             # best box 
    #             prob_o, bimax = torch.max(yolo_out[spli, sxi, syi, [C+bi*nbcol_coords for  bi in range(B)]])
    #             bboxes.append[yolo_out[spli, sxi, syi, bimax*nbcol_coords:]
    #             for bi in range(B):
    #                 pos =  C + bi * nbcol_coords
    #                 if yolo_out[spli, sxi, syi, pos] >= conf_score: 
    #                     bbox = np.zeros( 3  + nbcol_coords) # objectness, class_id, class_prob, x, y, w, h
    #                     yolo_out[spli, sxi, syi,pos + 1 : pos + nbcol_coords]
    #                     boxes.append()
    #                     ci = torch.max(yolo_out[spli, sxi, syi, : C])[1]
    #                     classes.append(ci)
                        
        


