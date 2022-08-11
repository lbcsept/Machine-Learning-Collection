
import torch
import numpy as np


def nms(boxes_list, threshoold):
    pass


def yolo_out_to_boxes_and_classes(yolo_out, conf_score=0.5, S=7, C=20, B=2, nbcol_coords=5):
    """Takes the torch tensor resulting from yolo model prediction
        Returns [[x1, y1, h1, w1],[x2, y2, h2, w2],..., [xn, yn, hn, wn]]


    Args:
        yolo_out ([Tensor]): tensor (batch, S, S, C + B*nbcol_coords)
        S (int, optional): [description]. Defaults to 7.
        C (int, optional): [description]. Defaults to 20.
        B (int, optional): [description]. Defaults to 2.
    """

    if isinstance(yolo_out, torch.tensor):
        yolo_out = torch.detach('cpu').numpy()
    
    # iterate through samples in batch
    for spli in range(yolo_out.shape[0]):
        #pred_sample = yolo_out[si]
        boxes, classes = [], []
        #start = C
        for si1 in range (S):
            for si2 in range (S):
                for bi in range(B):
                    pos =  C + bi * nbcol_coords
                    if yolo_out[spli, si1, si2, pos] >= conf_score: 
                        bbox = np.zeros( 3  + nbcol_coords) # objectness, class_id, class_prob, x, y, w, h
                        yolo_out[spli, si1, si2,pos + 1 : pos + nbcol_coords]
                        boxes.append()
                        ci = torch.max(yolo_out[spli, si1, si2, : C])[1]
                        classes.append(ci)
                        
        


