import logging
import torch
from torch.utils.data import Dataset
from torchvision import transforms, load, io

from glob import glob
import os 

import cv2
import numpy as np
# Image augmentation and preprocessing
#transform = transforms.Compose([transforms.ToTensor()])


#TrainingSet(nn))
#test_loader = DataLoader(dataset.ValidationSet())

def get_label_from_vignet_pict_name(fp):
    return os.path.basename(fp)[0]

class ClassifDatasetFromFolder(Dataset):
    def __init__(self, root_dir, labels_find_fn = get_label_from_vignet_pict_name, 
        classes = ["first_class", "second_class"], transform = None):
        
        super(ClassifDatasetFromFolder, self).__init__()
        self.img_fps = glob(os.path.join(root_dir, "**", "*") , recursive=True)
        self.labels_find_fn = labels_find_fn
        self.transform = transform

    def __len__(self):
        return len(self.img_fps)

    def __getitem__(self, ix):
        fp = self.img_fps[ix]
        img = io.read_image
        label_name = self.labels_find_fn(fp)


        return img, label_name


def read_yolo_annot_file(fps):
    
    #with open(fps) as f:
    # cl, x0, y0, w, h
    return np.load_txt(fps)   

class YoloDataset(Dataset):
    
    def __init__(self, pict_dir, label_dir, nclass = 20, nbox =2, s_grid =7, 
        label_classes = None, transforms = None, label_ext='.txt', boundaries = None,
        sampling=None):

        super(YoloDataset, self).__init__()
        
        self.pict_dir = pict_dir
        self.label_dir = label_dir
        self.C = nclass
        self.B = nbox
        self.S = s_grid
        self.transforms = transforms
        self.boundaries = boundaries
        self.sampling = sampling

        picts_dict = { os.path.splitext(os.path.basename(fps))[0]:os.path.join(pict_dir,fps)
                       for fps in os.listdir(pict_dir) if os.path.splitext(fps)[-1].lower() 
                       not in [label_ext]}

        labels_dict = { os.path.splitext(os.path.basename(fps))[0]:os.path.join(label_dir,fps)
                       for fps in os.listdir(label_dir) if os.path.splitext(fps)[-1].lower() 
                     in [label_ext]}
        self.fps = [(picts_dict[ky], labels_dict[ky]) for ky in picts_dict.keys() 
                    if ky in labels_dict.keys()]
        
        if self.boundaries is not None:
            start, end = self.boundaries[0], self.boundaries[1]
            logging.info(f"Dataset is filtered from indexes {start} to {end} in total {len(self.fps)}")
            self.fps = [(k, v) for ix, (k, v) in enumerate(self.fps) if start <= ix <= end ]

        if self.sampling is not None:
            logging.info(f"Dataset composed of {self.sampling} random samples from total {len(self.fps)}")
            ixs = np.random.choice(len(self.fps), self.sampling, replace=False)
            self.fps = [(k, v) for ix, (k, v) in enumerate(self.fps) if ix in ixs ]


        self.label_classes = label_classes
        
    def __len__(self):
        return len(self.fps)
    
    def __getitem__(self, ix):
        
        pict_fp, label_fp = self.fps[ix]
        
        img = cv2.cvtColor(cv2.imread(pict_fp), cv2.COLOR_BGR2RGB) #io.read_image(pict_fp)
        np_annot = np.loadtxt(label_fp, ndmin=2)
        clis, box_coords = np_annot[:, 0], np.clip(np_annot[:, 1:5], 0.0, 1.0) #np.clip(np_annot[:, 1:5], 0.0, 1.0)
        bboxes =  box_coords.tolist()
        clis = clis.astype(int).tolist()
        class_labels = [self.label_classes[int(cli)] for cli in clis]
        
        

        if self.transforms:
            #transformed = self.transforms(image=img, bboxes=bboxes, class_labels=class_labels)
            try :
                transformed = self.transforms(image=img, bboxes=bboxes)
            except:
                correct_bbox = []
                for ii in range(len(bboxes)):
                    try:
                        transformed = self.transforms(image=img, bboxes=[bboxes[ii]])
                        correct_bbox.append(bboxes[ii])

                    except Exception as e:
                        logging.debug((f"Skipping box {ii} on image {pict_fp}:{bboxes[ii]}, error message ::\n{e}"))
                        #raise(Exception(f"error on image {pict_fp}, bbox {ii}:{bboxes[ii]}, error message ::\n{e}"))
                transformed = self.transforms(image=img, bboxes=correct_bbox)

            img = transformed['image']
            bboxes = transformed['bboxes']
        
        #bboxes = np.array(bboxes)
        target = torch.zeros(self.S, self.S, self.C + 5) # * self.B)
        
        for bbxi, bbox in enumerate(bboxes):
            
            # adjust bboxe center relatively to origin of closest split
            x0, y0, w, h = bbox
            xsplit  = min(x0/(1/self.S), self.S-1)
            ysplit = min((y0/(1/self.S)), self.S-1)
            x0cell, y0cell = xsplit - int(xsplit), ysplit - int(ysplit)
            xsplit, ysplit = int(xsplit), int(ysplit)
            # if corresponding cell does not already have a bbox, then put current one
            if target[xsplit, ysplit, -5] == 0:
                target[xsplit, ysplit, -5] = 1
                target[xsplit, ysplit, clis[bbxi]] = 1
                target[xsplit, ysplit, -4] = x0cell
                target[xsplit, ysplit, -3] = y0cell
                target[xsplit, ysplit, -2] = w
                target[xsplit, ysplit, -1] = h
                 
                    
        
        return img.float(), target
