
import torch
from torch.utils.data import Dataset
from torchvision import transforms, load, io

from glob import glob
import os 

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

def YoloDataset(Dataset):
    
    def __init__(self, pict_dir, label_dir, 
                 nclass = 20, nbox =2, s_grid =7,
                 label_classes = None, 
                 transforms = None, annot_format="yolo", label_ext='.txt'):
        
        super(YoloDataset, self).__init__()
        
        self.pict_dir = pict_dir
        self.label_dir = label_dir
        self.annor_format = annot_format
        self.C = nclass
        self.B = nbox
        self.S = s_grid
        self.transforms = transforms
        picts_dict = { os.path.spliext(os.path.basename(fps))[0]:fps 
                       for fps in os.listdir(pict_dir) if os.path.splitext(fps)[-1].lower() 
                       not in [label_ext]}

        labels_dict = { os.path.spliext(os.path.basename(fps))[0]:fps 
                       for fps in os.listdir(label_dir) if os.path.splitext(fps)[-1].lower() 
                     in [label_ext]}
        self.fps = [(picts_dict[ky], labels_dict[ky]) for ky in picts_dict.keys() 
                    if ky in labels_dict.keys()]
        self.label_classes = label_classes
        
    def __len__(self):
        return len(self.fps)
    
    def __getitem__(self, ix):
        
        pict_fp, label_fp = self.fps[ix]
        
        img = io.read_image(pict_fp)
        np_annot = np.load_txt(label_fp)
        clis, box_coords = np_annot[0], np_annot[1:5]  
        class_labels = [self.label_classes.index(cli) for cli in clis]
            
        if self.transform:
            transformed = self.transforms(image=img, bboxes=box_coords.to_list(), class_labels=class_labels)
            img = transformed['image']
            bboxes = transformed['bboxes']
        
        
        # adjust bboxes to closest split origin    
        
        #bboxes = np.array(bboxes)
        target = torch.zeros(self.S * self.S, self.C + 5 * self.B)
        
        for bbox in bboxes:
            
            x0, y0, w, h = bbox
            xsplit, ysplit = int(x0/self.S), int((y0/self.S))
            x0cell, y0cell = x0 - xsplit, y0 - ysplit
        # create class table with 
        #class_tab = np.zeros[len(clis), self.C]
        # TODO :put 1 to corresponding class of clis
        ## class_tab[cli]
        
        
        return img, bboxes