
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import os
from loss_to_recode import YoloLossRC

from model import YoloV1
from dataset import YoloDataset
import config 
from utils import input_shape_from_image_shape
from loss import YoloLoss

import albumentations as A
from albumentations.pytorch import ToTensorV2

from tqdm import tqdm
import yaml 

# get working dir
wd = os.path.dirname(os.path.abspath(__file__))

## loading config file (exp_fp: experiment file path)
exp_fp = config.yolo_yml_file
if not os.path.isabs(exp_fp):
    exp_fp = os.path.join(wd, exp_fp)
    print(exp_fp)
with open(exp_fp) as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    dataset_confs = yaml.load(file, Loader=yaml.FullLoader)

config.__dict__.update(dataset_confs)
config.nclass = config.nc
print(config.__dict__)

# deterministic random
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(42)

# load device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Hyper parameters
batch_size = config.BATCH_SIZE
lr = config.LEARNING_RATE
num_epoch = config.EPOCHS

# # Loading model
config.mode = "object_detection" #classification"
model = YoloV1(**config.__dict__)
model = model.to(device)
model.print_params()

# # define loss criterion and optimizer
criterion = YoloLoss(**config.__dict__)
criterion_RC = YoloLossRC()
optim = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)


# # Train and Validation Dataloaders
if not os.path.isabs(config.train):
    config.train = os.path.join(wd, config.train)
if not os.path.isabs(config.val):
    config.val = os.path.join(wd, config.val)

# # data augmentation
import cv2
transform = A.Compose([
    A.LongestMaxSize(480, p=1),
    A.PadIfNeeded(min_height=480,min_width=480,border_mode=cv2.BORDER_CONSTANT, p=1),
    A.RandomCrop(448, 448, p=1),
    A.RandomBrightnessContrast(p=0.2),
    ToTensorV2()
], bbox_params=A.BboxParams(format='yolo', min_visibility=0.3, label_fields=[]))
#], bbox_params=A.BboxParams(format='yolo'))#, label_fields='class_labels'))

train_set = YoloDataset(pict_dir = config.train + "/images", label_dir = config.train + "/labels", transforms=transform,
                        nclass = config.nclass, nbox = config.nbox, s_grid = config.s_grid, label_classes=config.names)

train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True)
data, target = train_set[10]
print(data.shape)
print(target.shape)
print(target)

#train_set = YoloDataset(pict_dir = config.train + "/images", label_dir = config.train + "/labels", 
#                        nclass = config.nclass, nbox = config.nbox, s_grid = config.s_grid)
#train_loader = DataLoader(train_set, batch_size = config.BATCH_SIZE, shuffle=True)
#test_loader = DataLoader(valid_set, batch_size = config.BATCH_SIZE, shuffle=True)


## training one epoch
def train_one_epoch(model, criterion, optim, loader, device):
    model.train()
    losses = []
    prog_b = tqdm(loader)
    
    for bix, (data, target) in (enumerate(prog_b)):

        data, target = data.to(device), target.to(device)
        #data = data.float() # 
        # forward pass
        _, scores = model(data)
        
        # loss computation
        lossA = criterion_RC(scores, target)
        loss = criterion(scores, target)
        losses.append(loss.item())

        # flush out the gradients stored in the optimizer 
        optim.zero_grad()

        # backward pass
        loss.backward()

        # update the steps
        optim.step()



for epi, epoch in enumerate(range(num_epoch)):

    train_one_epoch(model, criterion, optim, train_loader, device)