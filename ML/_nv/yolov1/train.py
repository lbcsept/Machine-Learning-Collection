import logging
from re import S
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import os
from .utils.yolo_utils import yolo_out_to_boxes_and_classes
from to_recode.loss_to_recode import YoloLossRC
from utils import model_utils

from model import YoloV1
from dataset import YoloDataset
import config 
from utils.misc import input_shape_from_image_shape
from loss import YoloLoss

import albumentations as A
from albumentations.pytorch import ToTensorV2

from tqdm import tqdm
import yaml 

logging.basicConfig(level=config.logging_level)

# get working dir
wd = os.path.dirname(os.path.abspath(__file__))

## checkpoint dir update
if not hasattr(config, "CHECKPOINTS_DIR"):
    config.CHECKPOINTS_DIR = ""
if not os.path.isabs(config.CHECKPOINTS_DIR):
    config.CHECKPOINTS_DIR = os.path.join(wd, config.CHECKPOINTS_DIR)


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
if hasattr(config, "DEVICE") and config.DEVICE is not None:
    device = config.DEVICE
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Hyper parameters
batch_size = config.BATCH_SIZE
lr = config.LEARNING_RATE
num_epoch = config.EPOCHS

# # Loading model
# config.mode = "object_detection" #classification"
# config.nohead = True
# model_nh = YoloV1(**config.__dict__)
# torch.save(model_nh, os.path.join(wd, "model_no_head.pth"))
# #model = model.to(device)
# #model.print_params()
# config.nohead = False

config.mode = "object_detection" 
model = YoloV1(**config.__dict__)
model = model.to(device)
model.print_params()


# # define loss criterion and optimizer
criterion = YoloLoss(**config.__dict__)
criterion_RC = YoloLossRC()
optim = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)


# load previous checkpoint if needed
if hasattr(config, "LOAD_CHECKPOINT") and config.LOAD_CHECKPOINT is not None:
    if isinstance(config.LOAD_CHECKPOINT, bool) and not config.LOAD_CHECKPOINT:
        pass
    else:
        model_utils.load_checkpoint(config.CHECKPOINTS_DIR, model, optim=optim, name_prefix= None if config.EXP_NAME is None else config.EXP_NAME, epoch=config.LOAD_CHECKPOINT)


# # Train and Validation Dataloaders
if not os.path.isabs(config.train):
    config.train = os.path.join(wd, config.train)
if not os.path.isabs(config.val):
    config.val = os.path.join(wd, config.val)

# # data augmentation
import cv2



transform = A.Compose([
    A.LongestMaxSize(config.image_resize, p=1),
    A.PadIfNeeded(min_height=config.image_resize, min_width=config.image_resize, border_mode=cv2.BORDER_CONSTANT, p=1),
    A.RandomCrop(config.image_shape[0], config.image_shape[1], p=1),
    A.RandomBrightnessContrast(p=0.2),
    ToTensorV2()
], bbox_params=A.BboxParams(format='yolo', min_visibility=0.3, label_fields=[]))


train_set = YoloDataset(pict_dir = config.train + "/images", label_dir = config.train + "/labels", transforms=transform,
                        nclass = config.nclass, nbox = config.nbox, s_grid = config.s_grid, label_classes=config.names, 
                        boundaries=config.boundaries, sampling=config.sampling)

train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True)
data, target = train_set[10]

## training one epoch
def train_one_epoch(model, criterion, optim, loader, device, epoch=None, exp_name=None, txt=None):
    model.train()
    losses, all_losses = [], {}
    prog_b = tqdm(loader)
    desc = f"{exp_name}" if exp_name is not None else "" 
    desc += f" - epoch {epoch}" if epoch is not None else "" 
    desc += f" - {txt}" if txt is not None else ""
    prog_b.set_description(desc)
    for bix, (data, target) in (enumerate(prog_b)):

        data, target = data.to(device), target.to(device)
        #data = data.float() # 
        # forward pass
        _, scores = model(data)
        
        # loss computation
        loss, loss_dict = criterion(scores, target)

        # // comparison with orig loss implementation
        if config.compare_with_orig_impl:
            lossRC, lossRC_dict = criterion_RC(scores, target)
            ko, ko_txt = False, []
            for k, v in lossRC_dict.items():
                if v - loss_dict[k] != 0.0:
                    ko_txt.append(f"{k}:{v} |{loss_dict[k]}")
                    jo=True
                else:
                    ko_txt.append(f"{k}:OK")
            ko_txt.append("\n")
            if ko:
                print("; ".join(ko_txt))
            # // end 
        
        losses.append(loss.item())
        for k, v in loss_dict.items():
            if k not in all_losses.keys():
                all_losses[k] = []
            all_losses[k].append(v)

        # flush out the gradients stored in the optimizer 
        optim.zero_grad()

        # backward pass
        loss.backward()

        # update the steps
        optim.step()
        mean_loss = sum(losses)/len(losses)    
        #prog_b.set_postfix(loss = f"{loss:.2f}")
        prog_b.set_postfix(train_mean_loss = f"{mean_loss:.2f}")
        
        if bix >= len(loader):
            prog_b.set_postfix(epoch_loss = f"{mean_loss:.2f}")

    all_losses = {k:sum(v)/len(v) for k, v in all_losses.items()}

    return all_losses


best_loss = 1000000000000
epoch_start = 35
for epi, epoch in enumerate(range(num_epoch)):


    epoch_num = epoch_start + epi

    all_losses = train_one_epoch(model, criterion, optim, train_loader, device, epoch=epoch_num, exp_name=config.EXP_NAME)
    all_losses_print = {k:f"{v:0.2f}" for k, v in all_losses.items()}
    print(f"epoch {epoch_num} : {all_losses_print}")
    loss = all_losses["loss"]
    best_loss = min(loss, best_loss)
    if config.SAVE_CHECKPOINTS:
        model_utils.save_checkpoint(model, opt=optim, epoch=epoch_num, dir_path=config.CHECKPOINTS_DIR, 
            name_prefix=config.EXP_NAME, name_suffix=model_utils.model_metrics_pprint(all_losses_print))
    
    if loss == best_loss:
        model_utils.save_checkpoint(model, opt=optim, epoch="best", dir_path=config.CHECKPOINTS_DIR, 
            name_prefix=config.EXP_NAME, name_suffix=model_utils.model_metrics_pprint(all_losses_print))

model_utils.save_checkpoint(model, opt=optim, epoch="last", dir_path=config.CHECKPOINTS_DIR, 
    name_prefix=config.EXP_NAME, name_suffix=model_utils.model_metrics_pprint(all_losses_print))



