import torch
from torch.utils.data import DataLoader
from model import YoloV1

from utils import model_utils

from dataset import YoloDataset
import config
import cv2

import logging
import yaml
import os

import albumentations as A
from albumentations.pytorch import ToTensorV2

from to_recode import utils_to_recode



logging.basicConfig(level=config.logging_level)
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

# load device 
if hasattr(config, "DEVICE") and config.DEVICE is not None:
    device = config.DEVICE
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


## checkpoint dir update
if not hasattr(config, "CHECKPOINTS_DIR"):
    config.CHECKPOINTS_DIR = ""
if not os.path.isabs(config.CHECKPOINTS_DIR):
    config.CHECKPOINTS_DIR = os.path.join(wd, config.CHECKPOINTS_DIR)

inference_transform = A.Compose([
    A.LongestMaxSize(config.image_resize, p=1),
    A.PadIfNeeded(min_height=config.image_resize, min_width=config.image_resize, border_mode=cv2.BORDER_CONSTANT, p=1),
    A.RandomCrop(config.image_shape[0], config.image_shape[1], p=1),
    A.RandomBrightnessContrast(p=0.2),
    ToTensorV2()
], bbox_params=A.BboxParams(format='yolo', min_visibility=0.3, label_fields=[]))

valid_set = YoloDataset(pict_dir = config.val + "/images", label_dir = config.val + "/labels", transforms=inference_transform,
                        nclass = config.nclass, nbox = config.nbox, s_grid = config.s_grid, label_classes=config.names, 
                        boundaries=config.boundaries, sampling=config.sampling)

valid_loader = DataLoader(valid_set, batch_size=config.BATCH_SIZE, shuffle=False)
sample_ix = 0
data, target = valid_set[sample_ix]
target = target.to(device)

bboxes = utils_to_recode.cellboxes_to_boxes(target, S=config.s_grid)
bboxes = bboxes.numpy()
#target = torch.flatten(target[sample_ix], end_dim=-2).squeeze(0)
#bboxes = target[:, -4:].numpy()
_ , label_ix = torch.max(target[:, :-4], dim=0)
labels = [config.names[lbix] for lbix in label_ix]

images = data.to(device).swapaxes(0,2)
image = images[0].numpy()

#data = data.cpu().detach().numpy()

from pred_image.labelling import yolo as yolo_utils
from pred_image.img import bbx_n_cnts
cnts = bbx_n_cnts.bbox_xyxy2cnts(yolo_utils.bbox_yolo2xyxy(torch.flattentarget[..., -4:], image.shape))

#ppu.display()

utils_to_recode.plot_image(data[0])


#utils_to_recode.

model = YoloV1(**config.__dict__)

model_utils.load_checkpoint(config.CHECKPOINTS_DIR, model, None, name_prefix=config.EXP_NAME, name_suffix='model_only', epoch='best')

#utils_to_recode.