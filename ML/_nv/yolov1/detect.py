
import torch
from torch.utils.data import DataLoader
from ML._nv.yolov1.model import YoloV1

from utils import model_utils

from dataset import YoloDataset
import config

import logging
import yaml
import os

import albumentations as A
from albumentations.pytorch import ToTensorV2

from to_recode import utils_to_recode

logging.basicConfig(level=config.logging_level)
# get working dir
wd = os.path.dirname(os.path.abspath(__file__))

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

valid_set = YoloDataset(pict_dir = config.train + "/images", label_dir = config.train + "/labels", transforms=transform,
                        nclass = config.nclass, nbox = config.nbox, s_grid = config.s_grid, label_classes=config.names, 
                        boundaries=config.boundaries, sampling=config.sampling)

valid_loader = DataLoader(valid_set, batch_size=config.BATCH_SIZE, shuffle=False)
data, target = valid_set[10]

#utils_to_recode.


#utils_to_recode.

model = YoloV1(**config.__dict__)

model_utils.load_checkpoint(config.CHECKPOINTS_DIR, model, None, name_prefix=config.EXP_NAME, name_suffix='model_only', epoch='best')

#utils_to_recode.