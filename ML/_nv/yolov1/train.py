
import torch
from torch import nn
import numpy as np


from model import YoloV1
import config 
from utils import input_shape_from_image_shape

import tqdm


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
config.mode = "classification"
model = YoloV1(**config.__dict__)
model= model.to(device)


model.eval()
x = torch.randn(input_shape_from_image_shape(config.image_shape)).to(device)
print(f"input shape : {x.shape}")
backbone, pred = model(x)
print(f"output shape : {pred.shape}")

# # Train and Validation Dataloaders
dataset = None
#train_loader 


## training one epoch
def train_one_epoch(model, loader):
    pass