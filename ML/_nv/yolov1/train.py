
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np


from model import YoloV1
import config 
from utils import input_shape_from_image_shape

import tqdm
import yaml 


## loading config file
with open(config.yolo_yml_file) as file:
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
config.mode = "classification"
model = YoloV1(**config.__dict__)
model= model.to(device)
model.print_params()



# # Train and Validation Dataloaders
train_loader = DataLoader(dataset.TrainingSet())
test_loader = DataLoader(dataset.ValidationSet())


#train_loader 


## training one epoch
def train_one_epoch(model, criterion, optim, loader, device):
    losses = []
    prog_b = tqdm(loader)
    for bix, (data, target) in (enumerate(prog_b)):

        data, target = data.to(device), target.to(device)
        
        # forward pass
        _, scores = model(data)
        
        # loss computation
        loss = criterion(scores, target)
        losses.append(loss.item())

        # flush out the gradients stored in the optimizer 
        optim.zero_grad()

        # backward pass
        loss.backward()

        # update the steps
        optim.step()