
import torch

from model import YoloV3

import config
from tqdm import tqdm
# Hpyper params
batch_size = config.batch_size

# device
# TODO
device = config.device if config.device != 'auto' else torch.cuda 

# model
model = YoloV3()

# dataset

# one epoch training
 
def train_one_epoch(dataloader, model, criterion, optim, epi=-1, device='cpu'):
    
    model.train()
    pb = tqdm(dataloader)
    for bi, (X, target) in enumerate(pb):
        
        X, target = X.to(device), target.to(device)
        
        # do predict
        scores = model(X)
        
        # loss
        loss = criterion(scores, target)
        
        # flush out the gradients stored in the optimizer 
        optim.zero_grad()
        
        # backward
        loss.backward()
        
        # update the steps
        optim.step()