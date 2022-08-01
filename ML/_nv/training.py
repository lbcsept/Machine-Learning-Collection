

import numpy as np

import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import optim
from tqdm import tqdm
from utils import check_accuracy, load_checkpoint, save_checkpoint
import os
from cnn import BasicCNN

from torch.utils.tensorboard import SummaryWriter


# deterministic random
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

# torch device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

# hyper params 
batch_size = 256   #24 
num_epoch = 4
learning_rate =0.001
model_name = "basic_cnn"
checkpoint_dir = os.path.abspath("/home/nikoenki/Documents/models") #("~/Documents/models".replace('~/',''))
checkpoint_fp = os.path.join(checkpoint_dir, model_name)
load_last_checkpoint = False

# dataset and pre processing
transform = transforms.Compose([transforms.ToTensor()])
train = datasets.MNIST("~/Documents/datasets", train=True, transform=transform,download=True)
test = datasets.MNIST("~/Documents/datasets", train=False, transform=transform,download=True)

train_dataloader = DataLoader(train, batch_size=batch_size,shuffle=True)
test_dataloader = DataLoader(test, batch_size=batch_size, shuffle=True)

# model 
model = BasicCNN().to(device) # put model into device

#print(model)
# optimiser and loss
opt = optim.Adam(model.parameters(),lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# loading last model checkpoint
if load_last_checkpoint:
    chkptfps = os.listdir(checkpoint_dir)
    chkpts = [fp for fp in chkptfps if os.path.basename(checkpoint_fp) in fp]
    if len(chkpts)>0:
        load_checkpoint(os.path.join(checkpoint_dir, sorted(chkptfps)[-1]), model, opt)

# tensorboard
tbwriter = SummaryWriter(f"{checkpoint_dir}/runs/MNIST/{model_name}")
#tbwriter.add_graph(model=model)

step = 0


for ep_ix, epoch in enumerate(range(num_epoch)):
    
    losses, accuracies = [], []
    num_correct, num_samples = 0, 0
    model.train() #set model in train state 
    pbar = tqdm(train_dataloader)


    for b_ix, (data, target) in enumerate(pbar):
        
        #if b_ix > 0:
        #    break
        #put data and target into device
        data, target = data.to(device), target.to(device)

        # foward pass
        scores = model(data)
        #output = torch.argmax(output, dim=1) # no need to argmax, softmax crossentropy needs raw scores
        #print((scores, target))
        
        # loss computation
        loss = criterion(scores, target)
        losses.append(loss.item())

        # flush out the gradients stored in optimizer
        opt.zero_grad()

        # backward pass
        loss.backward()
        
        # calculate running training accuracy
        _, predictions = scores.max(1)
        num_correct += (predictions == target).sum()
        num_samples += predictions.size(0)

        pbar.set_description(f"training, batch {b_ix}, epoch {ep_ix}... "\
        f"{num_correct}/{num_samples} correct samples, acc:{100.0*num_correct/num_samples:.2f}")
        # update the steps
        opt.step()

        # update tensorboard
        tbwriter.add_scalar("training loss ", loss, global_step=step)
        tbwriter.add_scalar("training accuracy ", float(num_correct)/float(num_samples), global_step=step)
        lyp = []
        for name, params in model.named_parameters():
            lyp.append((name, params))
            tbwriter.add_histogram(f"histo_{name}", values=params, global_step=step)

        step += 1


    # checkpoint
    checkpoint = {'state_dict': model.state_dict(), 'optimizer': opt.state_dict()}
    save_checkpoint(checkpoint, filepath=checkpoint_fp + f"_epoch{ep_ix:03d}.pth.tar", verbose=False)

    acc = check_accuracy(test_dataloader, model, message="TEST SET")
    #tbwriter.add_scalar("test loss ", loss, global_step=step)
    tbwriter.add_scalar("test accuracy ", acc, global_step=step)


    #print(f"Accuracy on test set: {acc*100:.2f}")


print(f"Accuracy on training set: {check_accuracy(train_dataloader, model, message='TRAINING SET')*100:.2f}")

#check_accuraccy(test_dataloader, model)


   