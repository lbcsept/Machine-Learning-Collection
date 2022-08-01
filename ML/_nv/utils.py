
from tabnanny import verbose
import torch
from tqdm import tqdm 



def save_checkpoint(states_dicts = {'state_dict':{}, 'optimizer':{}}, filepath='model_checkpoint.pth.tar', verbose=False):
    print(f"saving checkpoint in {filepath}")
    if verbose:
        print(states_dicts)
    torch.save(states_dicts, filepath)

def load_checkpoint(filepath, model, optimizer):
    print(f"loading checkpoint from {filepath}")
    chkpt = torch.load(filepath)
    model.load_state_dict(chkpt['state_dict'])
    optimizer.load_state_dict(chkpt['optimizer'])

def check_accuracy(loader, model, message=""):
    is_train = model.training
    
    device = next(model.parameters()).device
    #print(f"device {device}")

    num_correct, num_samples = 0, 0 #len(loader) 
    model.eval() # model in eval state
    pbar = tqdm(loader)
    for bix, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)
        
        scores = model(x)
        ###print(scores)
        ### IS SOFTMAXX NEEDED ???
        preds = torch.softmax(scores, dim=-1)
        ###print(preds)
        _, predictions = preds.max(1)
        ###print(f"pred {predictions}, argmax: {torch.argmax(preds, dim=-1)}")
        num_correct += (predictions == y).sum()
        num_samples +=  predictions.size(0)
        pbar.set_description(f"{message}, {num_correct}/{num_samples} correct samples, batch acc:{100.0*num_correct/num_samples:.2f}")

    if is_train:
        model.train()
    #pbar.set_description(f"{num_correct}/{num_samples} correct samples, Dataset Accuraccy:{100.0*num_correct/num_samples:.1f}")
    return float(num_correct)/float(num_samples)


