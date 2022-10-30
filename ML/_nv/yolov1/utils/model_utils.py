import os
from glob import glob
import torch


def input_shape_from_image_shape(image_shape=(448, 448, 3), batch_size=2):
    """Return shape (batch_size, c, h, w) from image shape (h, w, c) """
    image_shape = list(image_shape)
    input_shape = [batch_size] + [image_shape[-1]] + image_shape[:-1]
    return tuple(input_shape)


def save_checkpoint(model, opt, epoch, dir_path="./", file_name=None, name_prefix=None, name_suffix=None, ext=".pth", save_state_dict=False):

    
    checkpoint = {'state_dict': model.state_dict(), 'optimizer': opt.state_dict()}    
    if file_name is None:
        name_prefix = "checkpoint" if name_prefix is None else name_prefix
        name_suffix = f"{name_suffix}" if not name_suffix is None else ""
        epoch_txt = f"epoch{epoch:03d}" if not isinstance(epoch , str) else epoch

        if epoch == "last":
            ## clean last lasts
            lasts = glob(os.path.join(dir_path, f"{name_prefix}_last_*{ext}"))
            #print(f"removing last files {lasts}")
            for fp in lasts:
                os.remove(fp)

        if epoch == "best":
            ## clean last best
            lasts = glob(os.path.join(dir_path, f"{name_prefix}_best_*{ext}"))
            #print(f"removing last best files {lasts}")
            for fp in lasts:
                os.remove(fp)

        file_name = f"{name_prefix}_{epoch_txt}_{name_suffix}{ext}"

    
    os.makedirs(dir_path, exist_ok=True)
    fp = os.path.join(dir_path, file_name)
    print(f"saving file {fp}")
    torch.save(checkpoint , fp)

    if epoch == "best":
        torch.save(model, fp.replace(ext, f"_model_only{ext}"))


def load_checkpoint(dir_path, model, optim=None, file_name=None, name_prefix=None, name_suffix=None, epoch=None):

    if isinstance(epoch, bool):
        epoch=None
    if file_name is None:
        name_prefix = "checkpoint" if name_prefix is None else name_prefix
        name_suffix = f"{name_suffix}" if not name_suffix is None else ""
        epoch_txt = "*" if epoch is None else f"epoch{epoch:03d}" if not isinstance(epoch , str) else epoch
        file_name = f"{name_prefix}_{epoch_txt}_{name_suffix}*"

    fps = glob(os.path.join(dir_path, file_name))
    import re
    fps = [(os.path.basename(fp), fp, int(re.findall(r"epoch\d+", os.path.basename(fp))[0].replace('epoch','')) 
        if "epoch" in os.path.basename(fp) else 10000 if os.path.basename(fp)[0:len(name_prefix)+5][-4:] == "last" else 0) 
            for fp in fps if not "model_only" in fp]
    if len(fps) > 1 and epoch is None:
        
        # find last epoch
        fps.sort(key= lambda x: x[2], reverse=True)
    if len(fps) == 0:
        print(f"No checkpoint found in dir {dir_path} with search pattern {file_name}, model and optimizer not updated on state_dict") 
        return 

    fp = fps[0][1]
    print(f"Model and optimizer updated are on stat_dict from file {fp}")
    chkpt = torch.load(fp)
    model.load_state_dict(chkpt['state_dict'])
    if optim is not None and 'optimizer' in chkpt.keys():
        optim.load_state_dict(chkpt['optimizer'])


def model_metrics_pprint(metrics_dict, show=["loss", "box_loss", "class_loss"], items_sep="_"):
    txt = []
    if show is None or len(show)==0:
        show = metrics_dict.keys()
    for k in show:
        txt.append(f"{k}_{metrics_dict[k]}")
    

    return items_sep.join(txt)