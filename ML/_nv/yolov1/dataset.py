
from torch.utils.data import Dataset
from torchvision import transforms, load

from glob import glob
import os 

# Image augmentation and preprocessing
#transform = transforms.Compose([transforms.ToTensor()])


#TrainingSet(nn))
#test_loader = DataLoader(dataset.ValidationSet())

def get_label_from_vignet_pict_name(fp):
    return os.path.basename(fp)[0]

class ClassifDatasetFromFolder(Dataset):
    def __init__(self, root_dir, labels_find_fn = get_label_from_vignet_pict_name, 
        classes = ["first_class", "second_class"], transform = None):
        
        super(ClassifDatasetFromFolder, self).__init__()
        self.img_fps = glob(os.path.join(root_dir, "**", "*") , recursive=True)
        self.labels_find_fn = labels_find_fn
        self.transform = transform

    def __len__(self):
        return len(self.img_fps)

    def __getitem__(self, ix):
        fp = self.img_fps[ix]
        img = torchvision.
        label_name = self.labels_find_fn(fp)


        return data, target
