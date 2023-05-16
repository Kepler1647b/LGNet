import torch.utils.data as data_utils
import torch
from PIL import Image
import numpy as np
import random
import glob
import os
def seed_torch(seed = 0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_torch(seed = 0)

dic = {'glioma': 0,
       'lymphoma': 1}

class ZSData(data_utils.Dataset):
    def __init__(self, slidepath, type, transforms=None, bi = None):
        self.mask = []
        self.img = []
        self.label = []
        patchs = glob.glob(os.path.join(slidepath, '*'))
        patchs.sort()
        for patch in patchs:
            self.img.append(patch)
            self.transforms = transforms
            if type in dic.keys():
                self.label.append(dic[type])
            else:
                self.label.append(0)


    def __len__(self):
        return len(self.label)
        
    def __getitem__(self, item):
        img = Image.open(self.img[item])
        img_label = self.label[item]

        if self.transforms is not None:
            img = self.transforms(img)
        return img, img_label

    def get_label_list(self):
        return self.label



