import torch.utils.data as data_utils
import torch
from PIL import Image 
import cv2
#import jpeg4py as jpeg
import numpy as np
import random
import glob
import os
import math
import logging

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
dic2 = {'normal': 0,
'tumor': 1}


class ZSData(data_utils.Dataset):
    def __init__(self, slidepath, type, transforms=None, bi = None):
        #self.datapath = imgs_dir
        #self.maskpath = masks_dir
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
                self.label.append(dic2[type])


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



