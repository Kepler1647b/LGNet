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

dic = {'lymphoma': 1,
       'glioma': 0}

class ZSData(data_utils.Dataset):
    def __init__(self, imgs_dir, imgs_dir2, transforms=None, bi = None):
        self.datapath = imgs_dir
        self.datapath2 = imgs_dir2
        #self.maskpath = masks_dir
        self.mask = []
        self.img = []
        self.label = []
        for datapath in [imgs_dir, imgs_dir2]:

        #logging.info(f'Creating dataset with {len(self.ids)} examples')
            for type in ['lymphoma', 'glioma']:
                #if not os.path.exists(os.path.join(self.maskpath, type)):
                    #print('No exist type of data in mask dataset:', type)
                    #continue
                if not os.path.exists(os.path.join(datapath, type)):
                    print('No exist type of data in image dataset:', type)
                    continue
                else:
                    cases = os.listdir(os.path.join(datapath, type))
                for case in cases:
                    slides = os.listdir(os.path.join(datapath, type, case))
                    for slide in slides:
                        slidepatchs = []
                        slidelabels = []
                        patchs = glob.glob(os.path.join(datapath, type, case, slide, '*'))
                        random.shuffle(patchs)
                        for patch in patchs:        
                            patchname = os.path.basename(patch)
                            slidepatchs.append(patch)
                            slidelabels.append(dic[type])
                            slidepatchs.append(patch)
                            slidelabels.append(dic[type])                
                        self.img += slidepatchs
                        self.label += slidelabels

        self.transforms = transforms

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



