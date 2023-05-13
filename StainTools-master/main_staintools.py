import staintools
import time
import PIL
import torch
from torchvision import transforms
import torchstain
import cv2
import matplotlib.pyplot as plt
from torchvision import utils
import numpy as np
import PIL
import time
import glob
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7' #use only GPU-0

sourcepath = '/home/21/zihan/Storage/brain/Vahadane_10x_zhuhai'
datapath = '/home/21/zihan/Storage/brain/bg_tiles_10x_512_multicenter/zhuhai/patchs'
outpath = '/home/21/zihan/Storage/brain/server4/brain/Vahadane_tiles_10x_staintools/vahadane'

METHOD = 'vahadane'
STANDARDIZE_BRIGHTNESS = True
TARGET_PATH = '/home/21/zihan/Storage/brain/server4/brain/code_brain/code/Vahadane/data/target_10x.jpeg'
targetimg = staintools.read_image(TARGET_PATH)
targetimg = staintools.LuminosityStandardizer.standardize(targetimg)
normalizer = staintools.StainNormalizer(method=METHOD)
normalizer.fit(targetimg)

missdic = {}

for type in os.listdir(sourcepath):
    #for casename in os.listdir(os.path.join(sourcepath, type)):
    for slidename in os.listdir(os.path.join(sourcepath, type)):
        print('start:', slidename)
        #if slidename in ['203124_', '203173_']:
            #continue
        if not os.path.exists(os.path.join(datapath, type, slidename)):
            missdic.append(slidename)
            continue
        for patchname in os.listdir(os.path.join(sourcepath, type, slidename)):
            print(patchname)
            patch = os.path.join(datapath, type, slidename, 'tissue', patchname)
            #patch = os.path.join(sourcepath, type, casename, slidename, patchname)
            if os.path.exists(os.path.join(outpath, type, slidename, patchname)):
                continue
            time_start=time.time()
            try:
                img = staintools.read_image(patch)
                img = staintools.LuminosityStandardizer.standardize(img)
                img_normalized = normalizer.transform(img)
                img_normalized = PIL.Image.fromarray(img_normalized)
                if not os.path.exists(os.path.join(outpath, type, slidename)):
                    os.makedirs(os.path.join(outpath, type, slidename))
                patchname = os.path.basename(patch)
                img_normalized.save(os.path.join(outpath, type, slidename, patchname))
            except Exception as e:
                print('error:', slidename, patchname)
                if slidename not in missdic:
                    missdic[slidename] = []
                missdic[slidename].append(patchname)

            time_end=time.time()
            print('time cost',time_end-time_start,'s')
        print('finish:', slidename)
        print('missed slide:', missdic)