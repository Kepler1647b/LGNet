import numpy as np
import matplotlib.pyplot as plt
import cv2
import utils
from vahadane import vahadane
from sklearn.manifold import TSNE
from glob import glob
import os
class1 = 'easy'
SOURCE_PATH = '/home/21/zihan/Storage/brain/tiles_10x_512_normal/easy'
TARGET_PATH = '/home/21/zihan/Storage/brain/server4/brain/code_brain/code/Vahadane/data/target_10x.jpeg'
RESULT_PATH = '/home/21/zihan/Storage/brain/Vahadane_10x_normal/%s' % class1



vhd = vahadane(LAMBDA1=0.01, LAMBDA2=0.01, fast_mode=1, getH_mode=0, ITER=50)
vhd.show_config()

vhd.fast_mode=0;vhd.getH_mode=0;

print(glob(TARGET_PATH))
target_image = utils.read_image(TARGET_PATH)
Wt, Ht = vhd.stain_separate(target_image)
i = 0
collectname = []
missname = []
addname = []
path = os.path.join(SOURCE_PATH)
for case_path in glob(os.path.join(path, '*')):
    i += 1
    print(i)
    case_name = os.path.basename(case_path)
    print(case_name)
    slide_names = os.listdir(case_path)
    #if not os.path.exists(os.path.join(RESULT_PATH, case_name)):        
        #os.makedirs(os.path.join(RESULT_PATH, case_name))
    #os.mkdir(os.path.join(RESULT_PATH, type, case_name, slide_name))
    for slidename in slide_names:
        #print(slidename)
        if not os.path.exists(os.path.join(case_path, slidename)):
            collectname.append(case_name)
            print(collectname)
            continue
        if not os.path.exists(os.path.join(RESULT_PATH, case_name, slidename)):        
            os.makedirs(os.path.join(RESULT_PATH, case_name, slidename))
        patch_name = os.listdir(os.path.join(case_path, slidename))
        for patch in patch_name:
            print(patch)
            if os.path.exists(os.path.join(RESULT_PATH, case_name, slidename, patch)):
                continue
            #if os.path.exists(os.path.join(RESULT_PATH, case_name, patch)):
                #continue
            #if case_name.split('_')[0][0:6] == '187697':
                #continue
            print(os.path.join(case_path, slidename))
            source_image = utils.read_image(os.path.join(case_path, slidename, patch))
            try:
                Ws, Hs = vhd.stain_separate(source_image)
                img = vhd.SPCN(source_image, Ws, Hs, Wt, Ht)
                if not os.path.exists(os.path.join(RESULT_PATH, case_name)):        
                    os.makedirs(os.path.join(RESULT_PATH, case_name))
                cv2.imwrite(os.path.join(RESULT_PATH, case_name, slidename, patch), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                print(patch)
                if case_name not in addname:
                    addname.append(case_name)
            except Exception as e:
                missname.append(case_name)
                missname.append(patch)
                continue
            print('empty:', collectname)
            print('add:', addname)
            print('miss', missname)
