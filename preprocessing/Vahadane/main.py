import cv2
import utils
from vahadane import vahadane
from glob import glob
import os
SOURCE_PATH = './source'
TARGET_PATH = './target.jpeg'
RESULT_PATH = './result'

missname = []

vhd = vahadane(LAMBDA1=0.01, LAMBDA2=0.01, fast_mode=1, getH_mode=0, ITER=50)
vhd.show_config()

vhd.fast_mode=0;vhd.getH_mode=0;

print(glob(TARGET_PATH))
target_image = utils.read_image(TARGET_PATH)
Wt, Ht = vhd.stain_separate(target_image)
i = 0
for case in glob.glob(os.path.join(SOURCE_PATH, '*')):
    case_name = os.path.basename(case)
    for patch in glob.glob(os.path.join(case, '*')):
        source_image = utils.read_image(patch)
        try:
            Ws, Hs = vhd.stain_separate(source_image)
            img = vhd.SPCN(source_image, Ws, Hs, Wt, Ht)
            if not os.path.exists(os.path.join(RESULT_PATH, case_name)):        
                os.makedirs(os.path.join(RESULT_PATH, case_name))
            cv2.imwrite(os.path.join(RESULT_PATH, case_name, patch), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print(patch)
        except Exception as e:
            missname.append(case_name)
            missname.append(patch)
            continue
        print('miss', missname)
