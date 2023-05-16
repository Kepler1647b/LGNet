import staintools
import time
import PIL
import PIL
import time
import glob
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' #use only GPU-0

datapath = './datapath'
outpath = './outpath'

METHOD = 'vahadane'
STANDARDIZE_BRIGHTNESS = True
TARGET_PATH = './target.jpeg'# path of target pic
targetimg = staintools.read_image(TARGET_PATH)
targetimg = staintools.LuminosityStandardizer.standardize(targetimg)
normalizer = staintools.StainNormalizer(method=METHOD)
normalizer.fit(targetimg)

missdic = {}

for slide in glob.glob(os.path.join(datapath, '*')):
    slidename = os.path.basename(slide)
    for patch in glob.glob(os.path.join(slide, '*')):
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