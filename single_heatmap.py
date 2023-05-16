# -*- coding: utf-8 -*
import os
from argparse import ArgumentParser
from glob import glob
from PIL import Image
from collections import OrderedDict
from Dataloader_LGNet.dataloader_instant import ZSData
from prefetch_generator import BackgroundGenerator
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
import time

import Utils.config_brain as CONFIG
plt.switch_backend('agg')
from Model.create_model import create_model
from Utils.utils import get_index
import Utils.utils.transform_test as data_transforms

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--ModelPath", dest='model_path', type=str)
    parser.add_argument("--DataPath", dest='data_path', type=str)
    parser.add_argument("--ResultPath", dest='resultpath', type=str)
    parser.add_argument("--DeviceId", dest='device_id', type=str)
    parser.add_argument("--Model", dest='model', type=str)
    parser.add_argument("--Foldn", dest='foldn', type=str)
    parser.add_argument("--Threshold", dest='threshold', type=str)
    args = parser.parse_args()

    time_start=time.time()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_id
    if args.device_id is not None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    modelpath = args.model_path
    modeldic = glob(os.path.join(args.model_path, '*'))
    modeldic.sort()
    print(modeldic)

    test_path = args.data_path

    slidename = ['013989_S1711703.svs']#the file name of target slide
    bs = 64

    slidename = slidename.split('.')[0]
    if not os.path.exists(os.path.join(args.resultpath, slidename)):
        os.makedirs(os.path.join(args.resultpath, slidename))
    patchs_aver = np.zeros([2, 1])
    patchs_count = np.zeros([2, 1])
    print(slidename)
    patchs = glob(os.path.join(test_path, slidename, '*.jpeg'))
    patchs.reverse()
    print(patchs)
    test_dataset = ZSData(os.path.join(test_path, slidename), 'else', transforms = data_transforms, bi = None)
    test_loader = DataLoaderX(test_dataset, batch_size=bs, num_workers=16, pin_memory=True, shuffle=False)
    print(test_loader)

    nclasses = len(CONFIG.CLASSES)
    lens = test_dataset.__len__()
    print('lens', lens)
    rlt_aver1 = np.zeros([2, len(patchs)])

    for model in modeldic:

        if os.path.basename(model).find('swa') != -1:
            model = torch.optim.swa_utils.AveragedModel(create_model(args.model, False))
            model_dict = model.state_dict()
            state_dict = torch.load(model) 
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if 'module' in k:
                    k = k.replace('module.', '',1)
                new_state_dict[k]=v
            ignore = {k: v for k, v in new_state_dict.items() if k not in model_dict}
            print(ignore)
            weights = {k: v for k, v in new_state_dict.items() if k in model_dict}
            model_dict.update(weights)
            model.load_state_dict(new_state_dict, True)
        else:
            state_dict = torch.load(model) 
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if 'module' in k:
                    k = k.replace('module.', '')
                new_state_dict[k]=v
            model = create_model(args.model, False)

            model_dict = model.state_dict()
            ignore = {k: v for k, v in new_state_dict.items() if k not in model_dict}
            print(ignore)
            weights = {k: v for k, v in new_state_dict.items() if k in model_dict}
            model_dict.update(weights)
            model.load_state_dict(model_dict, True)

        model = model.to(device)
        if torch.cuda.device_count() == 2:
            print(os.environ['CUDA_VISIBLE_DEVICES'])
            model = torch.nn.DataParallel(model, device_ids=[0,1])
        
        elif torch.cuda.device_count() == 4:
            print(os.environ['CUDA_VISIBLE_DEVICES'])
            model = torch.nn.DataParallel(model, device_ids=[0,1,2,3])

        model.eval()


        rlt_aver = np.zeros([2, len(patchs)])
        rlt_count = np.zeros([2, len(patchs)])
        uncertain_index = []

        

        current_patch = 0
        for (inputs, labels) in test_loader:
            tmp = len(labels)
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = F.softmax(outputs, dim=1)
            value, preds = torch.max(outputs, 1)
            outputs, v, p = outputs.detach().cpu().numpy(), value.detach().cpu().numpy(), preds.detach().cpu().numpy()
            for j in range(tmp):
                if outputs[j][0] <= 0.6 and outputs[j][0] >= 0.4:
                    uncertain_index.append(current_patch+j)
                for index in range(2):
                    rlt_aver[index][current_patch + j] = outputs[j][index]
                    rlt_count[index][current_patch + j] = (preds[j] == index)
            current_patch += tmp
        average = rlt_aver.mean(axis = 1)
        count = rlt_count.mean(axis = 1)
        patchs_aver = np.concatenate((patchs_aver, rlt_aver), axis = 1)
        patchs_count = np.concatenate((patchs_count, rlt_count), axis = 1)
        if float(average[1]) > float(args.threshold):
            predict_type = 1
        else:
            predict_type = 0
        print("predict: ", predict_type)
        print(rlt_aver.shape)
        
        rlt_aver = np.array(rlt_aver)
        rlt_aver1 = np.array(rlt_aver1)
        rlt_aver1 = rlt_aver + rlt_aver1
        print(rlt_aver1.shape)

    rlt_aver1 /= 5
    print(rlt_aver1.shape)

    dpi = 64
    position = [get_index(x) for x in patchs]
    print(position)

    width  = max([x for x,y in position]) + 1
    height = max([y for x,y in position]) + 1
    minwidth = min([x for x, y in position]) + 1
    minheight = min([y for x, y in position]) + 1 

    fig = plt.figure(figsize=((height - minheight + 1), (width - minwidth + 1)))
    result = np.ones([(width - minwidth + 1), (height - minheight + 1)]) * -1
    for i, (x, y) in enumerate(position):
        result[(x - minwidth + 1)][(y - minheight + 1)] = (rlt_aver1[0][i])

    s = sns.heatmap(result, cmap="RdYlBu", vmax=1.001, vmin=-0.001, mask=(result==-1), xticklabels=False, yticklabels=False, cbar=False)
    fig.tight_layout()
    plt.savefig(os.path.join(args.resultpath,slidename,
            '{}_heatmap.jpeg'.format(slidename)), dpi = dpi )
    plt.close()

    img = Image.open(os.path.join(args.resultpath,slidename,
            '{}_heatmap.jpeg'.format(slidename)))
    img = img.convert('RGBA')
    for x in range((width - minwidth + 1) * dpi):
        for y in range((height - minheight + 1) * dpi):
            color = img.getpixel((y, x))
            if sum(color) >= 1000:
                color = color[:-1] + (0, )
            else:
                color = color[:-1] + (100, )
            img.putpixel((y, x), color)
    img.save(os.path.join(args.resultpath,slidename,
            '{}_heatmap.png'.format(slidename)))

    img2 = Image.open(os.path.join(args.resultpath, '%s-thumbnails.png' % (slidename.split('.')[0])))
    size1, size2 = img2.size
    r, g, b, a = img.split()
    img2.paste(img, box=((minheight * dpi - dpi), (minwidth * dpi - dpi)), mask = a)
    img2.save(os.path.join(args.resultpath,slidename,
            'paste%s.png' % slidename))
    img3 = Image.open(os.path.join(args.resultpath, '%s-thumbnails.png' % (slidename.split('.')[0])))
    img3 = img3.resize((int((size1)/2), int((size2)/2)),Image.ANTIALIAS)
    img3.save(os.path.join(args.resultpath, slidename, '%s-thumbnail.png' % (slidename.split('.')[0])))








