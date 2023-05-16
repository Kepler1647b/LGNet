# -*- coding: utf-8 -*
import os
import random
from argparse import ArgumentParser
from glob import glob
from collections import OrderedDict
from Dataloader_LGNet.dataloader_instant import ZSData
from prefetch_generator import BackgroundGenerator
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from pandas import DataFrame
from torch.utils.data import DataLoader
import time
from Model.create_model import create_model

import Utils.config_brain as CONFIG
plt.switch_backend('agg')
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

    modelpath1 = args.model_path
    modeldic1 = []
    for modelfile in glob(os.path.join(args.model_path, '*')):
        modeldic1.append(os.path.join(modelfile, 'final.pt'))
    modeldic1.sort()
    path = args.data_path
    test_path = path
    print(modeldic1)


    path = args.data_path
    test_path = path
    test_cases = {t: set(map(lambda x: x, os.listdir(os.path.join(test_path, t)))) for t in os.listdir(test_path)}
    
    slides, aver, counts = [], [], []
    all_labels, all_predicts, all_values = [], [], []
    lensdic = []
    uncertaindic = []
    timedic = []
    confidencedic = []
    bs = 32

    cnt = 0
    for slidename in os.listdir(test_path):
        time_start = time.time()
        patchs_aver = np.zeros([2, 1])
        patchs_count = np.zeros([2, 1])
        print(slidename)

        cnt += 1
        if not os.path.exists(os.path.join(test_path, slidename)):
            continue
        patchs = glob(os.path.join(test_path, slidename, '*.jpeg'))
        random.shuffle(patchs)
        slides.append(slidename)

        test_dataset = ZSData(os.path.join(test_path, slidename), 'poc', transforms = data_transforms, bi = None)
        test_loader = DataLoaderX(test_dataset, batch_size=bs, num_workers=16, pin_memory=True)

        nclasses = len(CONFIG.CLASSES)
        lens = test_dataset.__len__()
        print('lens', lens)
        lensdic.append(lens)
        rlt_aver1 = np.zeros([2, len(patchs)])

        for model in modeldic1:

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
            
            rlt_aver = np.array(rlt_aver)
            rlt_aver1 = np.array(rlt_aver1)
            rlt_aver1 += rlt_aver
            print(rlt_aver1.shape)

        rlt_aver1 /= 5
        print(rlt_aver1.shape)
        average = rlt_aver1.mean(axis = 1)
        threshold = float(args.threshold)
        if float(average[1]) > float(args.threshold):
            predict_type = 1
            all_predicts.append(1)
            confidence = (float(average[1])-threshold)/(1-threshold)
        else:
            predict_type = 0
            all_predicts.append(0)
            confidence = (float(average[1])-threshold)/(0-threshold)
        all_values.append(average[1])
        confidencedic.append(confidence)
        print("predict: ", predict_type)
        time_end = time.time()
        print('time cost',time_end-time_start,'s')
        print('values:', average[1])
        print("confidence:", confidence)
        timedic.append(time_end-time_start)

    df = DataFrame(
        {'Slide:': slides,
        'predicting:': all_predicts,
        'Averaging': all_values,
        'Confidence': confidencedic,
        'Patch_sum': lensdic,
        'Time': timedic}
    )
    df.to_excel(os.path.join(args.resultpath, 'ensemble_poc.xlsx'))








