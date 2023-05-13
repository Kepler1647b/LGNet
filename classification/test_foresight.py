# -*- coding: utf-8 -*
import os
import random
import re
from argparse import ArgumentParser
from glob import glob
from math import ceil, fabs, floor
from posix import listdir
import sys
from PIL import Image
import openslide
from collections import OrderedDict
from dataloader_inner_instant import ZSData
from prefetch_generator import BackgroundGenerator
from efficientnet_pytorch import EfficientNet
import numpy as np
import numpy
import openpyxl
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from matplotlib import pyplot as plt
from pandas import DataFrame
from PIL import Image
from sklearn.metrics import auc, confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import scipy
import math
import time
import imageio


import config_brain as CONFIG
sys.path.append('/data0/zihan/Datasets')
#from dataloader import Map
from resnetfile import (ResNet, resnet18, resnet34, resnet50, resnet101,
                        resnet152, resnext50_32x4d, resnext101_32x8d)
plt.switch_backend('agg')

data_transforms = transforms.Compose([
    #transforms.Resize([224, 224]),
    transforms.CenterCrop(224),#图片随机裁剪
    transforms.ToTensor(),
    transforms.Normalize(
        mean = CONFIG.Vahadane_10x_Mean, std = CONFIG.Vahadane_10x_Std
    ),
])

def bootstrap_auc(all_labels, all_values, n_bootstraps=1000):
    rng_seed = 1  # control reproducibility
    bootstrapped_scores = []

    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(all_values), len(all_values))
        if len(np.unique(all_labels[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
        score = roc_auc_score(all_labels[indices], all_values[indices])
        bootstrapped_scores.append(score)
#         print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))
    return bootstrapped_scores

def clopper_pearson(x, n, alpha=0.05):
    """Estimate the confidence interval for a sampled Bernoulli random
    variable.
    `x` is the number of successes and `n` is the number trials (x <=
    n). `alpha` is the confidence level (i.e., the true probability is
    inside the confidence interval with probability 1-alpha). The
    function returns a `(low, high)` pair of numbers indicating the
    interval on the probability.
    """
    b = scipy.stats.beta.ppf
    lo = b(alpha / 2, x, n - x + 1)
    hi = b(1 - alpha / 2, x + 1, n - x)
    return 0.0 if math.isnan(lo) else lo, 1.0 if math.isnan(hi) else hi

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def get_index(path):
    path = os.path.basename(path)
    y, x = list(map(int, re.findall(r'\d+', path)[-2: ]))
    return x, y

def eer_threshold(all_labels, all_values):
    fpr, tpr, threshold = roc_curve(all_labels, all_values, pos_label=1)
    min_val = 999
    min_i = 0
    for i in range(len(fpr)):
        val = abs(fpr[i] + tpr[i] - 1)
        if val < min_val:
            min_val = val
            min_i = i
    print(threshold[min_i], fpr[min_i], tpr[min_i])
    return threshold[min_i]

def create_model(model, pretrain = True):
    if pretrain == True:
        if model == 'resnet18':
            Model = resnet18(pretrained=True)
        if model == 'resnet34':
            Model = resnet34(pretrained=True)
        if model == 'resnet50':
            Model = resnet50(pretrained=True)
        if model == 'resnet101':
            Model = resnet101(pretrained=True)
        if model == 'densenet121':
            Model = densenet121(pretrained=True)
        if model == 'efficientnet-b0':
            Model = EfficientNet.from_pretrained('efficientnet-b0')
        if model == 'vgg16':
            Model = models.vgg16(pretrained=True)
            infeatures = Model.classifier[6].in_features
            Model.classifier[6] = nn.Linear(in_features = infeatures, out_features = 2, bias = True)
    elif pretrain == False:
        if model == 'resnet18':
            Model = resnet18(pretrained=False)
        if model == 'resnet34':
            Model = resnet34(pretrained=False)
        if model == 'resnet50':
            Model = resnet50(pretrained=False)
        if model == 'resnet101':
            Model = resnet101(pretrained=False)
        if model == 'densenet121':
            Model = densenet121(pretrained=False)
        if model == 'efficientnet-b0':
            Model = EfficientNet.from_name('efficientnet-b0')
        if model == 'vgg16':
            Model = models.vgg16(pretrained=False)
            infeatures = Model.classifier[6].in_features
            Model.classifier[6] = nn.Linear(in_features = infeatures, out_features = 2, bias = True)
    print(model)
    
    return Model

Map = {
    'lymphoma': 1,
    'glioma': 0,
}

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
    
    slides, ground_truth, aver, counts = [], [], [], []
    all_labels, all_predicts, all_values = [], [], []
    cases, cases_labels, cases_predicts, cases_values, cases_aver = [], [], [], [], []
    empty_slide = []
    empty_case = []
    patchnum = []
    lensdic = []
    uncertaindic = []
    timedic = []
    confidencedic = []
    #slidedic = ['223932', '224018', '224392', '227922', '227927', '227932']
    bs = 32

    #for t in ['lymphoma', 'glioma']:
    cnt = 0
    for slidename in os.listdir(test_path):
        time_start = time.time()
        #if slidename not in ['234568_937596001', '224216_934157006']:
            #continue
        #if not os.path.exists(os.path.join(args.resultpath, slidename)):
            #os.makedirs(os.path.join(args.resultpath, slidename))
        patchs_aver = np.zeros([2, 1])
        patchs_count = np.zeros([2, 1])
        print(slidename)

        cnt += 1
        groundtruth = 'glioma'
        ground_truth.append(groundtruth)
        #patchs = glob(os.path.join(test_path, t, slidename,'tissue', '*.jpeg'))###novahadane
        if not os.path.exists(os.path.join(test_path, slidename)):
            continue
        patchs = glob(os.path.join(test_path, slidename, '*.jpeg'))
        random.shuffle(patchs)
        if len(patchs) == 0:
            empty_slide.append(slidename)
            continue
        slides.append(slidename)

        #test_dataset = ZSData(os.path.join(test_path, t, slidename, 'tissue'), t, transforms = data_transforms, bi = None)###novahadane
        test_dataset = ZSData(os.path.join(test_path, slidename), 'glioma', transforms = data_transforms, bi = None)
        #print('dataset', test_dataset)
        test_loader = DataLoaderX(test_dataset, batch_size=bs, num_workers=16, pin_memory=True)

        nclasses = len(CONFIG.CLASSES)
        lens = test_dataset.__len__()
        patchnum.append(lens)
        print('lens', lens)
        lensdic.append(lens)
        rlt_aver1 = np.zeros([2, len(patchs)])

        for model in modeldic1:

            if os.path.basename(model).find('swa') != -1:
                model = torch.optim.swa_utils.AveragedModel(create_model(args.model, False))
                model_dict = model.state_dict()
                #print(model_dict)
                state_dict = torch.load(model) 
                #print(torch.load(args.model_path))
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    if 'module' in k:
                        k = k.replace('module.', '',1)
                    new_state_dict[k]=v
                ignore = {k: v for k, v in new_state_dict.items() if k not in model_dict}
                print(ignore)
                weights = {k: v for k, v in new_state_dict.items() if k in model_dict}
                #print(weights)
                model_dict.update(weights)
                model.load_state_dict(new_state_dict, True)


                #model.load_state_dict(torch.load(args.model_path))
                #model.cuda()
            
            else:
                state_dict = torch.load(model) 
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    if 'module' in k:
                        k = k.replace('module.', '')
                    new_state_dict[k]=v
                model = create_model(args.model, False)

                model_dict = model.state_dict()
                #print(model_dict)
                #for k, v in model_dict.items():
                    #print(k)
                ignore = {k: v for k, v in new_state_dict.items() if k not in model_dict}
                print(ignore)
                weights = {k: v for k, v in new_state_dict.items() if k in model_dict}
                #print(weights)
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
                #print(inputs[0])
                #print(inputs)
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
                    #labeldic.append(labels[j])
                current_patch += tmp
            
            rlt_aver = np.array(rlt_aver)
            rlt_aver1 = np.array(rlt_aver1)
            rlt_aver1 += rlt_aver
            print(rlt_aver1.shape)

        rlt_aver1 /= 5
        print(rlt_aver1.shape)
        average = rlt_aver1.mean(axis = 1)#预测值的平均
        #all_labels.append(Map[groundtruth])
        #all_predicts.append(np.argmax(average))
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
        #print("label: ",Map[groundtruth])
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
    #df.to_excel(os.path.join(args.resultpath, 'ensemble_foresight.xlsx'))







