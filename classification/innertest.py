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
from dataloader_innertest_instant import ZSData
from prefetch_generator import BackgroundGenerator

import numpy as np
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

import config_brain as CONFIG
sys.path.append('/data0/zihan/Datasets')
#from dataloader import Map
from resnetfile import (ResNet, resnet18, resnet34, resnet50, resnet101,
                        resnet152, resnext50_32x4d, resnext101_32x8d)
plt.switch_backend('agg')

data_transforms = transforms.Compose([
    #transforms.Resize([224, 224]),
    transforms.CenterCrop(224),
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

def youden_threshold(all_labels, all_values):
    fpr, tpr, threshold = roc_curve(all_labels, all_values, pos_label=1)
    max_val = 0
    max_i = 0
    for i in range(len(fpr)):
        val = tpr[i] - fpr[i]
        if val > max_val:
            max_val = val
            max_i = i
    print(threshold[max_i], fpr[max_i], tpr[max_i])
    return threshold[max_i]

def create_model(model, pretrain):
    if pretrain == True:
        if model == 'resnet18':
            Model = resnet18(pretrained=True)
        if model == 'resnet50':
            Model = resnet50(pretrained=True)
        if model == 'resnet101':
            Model = resnet101(pretrained=True)
        if model == 'densenet121':
            Model = densenet121(pretrained=True)
    elif pretrain == False:
        if model == 'resnet18':
            Model = resnet18(pretrained=False)
        if model == 'resnet50':
            Model = resnet50(pretrained=False)
        if model == 'resnet101':
            Model = resnet101(pretrained=False)
        if model == 'densenet121':
            Model = densenet121(pretrained=False)
            
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
    
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_id

    if args.device_id is not None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    if os.path.basename(args.model_path).find('swa') != -1:
        model = torch.optim.swa_utils.AveragedModel(create_model(args.model, False))
        model_dict = model.state_dict()
        #print(model_dict)
        state_dict = torch.load(args.model_path) 
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
        state_dict = torch.load(args.model_path) 
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

    path = args.data_path
    test_path = os.path.join(path, args.foldn, 'valid')
    test_cases = {t: set(map(lambda x: x, os.listdir(os.path.join(test_path, t)))) for t in os.listdir(test_path)}
    
    slides, ground_truth, aver, counts = [], [], [], []
    all_labels, all_predicts, all_values = [], [], []
    cases, cases_labels, cases_predicts, cases_values, cases_aver = [], [], [], [], []
    empty_slide = []
    empty_case = []
    patchnum = []
    lensdic = []

    bs = 8

    for t in ['lymphoma', 'glioma']:
        cnt = 0
        for case_name in test_cases[t]:
            slide_names = os.listdir(os.path.join(test_path, t, case_name))
            slide_count = len(slide_names)
            patchs_aver = np.zeros([2, 1])
            patchs_count = np.zeros([2, 1])
            for slidename in slide_names:
                #if slidename != '188109_607500':
                    #continue

                cnt += 1
                groundtruth = t
                ground_truth.append(groundtruth)
                patchs = glob(os.path.join(test_path, t, case_name, slidename, '*.jpeg'))
                random.shuffle(patchs)
                if len(patchs) == 0:
                    empty_slide.append(slidename)
                    continue
                slides.append(slidename)

                test_dataset = ZSData(os.path.join(test_path, t, case_name, slidename), t, transforms = data_transforms, bi = None)
                #print('dataset', test_dataset)
                test_loader = DataLoaderX(test_dataset, batch_size=bs, num_workers=4, pin_memory=True)

                nclasses = len(CONFIG.CLASSES)
                lens = test_dataset.__len__()
                patchnum.append(lens)
                print('lens', lens)
                lensdic.append(lens)

                rlt_aver = np.zeros([2, len(patchs)])
                rlt_count = np.zeros([2, len(patchs)])
                

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
                        for index in range(2):
                            rlt_aver[index][current_patch + j] = outputs[j][index]
                            rlt_count[index][current_patch + j] = (preds[j] == index)
                        #labeldic.append(labels[j])
                    current_patch += tmp

                average = rlt_aver.mean(axis = 1)
                count = rlt_count.mean(axis = 1)
                patchs_aver = np.concatenate((patchs_aver, rlt_aver), axis = 1)
                print(patchs_aver.shape)
                patchs_count = np.concatenate((patchs_count, rlt_count), axis = 1)
                print(average, count)
                aver.append(average)
                counts.append(count)
                all_labels.append(Map[groundtruth])
                if average[1]> 0.31638985:
                    all_predicts.append(1)
                else:
                    all_predicts.append(0)
                all_values.append(average[1])
                print("label: ",Map[groundtruth])
                print("predict: ", np.argmax(average))

            if patchs_aver.shape[1] == 1:
                empty_case.append(case_name)
                continue
            else:
                cases.append(case_name)
                patchs_aver = np.delete(patchs_aver, 0, axis = 1)
                patchs_count = np.delete(patchs_count, 0, axis = 1)
                case_average = patchs_aver.mean(axis = 1)
                case_count = patchs_count.mean(axis = 1)
                cases_aver.append(case_average)
                cases_labels.append(Map[t])
                if case_average[1] > 0.31638985:
                    cases_predicts.append(1)
                else:
                    cases_predicts.append(0)
                cases_values.append(case_average[1])
                print('case %s finish' % case_name)

    all_labels = np.array(all_labels)
    all_predicts = np.array(all_predicts)
    all_values = np.array(all_values)
    eerthreshold = eer_threshold(all_labels, all_values)
    print('eer_threshold:' , eerthreshold)
    youden = youden_threshold(all_labels, all_values)
    print('youden_threshold:' , youden)
    TP = ((all_predicts == 1) & (all_labels == 1)).sum()
    TN = ((all_predicts == 0) & (all_labels == 0)).sum()
    FN = ((all_predicts == 0) & (all_labels == 1)).sum()
    FP = ((all_predicts == 1) & (all_labels == 0)).sum()
    print('slidelevel:')
    print('TP: %d, TN:%d, FN:%d, FP:%d' %(TP, TN, FN, FP))
    p = TP / (TP + FP)
    r = TP / (TP + FN)
    F1 = 2 * r * p / (r + p)
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0    
    ppv = TP / (TP + FP)
    npv = TN / (TN + FN)
    print('lymphoma: precision: {:.4f}, recall: {:.4f}, F1: {:.4f}'.format(p, r, F1))

    print('sensitivity:', sensitivity)
    print('specificity:', specificity)
    lo1, hi1 = clopper_pearson(TP, TP + FN)
    print('Sensitivity 95% confidence interval: ({:.2f}, {:.2f})'.format(lo1, hi1))
    lo2, hi2 = clopper_pearson(TN, FP + TN)
    print('Specificity 95% confidence interval: ({:.2f}, {:.2f})'.format(lo2, hi2))
    print('PPV:', ppv)
    print('NPV:', npv)
    lo3, hi3 = clopper_pearson(TP, TP + FP)
    print('PPV 95% confidence interval: ({:.2f}, {:.2f})'.format(lo3, hi3))
    lo4, hi4 = clopper_pearson(TN, FN + TN)
    print('NPV 95% confidence interval: ({:.2f}, {:.2f})'.format(lo4, hi4))

    print("Acc: ", (all_labels == all_predicts).sum() / len(all_predicts))
    fpr, tpr, theshold = roc_curve(all_labels, all_values, pos_label=1)
    print('AUC: ', auc(fpr, tpr))
    
    all_auc = bootstrap_auc(all_labels, all_values)
    auc_95_ci = scipy.stats.norm.interval(0.95, np.mean(all_auc), np.std(all_auc))
    print("AUC 95% CI:")
    print("(%.4f, %.4f)" % auc_95_ci)

    print('patientlevel:')

    cases_labels = np.array(cases_labels)
    cases_predicts = np.array(cases_predicts)
    cases_values = np.array(cases_values)

    TP = ((cases_predicts == 1) & (cases_labels == 1)).sum()
    TN = ((cases_predicts == 0) & (cases_labels == 0)).sum()
    FN = ((cases_predicts == 0) & (cases_labels == 1)).sum()
    FP = ((cases_predicts == 1) & (cases_labels == 0)).sum()
    print('TP: %d, TN:%d, FN:%d, FP:%d' %(TP, TN, FN, FP))
    p = TP / (TP + FP)
    r = TP / (TP + FN)
    F1 = 2 * r * p / (r + p)
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0    
    ppv = TP / (TP + FP)
    npv = TN / (TN + FN)
    print('lymphoma: precision: {:.4f}, recall: {:.4f}, F1: {:.4f}'.format(p, r, F1))
    print('sensitivity:', sensitivity)
    print('specificity:', specificity)
    lo1, hi1 = clopper_pearson(TP, TP + FN)
    print('Sensitivity 95% confidence interval: ({:.2f}, {:.2f})'.format(lo1, hi1))
    lo2, hi2 = clopper_pearson(TN, FP + TN)
    print('Specificity 95% confidence interval: ({:.2f}, {:.2f})'.format(lo2, hi2))
    print('PPV:', ppv)
    print('NPV:', npv)
    lo3, hi3 = clopper_pearson(TP, TP + FP)
    print('PPV 95% confidence interval: ({:.2f}, {:.2f})'.format(lo3, hi3))
    lo4, hi4 = clopper_pearson(TN, FN + TN)
    print('NPV 95% confidence interval: ({:.2f}, {:.2f})'.format(lo4, hi4))

    print("Acc: ", (cases_labels == cases_predicts).sum() / len(cases_predicts))
    fpr, tpr, theshold = roc_curve(cases_labels, cases_values, pos_label=1)
    print('AUC: ', auc(fpr, tpr))
    
    cases_auc = bootstrap_auc(cases_labels, cases_values)
    auc_95_ci = scipy.stats.norm.interval(0.95, np.mean(cases_auc), np.std(cases_auc))
    print("AUC 95% CI:")
    print("(%.4f, %.4f)" % auc_95_ci)


    modeldir = os.path.basename(os.path.dirname(os.path.dirname(args.model_path)))
    if not os.path.exists(os.path.join(args.resultpath, 'collected_6-4', modeldir)):
        os.makedirs(os.path.join(args.resultpath, 'collected_6-4', modeldir))    
    resultpath = os.path.join(args.resultpath, 'collected_6-4', modeldir)

    df = DataFrame(
        {'Slide:': slides,
        'Type:': all_labels,
        'predicting:': all_predicts,
        'Averaging': all_values,
        'patch_sum': lensdic}
    )
    #df.to_excel(os.path.join(resultpath, 'slidelevel_direct_inner_result_fold' + args.foldn + '.xlsx'))

    df2 = DataFrame(
        {'case:': cases,
        'Type:': cases_labels,
        'Predicting:': cases_predicts,
        'Averaging': cases_values
        }
        
    )
    #df2.to_excel(os.path.join(resultpath, 'patientlevel_direct_inner_result_fold' + args.foldn + '.xlsx'))



