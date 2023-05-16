# -*- coding: utf-8 -*
import os
import random
import re
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
from sklearn.metrics import auc, roc_curve
from torch.utils.data import DataLoader
import scipy

import Utils.config_brain as CONFIG
from Utils.config_brain import Map
plt.switch_backend('agg')
from Model.create_model import create_model
from Utils.utils import bootstrap_auc, clopper_pearson, eer_threshold, youden_threshold
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
    
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_id

    if args.device_id is not None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    if os.path.basename(args.model_path).find('swa') != -1:
        model = torch.optim.swa_utils.AveragedModel(create_model(args.model, False))
        model_dict = model.state_dict()
        state_dict = torch.load(args.model_path)
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
        state_dict = torch.load(args.model_path) 
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

    test_path = os.path.join(args.data_path, args.foldn, 'valid')
    test_cases = {t: set(map(lambda x: x, os.listdir(os.path.join(test_path, t)))) for t in os.listdir(test_path)}
    
    slides, aver, counts = [], [], [], []
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
                cnt += 1
                groundtruth = t
                patchs = glob(os.path.join(test_path, t, case_name, slidename, '*.jpeg'))
                random.shuffle(patchs)
                if len(patchs) == 0:
                    empty_slide.append(slidename)
                    continue
                slides.append(slidename)

                test_dataset = ZSData(os.path.join(test_path, t, case_name, slidename), t, transforms = data_transforms, bi = None)
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

    df = DataFrame(
        {'Slide:': slides,
        'Type:': all_labels,
        'predicting:': all_predicts,
        'Averaging': all_values,
        'patch_sum': lensdic}
    )
    df.to_excel(os.path.join(args.resultpath, 'slidelevel_direct_inner_result_fold' + args.foldn + '.xlsx'))

    df2 = DataFrame(
        {'case:': cases,
        'Type:': cases_labels,
        'Predicting:': cases_predicts,
        'Averaging': cases_values
        }
        
    )
    df2.to_excel(os.path.join(args.resultpath, 'patientlevel_direct_inner_result_fold' + args.foldn + '.xlsx'))



