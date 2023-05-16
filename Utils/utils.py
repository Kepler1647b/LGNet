from torchvision import transforms
import random
import os
import numpy as np
import torch
import itertools
from matplotlib import pyplot as plt
import scipy
from sklearn.metrics import roc_curve, roc_auc_score
import math
import scipy
import math
import re
import Utils.config_brain as CONFIG
import torch.nn as nn
import torch.nn.functional as F

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class Random90Rotation():
    def __init__(self, degree = [0, 90, 180, 270]):
        self.degree = degree
    def __call__(self, img):
        degree = random.sample(self.degree, k=1)[0]
        return img.rotate(degree)

transform_train = transforms.Compose([
    #transforms.Resize(256),
    transforms.RandomCrop(224),
    Random90Rotation(),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.8)], p=0.8),
    transforms.RandomApply([transforms.ColorJitter(contrast=0.8)], p=0.8),
    transforms.RandomApply([transforms.ColorJitter(saturation=0.8, hue=0.2)], p=0.8),

    transforms.ToTensor(),
    transforms.Normalize(
        mean = CONFIG.Vahadane_10x_Mean, std = CONFIG.Vahadane_10x_Std
    ),
])

transform_valid = transforms.Compose([
    #transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = CONFIG.Vahadane_10x_Mean, std = CONFIG.Vahadane_10x_Std
    ),
])

transform_test = transforms.Compose([
    #transforms.Resize([224, 224]),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = CONFIG.Vahadane_10x_Mean, std = CONFIG.Vahadane_10x_Std
    ),
])


def bootstrap_auc(all_labels, all_values, n_bootstraps=1000):
    rng_seed = 1
    bootstrapped_scores = []

    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        indices = rng.randint(0, len(all_values), len(all_values))
        if len(np.unique(all_labels[indices])) < 2:
            continue
        score = roc_auc_score(all_labels[indices], all_values[indices])
        bootstrapped_scores.append(score)
    return bootstrapped_scores

def clopper_pearson(x, n, alpha=0.05):
    b = scipy.stats.beta.ppf
    lo = b(alpha / 2, x, n - x + 1)
    hi = b(1 - alpha / 2, x + 1, n - x)
    return 0.0 if math.isnan(lo) else lo, 1.0 if math.isnan(hi) else hi

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

class SCELoss(nn.Module):
    def __init__(self, num_classes=2, a=0.1, b=1):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_classes = num_classes
        self.a = a
        self.b = b
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        ce = self.cross_entropy(pred, labels)
        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))

        loss = self.a * ce + self.b * rce.mean()
        return loss



