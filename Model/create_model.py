import torch.nn as nn
from Model.resnet import resnet18, resnet34, resnet50, resnet101
from efficientnet_pytorch import EfficientNet
from torchvision import models

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
        if model == 'efficientnet-b0':
            Model = EfficientNet.from_name('efficientnet-b0')
        if model == 'vgg16':
            Model = models.vgg16(pretrained=False)
            infeatures = Model.classifier[6].in_features
            Model.classifier[6] = nn.Linear(in_features = infeatures, out_features = 2, bias = True)
    print(model)
    
    return Model