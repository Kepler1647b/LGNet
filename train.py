# -*- coding: utf-8 -*
import os
import time
import numpy as np
import Utils.config_brain as CONFIG
from argparse import ArgumentParser
from collections import OrderedDict
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from Dataloader_LGNet.dataloader_nodown import ZSData

import matplotlib.pyplot as plt
plt.switch_backend('agg')
from prefetch_generator import BackgroundGenerator
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from torch.utils.tensorboard import SummaryWriter
from Model.create_model import create_model

from Utils.utils import seed_torch, transform_valid, SCELoss
import Utils.utils.transform_train as transform

Writer = None
bit = True

class DataloaderX(torch.utils.data.DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def run(model, dataloaders, phase, criterion, criterion2, optimizer, n_count):
    with tqdm(total=n_count, desc = phase, unit='img', ncols = 100) as pbar:

        if phase == 'train':
            model.train()
        elif phase == 'valid':
            model.eval()
        else:
            raise Exception('Error Phase')

        all_labels = np.array([])
        all_predicts = np.array([])
        all_values = np.array([])
        running_loss = 0.0
        for input, label in dataloaders:
            input, label = input.cuda(non_blocking = True), label.cuda(non_blocking = True)
            with torch.torch.set_grad_enabled(phase == 'train'):
                pbar.update(len(input))
                output = model(input)
                _, pred = torch.max(output, 1)
                value = output[: , 1]
                loss = criterion(output, label)
                optimizer.zero_grad()
                if phase == 'train':
                    loss = criterion(output, label)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                elif phase == 'valid':
                    loss = criterion2(output, label)
                    optimizer.zero_grad()
            pbar.set_postfix(**{'loss (batch)': loss.item()})
                    
            running_loss += loss.item() * input.size(0)
            all_labels = np.concatenate((all_labels, label.cpu().numpy()))
            all_predicts = np.concatenate((all_predicts, pred.cpu().numpy()))
            all_values = np.concatenate((all_values, value.detach().cpu().numpy()))

        Loss = running_loss / len(all_labels)
        Acc1 = (all_predicts == all_labels).sum()
        Acc = Acc1 / len(all_labels)
        TP = ((all_predicts == 1) & (all_labels == 1)).sum()
        TN = ((all_predicts == 0) & (all_labels == 0)).sum()
        FN = ((all_predicts == 0) & (all_labels == 1)).sum()
        FP = ((all_predicts == 1) & (all_labels == 0)).sum()

        p = TP / (TP + FP)
        r = TP / (TP + FN)
        F1 = 2 * r * p / (r + p)
        glioma_p = TN / (TN + FN)
        glioma_r = TN / (TN + FP)
        conf_matrix = confusion_matrix(all_labels, all_predicts, labels=list(range(CONFIG.NUM_CLASSES)))
        print('%s loss:' % phase, Loss)
        print('%s Acc:' % phase, Acc)

        Writer.add_scalar('%s/Loss' % phase, Loss, global_epoch)
        Writer.add_scalar('%s/Acc' % phase, Acc, global_epoch)

        fpr, tpr, theshold = roc_curve(all_labels, all_values, pos_label=1)
        print('AUC', auc(fpr, tpr))
        
        Writer.add_scalar('{}/AUC'.format(phase.capitalize()), auc(fpr, tpr), global_epoch)
        Writer.add_scalar('%s/lymphoma_Precision' % phase.capitalize(), p, global_epoch)
        Writer.add_scalar('%s/lymphoma_Recall' % phase.capitalize(), r, global_epoch)
        Writer.add_scalar('%s/lymphoma_F1' % phase.capitalize(), F1, global_epoch)
        Writer.add_scalar('%s/glioma_Precision' % phase.capitalize(), glioma_p, global_epoch)
        Writer.add_scalar('%s/glioma_Recall' % phase.capitalize(), glioma_r, global_epoch)
        Writer.add_scalar('%s/sum_Precision' % phase.capitalize(), (p+glioma_p), global_epoch)
        Writer.add_scalar('%s/sum_Recall' % phase.capitalize(), (r+glioma_r), global_epoch)

        return Loss, p, r, F1, Acc, auc(fpr, tpr), conf_matrix, (r+glioma_r)

def start_train(train_loader, valid_loader, model, device, criterion, criterion2, optimizer, scheduler, num_epochs, n_train, n_valid, swa_model, swa_start, swa_scheduler):
    best_F1 = .0
    best_auc = .0
    best_acc = .0
    best_loss = 1000000 
    best_recall = .0
    best_sum_r = .0
     
    for epoch in range(1, num_epochs + 1):

        global global_epoch
        global_epoch = epoch
        print('epoch:' , global_epoch)

        train_loss, train_p, train_r, train_F1, train_acc, train_auc, train_conf_matrix,_ = run(model, train_loader, 'train', criterion, criterion2, optimizer, n_train)
        valid_loss, valid_p, valid_r, valid_F1, valid_acc, valid_auc, valid_conf_matrix,valid_sumr = run(model, valid_loader, 'valid', criterion, criterion2, optimizer, n_valid)

        if epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            last_lr = swa_scheduler.get_last_lr()[0]
            Writer.add_scalar('Learing rate', last_lr, epoch)
        else:
            scheduler.step()
            last_lr = scheduler.get_last_lr()[0]
            Writer.add_scalar('Learing rate', last_lr, epoch)

        if valid_F1 > best_F1:
            best_F1 = valid_F1
            torch.save(model.state_dict(), os.path.join(checkpoint, 'bestF1.pt'))     

        if valid_loss < best_loss:
            best_loss = valid_loss
            es = 0
            torch.save(model.state_dict(), os.path.join(
                checkpoint, 'bestloss.pt')
            )
        else:
            es += 1
            print("Counter {} of 10".format(es))

        if es > 99:
            print("Early stopping with epoch %i" % epoch)
            torch.save(model.state_dict(), os.path.join(
                checkpoint, 'es.pt'))
            break            

        if valid_auc > best_auc:
            best_auc = valid_auc
            torch.save(model.state_dict(), os.path.join(
                checkpoint, 'bestauc.pt')
            )
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), os.path.join(
                checkpoint, 'bestacc.pt')
            )
        if valid_auc > best_auc:
            best_auc = valid_auc
            torch.save(model.state_dict(), os.path.join(
                checkpoint, 'bestauc.pt')
            )
        if valid_r > best_recall:
            best_recall = valid_r
            torch.save(model.state_dict(), os.path.join(
                checkpoint, 'bestrecall.pt')
            )
        if valid_sumr > best_sum_r:
            best_sum_r = valid_sumr
            torch.save(model.state_dict(), os.path.join(
                checkpoint, 'best_sumrecall.pt')
            )
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(checkpoint, '%i.pt' % epoch))   

        print('epoch %s complete' % global_epoch)
    torch.save(model.state_dict(), os.path.join(checkpoint, 'final.pt'))

    torch.optim.swa_utils.update_bn(train_loader, swa_model)   
    torch.save(swa_model.state_dict(), os.path.join(checkpoint, 'swa.pt'))   


def make_weights_for_balanced_classes(label_list, nclasses):
    count = [0] * nclasses
    print(label_list)
    for item in label_list:
        count[item] += 1
    weight_per_class = [0.0] * nclasses
    N = float(sum(count))
    print(N)
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(label_list)
    for num, label in enumerate(label_list):
        weight[num] = weight_per_class[label]
    return weight, weight_per_class

def prepare_data(img_path):
    train_dataset = ZSData(os.path.join(img_path, 'train'), '', transforms = transform, bi = bit)
    valid_dataset = ZSData(os.path.join(img_path, 'valid'), '', transforms = transform_valid, bi = bit)
    nclasses = len(CONFIG.CLASSES)
    weights, weights_per_class = make_weights_for_balanced_classes(train_dataset.get_label_list(), nclasses)
    train_loader = DataloaderX(train_dataset, shuffle = True, batch_size = batch_size, num_workers = 16, pin_memory = True)
    valid_loader = DataloaderX(valid_dataset, shuffle = True, batch_size = batch_size, num_workers = 16, pin_memory = True)

    return train_loader, valid_loader, weights_per_class, train_dataset.__len__(), valid_dataset.__len__()

if __name__ == '__main__':
    

    parser = ArgumentParser()
    parser.add_argument("--TrainFolder", dest='train_folder', type=str)
    parser.add_argument("--FoldN", dest='foldn', type=int)
    parser.add_argument("--Loss", dest='loss_func', type=str)
    parser.add_argument("--NumEpoch", dest='num_epochs', type=int)
    parser.add_argument("--Seed", dest='seed', type=int)
    parser.add_argument("--Model", dest='model', type=str)
    parser.add_argument("--LearningRate", dest='learning_rate', type=float)
    parser.add_argument("--BatchSize", dest='batch_size', type=int)
    parser.add_argument("--WeightDecay", dest='weight_decay', type=float)# learning rate decay
    parser.add_argument("--DeviceId", dest='device_id', type=str)
    parser.add_argument("--Comment", dest='comment', type=str)
    parser.add_argument("--Pretrain", dest='pretrain', action="store_false")
    parser.add_argument("--Modelpath", dest='modelpath', type=str)
    
    args = parser.parse_args()
    
    seed_torch(args.seed)
    
    batch_size = args.batch_size
    
    loss_func = args.loss_func
   
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_id
    os.environ['TORCH_HOME']='./pretrained_model'#path of imgnet-pretrained model checkpoint
    now = time.strftime("%Y_%m_%d_", time.localtime())
    
    global checkpoint
    checkpoint = os.path.join(CONFIG.CHECK_POINT, now+'split4-1_direct_classification,'+args.model,
        now+"split4-1_direct_classification,"+args.model+','+str(args.batch_size)+","+"seed"+str(args.seed)+",fold"+str(args.foldn))
    
    print('Summary write in %s' % checkpoint)
    
    Writer = SummaryWriter(log_dir=checkpoint)

    train_loader, valid_loader, weight_per_class, n_train, n_valid = prepare_data(os.path.join(args.train_folder, str(args.foldn)))

    print(len(CONFIG.CLASSES), 'classes:', CONFIG.CLASSES)
    
    print('num train images %d x %d' % (len(train_loader), args.batch_size))
    
    print('num val images %d x %d' % (len(valid_loader), args.batch_size))
    
    print("CUDA is_available:", torch.cuda.is_available())
    
    if args.device_id is not None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    print('\n### Build model %s' % args.model)

    
    if args.modelpath != None:
        state_dict = torch.load(args.modelpath) 
        state_dict = state_dict['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'backbone' in k:
                k = k.replace('module.backbone.', '')
            new_state_dict[k]=v
        model = create_model(args.model, args.pretrain)

        model_dict = model.state_dict()
        ignore = {k: v for k, v in new_state_dict.items() if k not in model_dict}

        weights = {k: v for k, v in new_state_dict.items() if k in model_dict}
        model_dict.update(weights)
        model.load_state_dict(model_dict, True)
    else:
        model = create_model(args.model, args.pretrain)
    if torch.cuda.device_count() == 2:
        print(os.environ['CUDA_VISIBLE_DEVICES'])
        model = torch.nn.DataParallel(model, device_ids=[0,1])
    elif torch.cuda.device_count() == 4:
        print(os.environ['CUDA_VISIBLE_DEVICES'])
        model = torch.nn.DataParallel(model, device_ids=[0,1,2,3])
    model = model.to(device)

    # cross entropy loss
    if loss_func == 'cross':
        weight_per_class = torch.Tensor(weight_per_class).to(device)
        criterion = nn.CrossEntropyLoss(weight = weight_per_class)
        criterion2 = nn.CrossEntropyLoss()
        criterionsce = SCELoss(a=0.1, b=1, num_classes=2)
    optimizer = optim.SGD(model.parameters(),
                            momentum = 0.9,
                            lr=args.learning_rate,
                            weight_decay=args.weight_decay)
                            
    #scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=0.)

    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_start = 95
    swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=0.0001)
    start_train(train_loader, valid_loader, model, device, criterionsce, criterion2, 
                optimizer, scheduler, args.num_epochs, n_train, n_valid, swa_model, swa_start, swa_scheduler)



        

