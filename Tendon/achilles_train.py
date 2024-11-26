import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from copy import deepcopy
import time
from tempfile import TemporaryDirectory
import random
import math
import yaml

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, utils
from torchvision.io import read_image
from torch.utils.data import Dataset, default_collate
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, ConcatDataset, random_split, RandomSampler
from torcheval.metrics import MulticlassPrecisionRecallCurve
from torcheval.metrics.functional import binary_auprc

from sklearn import metrics as skmet

# Local
from utils import pad, num_parameters, EarlyStopper, custom_collate
from datasets import Achilles
from models import AchillesNet
import transforms

np.random.seed(1234)
random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


'''
--------------------------
*** TRAINING FUNCTIONS ***
--------------------------
'''
def train_one_epoch(model, optimizer, scheduler, train_loader, device):
    model.train()

    num_steps_per_epoch = len(train_loader)

    target_ls = []
    output_ls = []
    losses = []
    for i, batch in enumerate(train_loader):
        inputs = batch['study'].unsqueeze(1).to(device).type(torch.float32)
        num_imgs = batch['num_imgs']
        targets = batch['label'].to(device).type(torch.float32)
        target_ls.append(targets.cpu().detach().numpy())
        #print(inputs.shape)
        #print(targets.shape)
        #print(len(num_imgs))
        outputs, _ = model(inputs, num_imgs)
        output_ls.append(outputs.cpu().detach().numpy())
        #print(outputs.shape)
        #print(outputs)
        loss = nn.functional.binary_cross_entropy_with_logits(outputs.squeeze(), targets.squeeze())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        losses.append(loss.detach().item())
        #print(f"\tBatch {i+1} of {num_steps_per_epoch}. Loss={loss.detach().item():0.3f}", end='\r')
    
    #print(' '*100, end='\r')
    
    metrics = compute_metrics(np.concatenate(target_ls), np.concatenate(output_ls))
    return np.mean(losses), metrics 
            
def evaluate(model, test_loader, device):
    model.eval()

    num_steps_per_epoch = len(test_loader)

    target_ls = []
    output_ls = []
    losses = []
    for i, batch in enumerate(test_loader):
        inputs = batch['study'].unsqueeze(1).to(device).type(torch.float32)
        num_imgs = batch['num_imgs']
        targets = batch['label'].to(device).type(torch.float32)
        target_ls.append(targets.cpu().numpy())
        
        with torch.no_grad():
            outputs, _ = model(inputs, num_imgs)
            output_ls.append(outputs.cpu().numpy())
            loss = nn.functional.binary_cross_entropy_with_logits(outputs.squeeze(), targets.squeeze())
            
        losses.append(loss.detach().item())
        
    metrics = compute_metrics(np.concatenate(target_ls), np.concatenate(output_ls))
    return np.mean(losses), metrics

def compute_metrics(y_true, y_pred):
    mets = dict()
    
    y_pred = 1/(1+np.exp(-y_pred)) #sigmoid to get probability 0-1
    y_pred_cls = np.rint(y_pred)
    prec,recall,thresh = skmet.precision_recall_curve(y_true, y_pred)
    
    mets['roc_auc'] = skmet.roc_auc_score(y_true, y_pred)
    mets['prc_auc'] = skmet.auc(recall,prec)
    mets['conf_mat'] = skmet.confusion_matrix(y_true, y_pred_cls)
    mets['accuracy'] = skmet.accuracy_score(y_true, y_pred_cls)
    mets['precision'] = skmet.precision_score(y_true, y_pred_cls,zero_division=0)
    mets['sensitivity'] = skmet.recall_score(y_true, y_pred_cls,zero_division=0)
    mets['specificity'] = skmet.recall_score(y_true, y_pred_cls, pos_label=0,zero_division=0)
    mets['f1score'] = skmet.f1_score(y_true, y_pred_cls,zero_division=0)
    
    return mets


def main(cfg):
    
    os.makedirs(cfg['save_dir'], exist_ok=True)
    
    # copy the config file to the artifact folder
    with open(cfg['save_dir'] + '/config.yaml', 'w') as f: 
        yaml.dump(cfg, f)

    
    # Encoder and Model
    encoder = models.resnet18(weights='IMAGENET1K_V1')
    encoder.conv1 = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
    )
    num_features = encoder.fc.in_features
    encoder.fc = nn.Identity()
    
    num_pars_encoder = num_parameters(encoder)
    model = AchillesNet(encoder, num_features=num_features, **cfg['net_kwargs'])
    model = model.to(device)
    num_pars = num_parameters(model)
    
    # Datasets
    train_df = pd.read_csv(cfg['train_csv'])
    val_df = pd.read_csv(cfg['val_csv'])
    train_dataset = Achilles(train_df, transform=transforms.transform_augment)
    val_dataset = Achilles(val_df, transform=transforms.transform_original)
    # Dataloaders
    train_loader = DataLoader(train_dataset, shuffle=True, collate_fn=custom_collate, **cfg['loader_kwargs'])
    val_loader = DataLoader(val_dataset, shuffle=True, collate_fn=custom_collate, **cfg['loader_kwargs'])

    # Hyperparameters and arguments
    optimizer = optim.Adam(model.parameters(),**cfg['optim_kwargs'])
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, **cfg['sched_kwargs'])

    train_loss_ls = []
    val_loss_ls = []
    train_acc_ls = []
    val_acc_ls = []
    train_prec_ls = []
    val_prec_ls = []
    train_sens_ls = []
    val_sens_ls = []
    train_spec_ls = []
    val_spec_ls = []
    train_rocauc_ls = []
    val_rocauc_ls = []
    train_prcauc_ls = []
    val_prcauc_ls = []
    train_f1_ls = []
    val_f1_ls = []
    best_loss = 100

    epochs = cfg['num_epochs']
    verbose = cfg['verbose']
    since = time.time()
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}\n","-"*10) if verbose else None
        
        # train for a single epoch
        train_loss, train_metrics = train_one_epoch(model, optimizer, lr_scheduler, train_loader, device)
        train_loss_ls.append(train_loss)
        train_acc_ls.append(train_metrics['accuracy'])
        train_prec_ls.append(train_metrics['precision'])
        train_sens_ls.append(train_metrics['sensitivity'])
        train_spec_ls.append(train_metrics['specificity'])
        train_rocauc_ls.append(train_metrics['roc_auc'])
        train_prcauc_ls.append(train_metrics['prc_auc'])
        train_f1_ls.append(train_metrics['f1score'])
        print(f"[TRAIN] BCE loss: {train_loss:0.4f} | Acc: {100*train_metrics['accuracy']:0.2f}%") if verbose else None

        # evaluate
        val_loss, val_metrics = evaluate(model, val_loader, device)
        val_loss_ls.append(val_loss)
        val_acc_ls.append(val_metrics['accuracy'])
        val_prec_ls.append(val_metrics['precision'])
        val_sens_ls.append(val_metrics['sensitivity'])
        val_spec_ls.append(val_metrics['specificity'])
        val_rocauc_ls.append(val_metrics['roc_auc'])
        val_prcauc_ls.append(val_metrics['prc_auc'])
        val_f1_ls.append(val_metrics['f1score'])
        '''
        if val_loss <= best_loss:
            best_loss = val_loss
            torch.save({'epoch':epoch, 
                        'model_state_dict':model.state_dict(), 
                        'optimizer_state_dict':optimizer.state_dict(),
                        'loss':val_loss, 
                        'metrics':val_metrics}, 
                       'models/model.ckpt')
        '''
        print(f"[VALID] BCE loss: {val_loss:0.4f} | Acc: {100*val_metrics['accuracy']:0.2f}%") if verbose else None
        print() if verbose else None
        if not verbose and (epoch % int(epochs*.25) == 0):
            print(f"Epoch {epoch}/{epochs}")

    print("Finished Training.")
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    torch.save({'epoch':epoch, 
                'model_state_dict':model.state_dict(), 
                'optimizer_state_dict':optimizer.state_dict(),
                'loss':val_loss, 
                'metrics':val_metrics}, 
                cfg['save_dir']+'/model.ckpt')
    checkpoint = torch.load(cfg['save_dir']+'/model.ckpt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Best Validation Epoch: {checkpoint['epoch']},\n\tLoss: {checkpoint['loss']:.4f},\n\tAccuracy: {checkpoint['metrics']['accuracy']*100:.2f}")
    tn, fp, fn, tp = checkpoint['metrics']['conf_mat'].ravel()
    print("Confusion Matrix:\n\t\tActual Healthy  |  Actual Unhealthy")
    print(f"Pred Healthy   | \t{tp}\t|\t{fp}")
    print(f"Pred Unhealthy | \t{fn}\t|\t{tn}")

    # Metrics excel file
    metrics_df = pd.DataFrame({'Epoch':range(0,epochs), 
                               'Training Loss':train_loss_ls, 'Validation Loss':val_loss_ls,
                               'Training Accuracy':train_acc_ls, 'Validation Accuracy':val_acc_ls,
                               'Training Precision':train_prec_ls, 'Validation Precision':val_prec_ls,
                               'Training Sensitivity':train_sens_ls, 'Validation Sensitivity':val_sens_ls,
                               'Training Specificity':train_spec_ls, 'Validation Specificity':val_spec_ls,
                               'Training F1 Score':train_f1_ls, 'Validation F1 Score':val_f1_ls,
                               'Training ROCAUC':train_rocauc_ls, 'Validation ROCAUC':val_rocauc_ls,
                               'Training PRCAUC':train_prcauc_ls, 'Validation PRCAUC':val_prcauc_ls,
                              })
    metrics_df.to_csv(cfg['save_dir']+'/metrics.csv')



if __name__=='__main__':
    import argparse
    import sys
    
    parser = argparse.ArgumentParser('Train Achilles Tendon classifier')
    parser.add_argument('--config', type=str, required=False, default=None,
                        help='YAML configuration file.')
    parser.add_argument('--num-heads', type=int, required=False, default=None,
                        help='Number of attention heads for MedVidNet model. Overrides "net_kwargs: num_heads" in config if provided.')
    parser.add_argument('--pooling-method', type=str, required=False, default=None,
                        choices=['attn', 'tanh_attn', 'max', 'avg'],
                        help='Pooling method. Overrides "vidnet_kwargs: pooling_method" in config if provided.')
    parser.add_argument('--device', type=str, required=False, default=None,
                        help='Compute devie. Overrides "device" in config if provided')
    args = parser.parse_args()
    
    # load config
    if args.config != None:
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
    else:
        with open(snakemake.input.config, 'r') as f:
            cfg = yaml.safe_load(f)
    
        
        
    print("Running training script with configuration:")
    print('-'*30)
    yaml.dump(cfg, sys.stdout)
    print('-'*30)

    main(cfg)


