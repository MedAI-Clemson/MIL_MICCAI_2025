
#Load package
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from torch.utils.data import DataLoader
import timm
from timm import optim, scheduler
import torch
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import transforms as tfm
from sklearn import metrics as skmet
from jupyterplot import ProgressPlot
import matplotlib.pyplot as plt
import copy
from dataset import ImageData
import pickle
import load_splits
import torch.optim as optim
import os

#Pre-setting
kfold = 10  # number of folds in repeated k-fold
nrepeats = 1  # number of repeats in repeated k-fold
bs_train = 32  # batch size for training
bs_test = 50  # batch size for testing
num_workers = 0  # number of parallel data loading workers
res = 224 # pixel size along height and width
device = torch.device('cuda:0')
num_classes = 3
model = 'resnet50d'
num_epochs=30
lr = 0.001
lr_gamma = 0.92
dropout = 0.3
weight_decay = 0.001
pretrained=True
keep_views = ['AP_LIKE', 'Y']
unfreeze_after_n=8

#Load Dataset
df_img = pd.read_csv('../data/model_images.csv')
df_trg = pd.read_csv('../data/model_targets.csv')

tfms_train = tfm.Compose([
    tfm.Resize(res),
    tfm.CenterCrop(res),
    tfm.RandomEqualize(p=0.5),
    tfm.RandAugment(),
    tfm.ToTensor(),
    tfm.Normalize(mean=torch.Tensor([0.4850, 0.4560, 0.4060]), std=torch.Tensor([0.2290, 0.2240, 0.2250])),
    tfm.RandomHorizontalFlip(),
    tfm.RandomRotation((-45, 45)),
    tfm.RandomInvert(),
    tfm.Grayscale()
])

tfms_test = tfm.Compose([
    tfm.Resize(res),
    tfm.CenterCrop(res),
    tfm.ToTensor(),
    tfm.Normalize(mean=torch.Tensor([0.4850, 0.4560, 0.4060]), std=torch.Tensor([0.2290, 0.2240, 0.2250])),
    tfm.Grayscale()
])

#Train and Evaluate Function
def train_one_epoch(model, train_dataloader, loss_function, device):
    model.train()

    num_steps_per_epoch = len(train_dataloader)

    losses = []
    for ix, batch in enumerate(train_dataloader):
        inputs = batch['img'].to(device)
        targets = batch['trg'].to(device)
        outputs = model(inputs)
        loss = loss_function(outputs, targets)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.detach().item())
        #print(f"\tBatch {ix+1} of {num_steps_per_epoch}. Loss={loss.detach().item():0.3f}", end='\r')
    
    print(' '*100, end='\r')
    print(np.mean(losses))
    return np.mean(losses)
            
            
def evaluate(model, test_dataloader, loss_function, device):
    model.eval()

    num_steps_per_epoch = len(test_dataloader)

    patient_ls = []
    target_ls = []
    output_ls = []
    losses = []
    for ix, batch in enumerate(test_dataloader):
        inputs = batch['img'].to(device)
        targets = batch['trg'].to(device)
        target_ls.append(targets.cpu().numpy())
        patient_ls.append(batch['patient_id'])
        
        with torch.no_grad():
            outputs = model(inputs)
            output_ls.append(outputs.cpu().numpy())
            loss = loss_function(outputs, targets)
            
        losses.append(loss.detach().item())
        
    pids = [pid for ids in patient_ls for pid in ids]
    metrics = compute_metrics(np.concatenate(target_ls), np.concatenate(output_ls))
    metrics_agg = compute_metrics_agg(np.concatenate(target_ls), np.concatenate(output_ls), pids)
    return np.mean(losses), metrics, metrics_agg

def compute_metrics(y_true, y_pred):
    mets = dict()
    
    y_pred = np.exp(y_pred) / np.exp(y_pred).sum(axis=-1, keepdims=True)
    y_pred_cls = np.argmax(y_pred, axis=-1)
    y_true_onehot = np.eye(num_classes, dtype=int)[y_true]
    
    mets['roc_auc'] = skmet.roc_auc_score(y_true_onehot, y_pred, multi_class='ovr')
    mets['average_precision'] = skmet.average_precision_score(y_true_onehot, y_pred)
    mets['accuracy'] = skmet.accuracy_score(y_true, y_pred_cls)
    mets['balanced_accuracy'] = skmet.balanced_accuracy_score(y_true, y_pred_cls)
    
    return mets

def compute_metrics_agg(y_true, y_pred, patient_ids):
    mets = dict()
    
    # aggregate up too the patient level
    df = pd.DataFrame(pd.DataFrame({'patient_id': patient_ids, 'y_true': y_true}).value_counts()).reset_index()
    df.columns = ['patient_id', 'y_true', 'img_count']
    p_pred = np.exp(y_pred) / np.exp(y_pred).sum(axis=-1, keepdims=True)
    df_pred = pd.DataFrame(p_pred, columns = [f'p{ix}' for ix in range(3)])
    df_pred['patient_id'] = patient_ids
    df_pred = df_pred.groupby('patient_id', as_index=False).agg('mean')
    df_pred['norm'] = df_pred['p0'] + df_pred['p1'] + df_pred['p2']
    df_met = df.merge(df_pred, on='patient_id')
    
    y_true = df_met.y_true
    y_true_onehot = np.eye(num_classes, dtype=int)[y_true]
    y_pred = df_met[['p0', 'p1', 'p2']].values
    y_pred_cls = np.argmax(y_pred, axis=-1)
    
    mets['roc_auc'] = skmet.roc_auc_score(y_true_onehot, y_pred, multi_class='ovr')
    mets['average_precision'] = skmet.average_precision_score(y_true_onehot, y_pred)
    mets['accuracy'] = skmet.accuracy_score(y_true, y_pred_cls)
    mets['balanced_accuracy'] = skmet.balanced_accuracy_score(y_true, y_pred_cls)
    
    return mets

#Run the code
with open('cv_splits.pkl', 'rb') as f:
    saved_splits = pickle.load(f)
    

split_results = []
save_dir = '/home/xiaofey/xray/xray-master/code/test6'

for ix, (train_ix, test_ix) in enumerate(saved_splits):
    print('-'*40)
    print('-'*40)
    print(f"Split {ix+1} of {kfold*nrepeats}.")
    
    df_train = df_trg.iloc[train_ix].merge(df_img)
    df_test = df_trg.iloc[test_ix].merge(df_img)
    
    # create datasets
    d_train = ImageData(df_train, transforms = tfms_train, view_filter=keep_views)
    dl_train = DataLoader(d_train, batch_size=bs_train, num_workers=num_workers, shuffle=True)
    
    d_test = ImageData(df_test, transforms = tfms_test, view_filter=keep_views)
    dl_test = DataLoader(d_test, batch_size=bs_test, num_workers=num_workers)
    
    # create model
    m = timm.create_model(model, pretrained=pretrained, num_classes=num_classes, in_chans=1, drop_rate=dropout)
    m.to(device)
    
    # freeze model weights
    # don't freeze classifier or first conv/bn
    for layer in list(m.children())[2:-1]:
        for p in layer.parameters():
            p.requires_grad = False
    is_frozen=True
    
    # fit
    optimizer = optim.Adam(m.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ExponentialLR(optimizer, gamma=lr_gamma)
    loss_function = torch.functional.F.cross_entropy
    
    train_loss_ls = []
    test_loss_ls = []
    metrics_ls = []
    metrics_agg_ls = []
 
    best_test_loss = 1e10
    state_dict = None
    for epoch in range(num_epochs):
        #print("-"*40)
        print(f"\rEpoch {epoch+1} of {num_epochs}:", end='')
        
        # maybe unfreeze 
        if epoch >= unfreeze_after_n and is_frozen:
            print("Unfreezing model encoder.")
            is_frozen=False
            for p in m.parameters():
                p.requires_grad = True
        
        # train for a single epoch
        train_loss = train_one_epoch(m, dl_train, loss_function, device)
        train_loss_ls.append(train_loss)
        #print(f"Training:")
        #print(f"\tcross_entropy = {train_loss:0.3f}")       
    
        # evaluate
        test_loss, metrics, metrics_agg = evaluate(m, dl_test, loss_function, device)
        test_loss_ls.append(test_loss)
        metrics_ls.append(metrics)
        metrics_agg_ls.append(metrics_agg)
        #print(f"Test:")
        #print(f"\tcross_entropy = {test_loss:0.3f}")
        #print(f"\tmetrics:")
        #for k, v in metrics.items():
        #    print(f"\t\t{k} = {v:0.3f}")
        #print(f"\tmetrics_agg:")
        #for k, v in metrics_agg.items():
        #    print(f"\t\t{k} = {v:0.3f}")
            
        # save weights if improved
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            state_dict = copy.deepcopy(m.state_dict())
        
        scheduler.step()
        
    model_save_path = os.path.join(save_dir, f'loss_and_metrics_split_test6_{ix+1}.pth')
    torch.save(state_dict, model_save_path)
        
    split_results.append({
        'split_ix': ix, 
        'best_test_loss': best_test_loss, 
        'state_dict': state_dict,
        'train_ix': train_ix,
        'test_ix': test_ix,
        'train_loss': train_loss_ls, 
        'test_loss': test_loss_ls, 
        'metrics': metrics_ls, 
        'metrics_agg': metrics_agg_ls})
    
# compute average losses
train_losses = np.array([res['train_loss'] for res in split_results])
test_losses = np.array([res['test_loss'] for res in split_results])

# compute average metrics
df_met_ls = [pd.DataFrame(s['metrics']) for s in split_results]
df_met_agg = pd.concat(df_met_ls).reset_index().groupby('index').agg('mean')

# and at the patient level
df_met_agg_ls = [pd.DataFrame(s['metrics_agg']) for s in split_results]
df_met_agg_agg = pd.concat(df_met_agg_ls).reset_index().groupby('index').agg('mean')

#
plt.rcParams['font.size'] = 22

# First loop: Individual split plots
for ix, res in enumerate(split_results):
    fig, ax = plt.subplots(ncols=2, figsize=(20, 10))
    x = np.arange(1, num_epochs + 1)
    
    # Plot train and test loss
    ax[0].plot(x, res['train_loss'], color='k', label='Train Cross Entropy')
    ax[0].plot(x, res['test_loss'], color='b', label='Test Cross Entropy')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    ax[0].set_title(f'Cross-Entropy Loss - Split {ix+1}')
    
    # Plot metrics
    metrics = ['roc_auc', 'average_precision', 'accuracy', 'balanced_accuracy']
    colors = {'roc_auc': 'k', 'average_precision': 'blue', 'accuracy': 'green', 'balanced_accuracy': 'red'}
    for metric in metrics:
        ax[1].plot(x, [m[metric] for m in res['metrics']], color=colors[metric], label=metric)
    
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Metric Value')
    ax[1].legend()
    ax[1].set_title(f'Evaluation Metrics - Split {ix+1}')
    
    plt.tight_layout()
    plt.savefig(f'loss_and_metrics_test6_split_{ix+1}.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# Second plot: Combined results
fig, ax = plt.subplots(ncols=3, figsize=(30, 10))
x = np.arange(1, num_epochs + 1)

# Plot losses
for s in split_results:
    ax[0].plot(x, s['train_loss'], color='k', linestyle='--', alpha=0.4)
    ax[0].plot(x, s['test_loss'], color='b', linestyle='--', alpha=0.4)
ax[0].plot(x, np.mean([s['train_loss'] for s in split_results], axis=0), color='k', label='Avg Train Cross Entropy')
ax[0].plot(x, np.mean([s['test_loss'] for s in split_results], axis=0), color='b', label='Avg Test Cross Entropy')
ax[0].set_title('Cross-Entropy Loss')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Loss')
ax[0].legend()

# Plot image-level metrics
for s in split_results:
    for metric in metrics:
        ax[1].plot(x, [m[metric] for m in s['metrics']], color=colors[metric], linestyle='--', alpha=0.4)
for metric in metrics:
    avg_metric = np.mean([[m[metric] for m in s['metrics']] for s in split_results], axis=0)
    ax[1].plot(x, avg_metric, color=colors[metric], label=f'Avg {metric}')
ax[1].set_title('Image-level Metrics')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Metric Value')
ax[1].set_ylim(0, 1)
ax[1].legend()

# Plot patient-level metrics
for s in split_results:
    for metric in metrics:
        ax[2].plot(x, [m[metric] for m in s['metrics_agg']], color=colors[metric], linestyle='--', alpha=0.4)
for metric in metrics:
    avg_metric = np.mean([[m[metric] for m in s['metrics_agg']] for s in split_results], axis=0)
    ax[2].plot(x, avg_metric, color=colors[metric], label=f'Avg {metric}')
ax[2].set_title('Patient-level Metrics')
ax[2].set_xlabel('Epoch')
ax[2].set_ylabel('Metric Value')
ax[2].set_ylim(0, 1)
ax[2].legend()

for a in ax:
    a.axvline(x=unfreeze_after_n, color='gray', linestyle='--', label='Unfreeze')
    a.grid(True)
    a.legend(loc='lower right')

plt.tight_layout()
plt.savefig('loss_and_metrics_test6.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()