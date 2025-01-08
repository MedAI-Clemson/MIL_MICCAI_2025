import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import argparse
import yaml
import sys
import os

def conf_int(values):
    tval = 1.833113 # 9 df / 90% confidence interval
    inter = tval * (np.std(values) / math.sqrt(len(values)))
    lower = np.mean(values) - inter
    upper = np.mean(values) + inter

    return np.mean(values),inter,lower,upper

parser = argparse.ArgumentParser('Plot metrics of classifier')
parser.add_argument('--config', type=str, required=False, default=None,
                    help='YAML configuration file.')
args = parser.parse_args()

# load config
if args.config != None:
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    metrics = os.listdir(cfg['save_dir'])
    metrics = [cfg['save_dir']+'/'+file for file in metrics if ('metrics_' in file and 'csv' in file)]
else:
    with open(snakemake.input.config, 'r') as f:
        cfg = yaml.safe_load(f)
    metrics = snakemake.input.metrics

# Training Curves
fig = plt.figure(figsize=(3,12))
rows, cols = 8,1
row = 1

metric_ls = []
last_epochs = []
for metric in metrics:
    df = pd.read_csv(metric)
    metric_ls.append(df)
    last_epochs.append(df.iloc[-1])


# Accuracy Curve
ax = fig.add_subplot(rows,cols,row)
bp = ax.boxplot([epoch['Validation Accuracy'] for epoch in last_epochs])
ax.set_title('Accuracy')
ax.set_ylim([0,1.2])
avg,inter,_,_ = conf_int([epoch['Validation Accuracy'] for epoch in last_epochs])
text = f'90% CI: {avg:.2f}+-{inter:.2f}'
ax.text(0.5,0.1,text)
row+=1
# Precision Curve
ax = fig.add_subplot(rows,cols,row)
bp = ax.boxplot([epoch['Validation Precision'] for epoch in last_epochs])
ax.set_title('Precision')
ax.set_ylim([0,1.2])
avg,inter,_,_ = conf_int([epoch['Validation Precision'] for epoch in last_epochs])
text = f'90% CI: {avg:.2f}+-{inter:.2f}'
ax.text(0.5,0.1,text)
row+=1
# Sensitivity Curve
ax = fig.add_subplot(rows,cols,row)
bp = ax.boxplot([epoch['Validation Sensitivity'] for epoch in last_epochs])
ax.set_title('Sensitivity')
ax.set_ylim([0,1.2])
avg,inter,_,_ = conf_int([epoch['Validation Sensitivity'] for epoch in last_epochs])
text = f'90% CI: {avg:.2f}+-{inter:.2f}'
ax.text(0.5,0.1,text)
row+=1
'''
# Specificity Curve
ax = fig.add_subplot(rows,cols,row)
bp = ax.boxplot([epoch['Validation Specificity'] for epoch in last_epochs])
ax.set_title('Specificity')
ax.set_ylim([0,1.2])
avg,inter,_,_ = conf_int([epoch['Validation Specificity'] for epoch in last_epochs])
text = f'90% CI: {avg:.2f}+-{inter:.2f}'
ax.text(0.5,0.1,text)
row+=1
'''
# F1 Score Curve
ax = fig.add_subplot(rows,cols,row)
bp = ax.boxplot([epoch['Validation F1 Score'] for epoch in last_epochs])
ax.set_title('F1 Score')
ax.set_ylim([0,1.2])
avg,inter,_,_ = conf_int([epoch['Validation F1 Score'] for epoch in last_epochs])
text = f'90% CI: {avg:.2f}+-{inter:.2f}'
ax.text(0.5,0.1,text)
row+=1
# ROC_AUC 
ax = fig.add_subplot(rows,cols,row)
bp = ax.boxplot([epoch['Validation ROCAUC'] for epoch in last_epochs])
ax.set_title('ROC AUC')
ax.set_ylim([0,1.2])
avg,inter,_,_ = conf_int([epoch['Validation ROCAUC'] for epoch in last_epochs])
text = f'90% CI: {avg:.2f}+-{inter:.2f}'
ax.text(0.5,0.1,text)
row+=1
plt.subplots_adjust(hspace=0.5)
plt.savefig(cfg['save_dir']+'/metrics.png')