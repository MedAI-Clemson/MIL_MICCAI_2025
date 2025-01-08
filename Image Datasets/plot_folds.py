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

    return inter,lower,upper

parser = argparse.ArgumentParser('Plot folds of classifier')
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

        
for i, metric in enumerate(metrics):
    # Training Curves
    fig = plt.figure(figsize=(4,12))
    rows, cols = 8,1
    row = 1

    metrics_df = pd.read_csv(metric)

    # Loss Curve
    ax1 = fig.add_subplot(rows,cols,row)
    ax1.plot(metrics_df['Training Loss'])
    ax1.plot(metrics_df['Validation Loss'])
    ax1.set_title('Loss')
    row+=1
    # Accuracy Curve
    ax2 = fig.add_subplot(rows,cols,row)
    ax2.plot(metrics_df['Training Accuracy'])
    ax2.plot(metrics_df['Validation Accuracy'])
    ax2.set_title('Accuracy')
    ax2.set_ylim([0,1.2])
    row+=1
    # Precision Curve
    ax3 = fig.add_subplot(rows,cols,row)
    ax3.plot(metrics_df['Training Precision'])
    ax3.plot(metrics_df['Validation Precision'])
    ax3.set_title('Precision')
    ax3.set_ylim([0,1.2])
    row+=1
    # Sensitivity Curve
    ax4 = fig.add_subplot(rows,cols,row)
    ax4.plot(metrics_df['Training Sensitivity'])
    ax4.plot(metrics_df['Validation Sensitivity'])
    ax4.set_title('Sensitivity')
    ax4.set_ylim([0,1.2])
    row+=1
    '''
    # Specificity Curve
    ax5 = fig.add_subplot(rows,cols,row)
    ax5.plot(metrics_df['Training Specificity'])
    ax5.plot(metrics_df['Validation Specificity'])
    ax5.set_title('Specificity')
    ax5.set_ylim([0,1.2])
    row+=1
    '''
    # F1 Score Curve
    ax6 = fig.add_subplot(rows,cols,row)
    ax6.plot(metrics_df['Training F1 Score'])
    ax6.plot(metrics_df['Validation F1 Score'])
    ax6.set_title('F1 Score')
    ax6.set_ylim([0,1.2])
    row+=1
    # ROC_AUC 
    ax7 = fig.add_subplot(rows,cols,row)
    ax7.plot(metrics_df['Training ROCAUC'])
    ax7.plot(metrics_df['Validation ROCAUC'])
    ax7.set_title('ROC AUC')
    ax7.set_ylim([0,1.2])
    row+=1
    plt.subplots_adjust(hspace=0.5)
    plt.savefig(cfg['save_dir']+f'/metrics_fold{i+1}.png')