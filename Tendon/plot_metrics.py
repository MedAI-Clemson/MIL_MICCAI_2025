import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

def conf_int(values):
    tval = 1.833113 # 9 df / 90% confidence interval
    inter = tval * (np.std(values) / math.sqrt(len(values)))
    lower = np.mean(values) - inter
    upper = np.mean(values) + inter

    return inter,lower,upper


# Training Curves
fig = plt.figure(figsize=(8,4))
rows, cols = 4,2

metrics_df = pd.read_csv(snakemake.input.metrics)

# Loss Curve
ax1 = fig.add_subplot(rows,cols,1)
ax1.plot(metrics_df['Training Loss'])
ax1.plot(metrics_df['Validation Loss'])
ax1.set_title('Loss')
# Accuracy Curve
ax2 = fig.add_subplot(rows,cols,2)
ax2.plot(metrics_df['Training Accuracy'])
ax2.plot(metrics_df['Validation Accuracy'])
ax2.set_title('Accuracy')
ax2.set_ylim([0,1.2])
# Precision Curve
ax2 = fig.add_subplot(rows,cols,3)
ax2.plot(metrics_df['Training Precision'])
ax2.plot(metrics_df['Validation Precision'])
ax2.set_title('Precision')
ax2.set_ylim([0,1.2])
# Sensitivity Curve
ax2 = fig.add_subplot(rows,cols,4)
ax2.plot(metrics_df['Training Sensitivity'])
ax2.plot(metrics_df['Validation Sensitivity'])
ax2.set_title('Sensitivity')
ax2.set_ylim([0,1.2])
# Specificity Curve
ax2 = fig.add_subplot(rows,cols,5)
ax2.plot(metrics_df['Training Specificity'])
ax2.plot(metrics_df['Validation Specificity'])
ax2.set_title('Specificity')
ax2.set_ylim([0,1.2])
# F1 Score Curve
ax3 = fig.add_subplot(rows,cols,6)
ax3.plot(metrics_df['Training F1 Score'])
ax3.plot(metrics_df['Validation F1 Score'])
ax3.set_title('F1 Score')
ax3.set_ylim([0,1.2])
# ROC_AUC 
ax4 = fig.add_subplot(rows,cols,7)
ax4.plot(metrics_df['Training ROCAUC'])
ax4.plot(metrics_df['Validation ROCAUC'])
ax4.set_title('ROC AUC')
ax4.set_ylim([0,1.2])
# PRC_AUC
ax5 = fig.add_subplot(rows,cols,8)
ax5.plot(metrics_df['Training PRCAUC'])
ax5.plot(metrics_df['Validation PRCAUC'])
ax5.set_title('PRC AUC')
ax5.set_ylim([0,1.2])
plt.subplots_adjust(hspace=0.5)
plt.savefig('results/metrics.png')