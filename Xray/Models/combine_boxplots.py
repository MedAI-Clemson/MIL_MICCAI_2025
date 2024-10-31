import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import os

# Function to load ROC AUC scores
def load_scores(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Load scores from each MIL file
scores0 = load_scores('Img_roc_auc_scores.pkl')
scores1 = load_scores('MIL1_roc_auc_scores.pkl')
scores2 = load_scores('MIL2_roc_auc_scores.pkl')
scores3 = load_scores('MIL3_roc_auc_scores.pkl')
scores4 = load_scores('MIL4_roc_auc_scores.pkl')

# Combine all scores into a single list
all_scores = [scores0, scores1, scores2, scores3, scores4]

# Create labels for each model
model_labels = ['Average', 'MIL1', 'MIL2', 'MIL3', 'MIL4']

# Set up the plot
plt.figure(figsize=(12, 8))

# Create the boxplot
box_plot = plt.boxplot(all_scores, labels=model_labels, patch_artist=True)
plt.xticks(fontsize=18)

# Customize colors for each box
colors = ['orange', 'lightblue', 'lightgreen', 'pink', 'lightyellow']
for patch, color in zip(box_plot['boxes'], colors):
    patch.set_facecolor(color)

# Add individual points
for i, scores in enumerate(all_scores):
    x = np.random.normal(i+1, 0.04, size=len(scores))
    plt.scatter(x, scores, alpha=0.5)

# Customize the plot
plt.title('ROC AUC Scores Across MIL Models (10-Fold CV)', fontsize=23)
plt.ylabel('ROC AUC Score', fontsize=20)
plt.ylim(0.4, 1)  # Set y-axis limits from 0 to 1 for AUC scores

# Add mean and std for each model
for i, scores in enumerate(all_scores):
    mean = np.mean(scores)
    std = np.std(scores)
    plt.text(i+1, 0.35, f"Mean: {mean:.4f}\nStd: {std:.4f}", 
             horizontalalignment='center', verticalalignment='top',
             fontsize=18)

# Adjust layout and save
plt.tight_layout()
plt.savefig('combined_mil_roc_auc_boxplot.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

# Print the scores for each model
for label, scores in zip(model_labels, all_scores):
    print(f"\n{label}:")
    mean = np.mean(scores)
    std = np.std(scores)
    print(f"Overall ROC AUC = {mean:.4f} ± {std:.4f}")
    for i, score in enumerate(scores, 1):
        print(f"Fold {i}: ROC AUC = {score:.4f}")