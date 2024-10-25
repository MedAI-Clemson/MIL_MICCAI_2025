import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy import stats
import argparse

def load_scores(model_name):
    with open(f'{model_name}_roc_auc_scores.pkl', 'rb') as f:
        return pickle.load(f)

def compare_models(model1, model2):
    # Load the ROC AUC scores for both models
    scores1 = load_scores(model1)
    scores2 = load_scores(model2)

    # Perform paired t-test
    t_statistic, p_value = stats.ttest_rel(scores2, scores1)

    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Plot ROC AUC scores for each CV split
    x = range(1, len(scores1) + 1)
    plt.plot(x, scores1, 'b-o', label=model1)
    plt.plot(x, scores2, 'r-o', label=model2)
    
    # Customize the plot
    plt.title(f'ROC AUC Scores: {model1} vs {model2}', fontsize=16)
    plt.xlabel('CV Split', fontsize=14)
    plt.ylabel('ROC AUC Score', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add t-test results to the plot
    plt.text(0.05, 0.05, f'Paired t-test:\nt-statistic = {t_statistic:.4f}\np-value = {p_value:.4f}',
             transform=plt.gca().transAxes, fontsize=12, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set y-axis limits with some padding
    plt.ylim(min(min(scores1), min(scores2)) - 0.05, 
             max(max(scores1), max(scores2)) + 0.05)
    
    # Save the figure
    plt.savefig(f'{model1}_vs_{model2}_roc_auc_comparison.png', dpi=300, bbox_inches='tight')
    
    # Show the plot
    plt.show()
    
    # Print the results
    print(f"Paired t-test results:")
    print(f"t-statistic: {t_statistic}")
    print(f"p-value: {p_value}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare ROC AUC scores of two MIL models.')
    parser.add_argument('model1', type=str, help='Name of the first model (e.g., MIL1)')
    parser.add_argument('model2', type=str, help='Name of the second model (e.g., MIL2)')
    args = parser.parse_args()

    compare_models(args.model1, args.model2)