import pickle
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
import os

def create_splits():
    # Print working directory and available files for debugging
 
    # Read input files
    df_img = pd.read_csv(snakemake.input.images)
    df_trg = pd.read_csv(snakemake.input.targets)
    # Create splits
    cv = RepeatedStratifiedKFold(
        n_splits=snakemake.config["num_folds"],
        n_repeats=1,
        random_state=0
    )
    
    splits = list(cv.split(df_trg, df_trg['fracture_type']))


    # Save splits
    with open(snakemake.output.splits, 'wb') as f:
        pickle.dump(splits, f)
    

if __name__ == "__main__":
    create_splits()