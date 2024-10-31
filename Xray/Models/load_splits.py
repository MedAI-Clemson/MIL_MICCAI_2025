import pickle
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

# Set the parameters
kfold = 10
nrepeats = 1
random_seed = 0

# Load datasets
df_img = pd.read_csv('../data/model_images.csv')
df_trg = pd.read_csv('../data/model_targets.csv')

# Create cross-validation splits
cv = RepeatedStratifiedKFold(n_splits=kfold, n_repeats=nrepeats, random_state=random_seed)
splits = list(cv.split(df_trg, df_trg['fracture_type']))

# Save the splits to a file
with open('cv_splits.pkl', 'wb') as f:
    pickle.dump(splits, f)