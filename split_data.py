## SCRIPT TO PROCESS DATA INTO K FOLDS: 5 THROUGH 10 AND LOOCV ##
## INPUT: proteomics_merged.csv ##
## OUTPUTS:
# - full_datasets subdirectory with log2 protein data and linearized protein data
# - split_data subdirectory with kfold and loocv train and test split: "5_fold, 10_fold" and "loocv"
# - these split_data subdirectories each have linear and log2 subdirectories within them with corresponding data
# - the train and test indices for k-fold data are saved in linear subdir
## To ensure kfold data was split correctly, see split_data/kfold_split_validation.py ##

## Packages ##
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split, LeaveOneOut
import os

# paths to load full proteomics data
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
data_path = os.path.join(parent_dir, "proteomics_merged.csv")

data = pd.read_csv(data_path)
data = data.drop(columns=["Unnamed: 0"])

# data.head()

## Part 1: Split full data into Linearized and Log2 Subsets of just protein data ##
linear_protein_cols = [x for x in data.columns.to_list() if 'linear_UniProt' in x] # linearized data
log_protein_cols = [x for x in data.columns.to_list() if 'UniProt' in x and 'linear_UniProt' not in x] # original log2 data

linear_protein_cols.insert(0, "mtx_binary") # include the response variables
log_protein_cols.insert(0, "mtx_binary")

linear_protein_cols.insert(0, "EAC_ID") # include the IDs
log_protein_cols.insert(0, "EAC_ID")

linear_proteins = data[linear_protein_cols]
log_proteins = data[log_protein_cols]

# print(linear_proteins.shape) # should only have the 60 obs and the ~2900 proteins
# print(log_proteins.shape) # should only have the 60 obs and the ~2900 proteins

# Define output directory for these linear or log2 datasets
output_dir = 'full_datasets'
os.makedirs(output_dir, exist_ok=True)

# subset the full datasets themselves log or linear for future use
y = linear_proteins["mtx_binary"]
X = linear_proteins.drop(["mtx_binary"], axis=1)

log_y = log_proteins["mtx_binary"]
log_X = log_proteins.drop(["mtx_binary"], axis=1)

linear_feats = pd.concat([y, X], axis=1)
log_feats  = pd.concat([log_y, log_X], axis=1)

# Save as csv files
# makes the full data set of linearized features (EACID and mtx_binary included, no indices)
linear_feats.to_csv(os.path.join(output_dir, 'linear_proteins.csv'), index=False)
# makes the full data set of original log base 2 features (EACID and mtx_binary included, no indices)
log_feats.to_csv(os.path.join(output_dir, 'log_proteins.csv'), index=False)

## Section 2: KFold Splitting ##

# Base output directory for splitting data
output_dir = "split_data"

# Loop through the range of splits
# can do range(5, 11) for more values of k
for k in [5, 10]:
    # Initialize stratified k-fold cross-validator
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42) # INITIAL STATE = 42
    
    # Create main directory for the current number of splits
    kfold_dir = os.path.join(output_dir, f"{k}_fold")
    os.makedirs(kfold_dir, exist_ok=True)
        
    # Create subdirectories 'linear' and 'log2' for each fold
    linear_dir = os.path.join(kfold_dir, "linear")
    log2_dir = os.path.join(kfold_dir, "log2")
    os.makedirs(linear_dir, exist_ok=True)
    os.makedirs(log2_dir, exist_ok=True)

    
    # Loop through the folds and save data
    for fold, (train_index, test_index) in enumerate(kf.split(X, y), start=1):
        # Split the data
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        log_X_train, log_X_test = log_X.iloc[train_index], log_X.iloc[test_index]
        log_y_train, log_y_test = log_y.iloc[train_index], log_y.iloc[test_index]
        
        # Combine X_train and y_train for saving to CSV
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        
        log_train_df = pd.concat([log_X_train, log_y_train], axis=1)
        log_test_df = pd.concat([log_X_test, log_y_test], axis=1)
        
        # Save indices to text files
        train_indices_file = os.path.join(linear_dir, f'train_indices_fold_{fold}.txt')
        test_indices_file = os.path.join(linear_dir, f'test_indices_fold_{fold}.txt')
        
        np.savetxt(train_indices_file, train_index, fmt='%d')
        np.savetxt(test_indices_file, test_index, fmt='%d')
        
        # Save train and test datasets to CSV files for linear data
        train_df.to_csv(os.path.join(linear_dir, f'train_{fold}.csv'), index=True)
        test_df.to_csv(os.path.join(linear_dir, f'test_{fold}.csv'), index=True)
        
        # Save train and test datasets to CSV files for log2 data
        log_train_df.to_csv(os.path.join(log2_dir, f'train_{fold}.csv'), index=True)
        log_test_df.to_csv(os.path.join(log2_dir, f'test_{fold}.csv'), index=True)

## Section 3: Repeat for LOOCV ##
loo = LeaveOneOut()
loo.get_n_splits(X)

# Create main directory for the current number of splits
loocv_dir = os.path.join(output_dir, "loocv")
os.makedirs(loocv_dir, exist_ok=True)
    
# Create subdirectories 'linear' and 'log2' for each fold
linear_dir = os.path.join(loocv_dir, "linear")
log2_dir = os.path.join(loocv_dir, "log2")
os.makedirs(linear_dir, exist_ok=True)
os.makedirs(log2_dir, exist_ok=True)

for fold, (train_index, test_index) in enumerate(loo.split(X, y), start=1):
        # Split the data
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        log_X_train, log_X_test = log_X.iloc[train_index], log_X.iloc[test_index]
        log_y_train, log_y_test = log_y.iloc[train_index], log_y.iloc[test_index]
        
        # Combine X_train and y_train for saving to CSV
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        
        log_train_df = pd.concat([log_X_train, log_y_train], axis=1)
        log_test_df = pd.concat([log_X_test, log_y_test], axis=1)
        
        # Don't need to save indices-- unseen test obs will be the index of the file name minus one (because of python numbering)
        
        # Save train and test datasets to CSV files for linear data
        train_df.to_csv(os.path.join(linear_dir, f'train_{fold}.csv'), index=True)
        test_df.to_csv(os.path.join(linear_dir, f'test_{fold}.csv'), index=True)
        
        # Save train and test datasets to CSV files for log2 data
        log_train_df.to_csv(os.path.join(log2_dir, f'train_{fold}.csv'), index=True)
        log_test_df.to_csv(os.path.join(log2_dir, f'test_{fold}.csv'), index=True)