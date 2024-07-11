## Data Validation for split_data.py ##

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
import os
from collections import Counter

# import data the same way as before
# paths to load full proteomics data
base_directory = os.getcwd() # to check against later, want this to be the split directory
parent_dir = os.path.dirname(base_directory)
grandparent_dir = os.path.dirname(parent_dir)
data_path = os.path.join(grandparent_dir, "proteomics_merged.csv")
data = pd.read_csv(data_path)
data = data.drop(columns=["Unnamed: 0"])

# same code as split data to subset data
# subset based on proteomics
linear_protein_cols = [x for x in data.columns.to_list() if 'linear_UniProt' in x] # linearized data
log_protein_cols = [x for x in data.columns.to_list() if 'UniProt' in x and 'linear_UniProt' not in x] # original log2 data

linear_protein_cols.insert(0, "mtx_binary") # include the response variables
log_protein_cols.insert(0, "mtx_binary")

linear_protein_cols.insert(0, "EAC_ID") # include the IDs
log_protein_cols.insert(0, "EAC_ID")

linear_proteins = data[linear_protein_cols]
log_proteins = data[log_protein_cols]

# the data will be checked against the full datasets subset using the split object again
y = linear_proteins["mtx_binary"]
X = linear_proteins.drop(["mtx_binary"], axis=1)
log_y = log_proteins["mtx_binary"]
log_X = log_proteins.drop(["mtx_binary"], axis=1)

# Function to read data from CSV files
def read_data(directory, fold):
    train_file = os.path.join(directory, f'train_{fold}.csv')
    test_file = os.path.join(directory, f'test_{fold}.csv')
    train_df = pd.read_csv(train_file, index_col=0)
    test_df = pd.read_csv(test_file, index_col=0)
    return train_df, test_df

# Iterate over each split (can make range(5,11))
for k in [5, 10]:
    # Initialize stratified k-fold cross-validator
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
    # Main directory for the current number of splits
    main_directory = os.path.join(base_directory, f"{k}_fold")
    
    train_indices = []
    test_indices = []
    # Loop through each fold and read the saved data
    for fold, (train_index, test_index) in enumerate(kf.split(X, y), start=1):
        # Initialize lists to store train and test indices for the current fold
        
        # Original data splits
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        log_X_train, log_X_test = log_X.iloc[train_index], log_X.iloc[test_index]
        log_y_train, log_y_test = log_y.iloc[train_index], log_y.iloc[test_index]
        
        # Read and validate linear data
        linear_directory = os.path.join(main_directory, "linear")
        train_df, test_df = read_data(linear_directory, fold)
        linear_X_train = train_df.drop(columns=['mtx_binary']) # leave in EAC_ID for validation purposes
        linear_y_train = train_df['mtx_binary']
        linear_X_test = test_df.drop(columns=['mtx_binary']) # leave in EAC_ID for validation purposes
        linear_y_test = test_df['mtx_binary']
        
        if not X_train.reset_index(drop=True).equals(linear_X_train.reset_index(drop=True)):
            print(f"Linear X_train mismatch in fold {fold}")
        if not X_test.reset_index(drop=True).equals(linear_X_test.reset_index(drop=True)):
            print(f"Linear X_test mismatch in fold {fold}")
        if not y_train.reset_index(drop=True).equals(linear_y_train.reset_index(drop=True)):
            print(f"Linear y_train mismatch in fold {fold}")
        if not y_test.reset_index(drop=True).equals(linear_y_test.reset_index(drop=True)):
            print(f"Linear y_test mismatch in fold {fold}")
        
        # Read and validate log2 data
        log2_directory = os.path.join(main_directory, "log2")
        train_df, test_df = read_data(log2_directory, fold)
        log2_X_train = train_df.drop(columns=['mtx_binary'])
        log2_y_train = train_df['mtx_binary']
        log2_X_test = test_df.drop(columns=['mtx_binary'])
        log2_y_test = test_df['mtx_binary']
        
        if not log_X_train.reset_index(drop=True).equals(log2_X_train.reset_index(drop=True)):
            print(f"Log2 X_train mismatch in fold {fold}")
        if not log_X_test.reset_index(drop=True).equals(log2_X_test.reset_index(drop=True)):
            print(f"Log2 X_test mismatch in fold {fold}")
        if not log_y_train.reset_index(drop=True).equals(log2_y_train.reset_index(drop=True)):
            print(f"Log2 y_train mismatch in fold {fold}")
        if not log_y_test.reset_index(drop=True).equals(log2_y_test.reset_index(drop=True)):
            print(f"Log2 y_test mismatch in fold {fold}")
        
        # Read index files and check for data leakage
        train_index_file = os.path.join(linear_directory, f'train_indices_fold_{fold}.txt')
        test_index_file = os.path.join(linear_directory, f'test_indices_fold_{fold}.txt')
        
        with open(train_index_file, 'r') as f:
            train_idx = [int(line.strip()) for line in f]
            train_indices.extend(train_idx)
        
        with open(test_index_file, 'r') as f:
            test_idx = [int(line.strip()) for line in f]
            test_indices.extend(test_idx)
        
    # Count the occurrences of each index in train and test sets
    train_counter = Counter(train_indices)
    test_counter = Counter(test_indices)
        
    # Check train set: each index should appear exactly K-1 times for K folds
    train_check = all(count == (k - 1) for count in train_counter.values())
    # Check test set: each index should appear exactly once
    test_check = all(count == 1 for count in test_counter.values())
        
    # Get all unique indices from the combined indices
    all_indices = set(train_indices + test_indices)
        
    # Check if all unique indices in the combined list are the same as in the original data
    original_indices = set(X.index)
        
    train_indices_correct = all_indices == original_indices
    test_indices_correct = all_indices == original_indices
        
    # Print results
    if train_check and train_indices_correct:
        print(f"All observations show up in the training data exactly {k - 1} times when K = {k}.")
    else:
        print(f"Error: Some observations do not show up in the training data exactly {k - 1} times or there are missing indices when K = {k}.")
        
    if test_check and test_indices_correct:
        print(f"All observations show up in the test data exactly once without repeats in when K = {k}.")
    else:
        print(f"Error: Some observations do not show up in the test data exactly once or there are repeats when K = {k}.")