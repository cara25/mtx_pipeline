## Functions to unpack data or load paths ##
import os
import pandas as pd

## This function will retrieve the paths of X_train, X_test, y_train, y_test in a list for specified split ##
## inputs: split (as int or 'loocv'), 'linear' or 'log', and optional feat_sel that is 'L1', 'L2', or 'enet' ##
## outputs: list of tuples with [('.../train_1.csv', '.../test_1.csv')... ]##
def path_from_inputs(splits, data_combo, seed, feat_sel='none'):
    # these are initialized for paths later
    feat_sel_type = {'none': '', 'L1': 'lasso_subsets', 'L2': 'ridge_subsets', 'enet': 'elastic_net_subsets'}
    # go from the pipeline folder to the processed/split_data folder
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    grandparent_dir = os.path.dirname(parent_dir)
    base_dir = os.path.join(grandparent_dir, 'processed', 'split_data')
    if splits >= 5 and splits <= 10:
        split_str = f'{splits}_fold'
    else:
        split_str = 'loocv'
    if feat_sel != "none": # if feature selection method specified, include seed in path
        seed_str = f'seed_{seed}'
        dir_path = os.path.join(base_dir, seed_str, split_str, data_combo, feat_sel_type[feat_sel])
    else:
        # there is no seed in the initial data routing, only comes into play with feature selection
        dir_path = os.path.join(base_dir, split_str, data_combo)
    
    # initialize the list of directories that will get returned
    dir_list = []
    for i in range(1, splits + 1): # retrieve data path name split by split
        train_path = os.path.join(dir_path, f'train_{i}.csv')
        test_path = os.path.join(dir_path, f'test_{i}.csv')
        dir_list.append((train_path, test_path))

    return dir_list

## This function will unpack X_train, X_test, y_train, y_test from tuples created in path_from_inputs function ##
## This allows for files to be unpacked one at a time in a loop to be more memory efficient ##
## inputs: tuple of paths to train, test data ('.../train_1.csv', '.../test_1.csv')##
## outputs: X_train, X_test, y_train, y_test ##
def split_from_path(tup_path, drop_cols=['mtx_binary', 'Unnamed: 0', 'EAC_ID'], target='mtx_binary'):
    train_file = tup_path[0] # train path is first
    test_file = tup_path[1] # then test path
    # Read training data and split into X_train and y_train
    train_df = pd.read_csv(train_file)
    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df[target]
    # Read test data and split into X_test and y_test
    test_df = pd.read_csv(test_file)
    X_test = test_df.drop(columns=drop_cols)
    y_test = test_df[target]
    return X_train, X_test, y_train, y_test


## This function will retrieve the X_train, X_test, y_train, y_test in lists for respective split ##
## inputs: split (as int or 'loocv'), data_combo that is the type of data being tested, possibly in combination
## and optional feat_sel that is 'L1', 'L2', or 'enet' ##
## outputs: X_train_list, X_test_list, y_train_list, y_test_list ##
# Example use: X_train_list, X_test_list, y_train_list, y_test_list = data_from_inputs(5, 'linear') #
def data_from_inputs(splits, data_combo, feat_sel='none'):
    # These are initialized for paths later
    feat_sel_type = {'none': '', 'L1': 'lasso_subsets', 'L2': 'ridge_subsets', 'enet': 'elastic_net_subsets'}
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    grandparent_dir = os.path.dirname(parent_dir)
    base_dir = os.path.join(grandparent_dir, 'processed', 'split_data')
    split_str = f'{splits}_fold'
    
    # Determine the directory path based on inputs
    if feat_sel != 'none':  # If feature selection method specified, include in path
        dir_path = os.path.join(base_dir, split_str, data_combo, feat_sel_type[feat_sel])
    else: # do not include feat_sel in the path name
        dir_path = os.path.join(base_dir, split_str, data_combo)
    
    # Initialize the file lists that will get returned
    X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []
    for i in range(1, splits + 1):  # Retrieve data fold by fold
        train_file = os.path.join(dir_path, f'train_{i}.csv')
        test_file = os.path.join(dir_path, f'test_{i}.csv')
        
        # Read training data and split into X_train and y_train
        train_df = pd.read_csv(train_file)
        X_train = train_df.drop(columns=['mtx_binary', 'Unnamed: 0', 'EAC_ID'])
        y_train = train_df['mtx_binary']
        
        # Read test data and split into X_test and y_test
        test_df = pd.read_csv(test_file)
        X_test = test_df.drop(columns=['mtx_binary', 'Unnamed: 0', 'EAC_ID'])
        y_test = test_df['mtx_binary']
        
        # Append to respective lists
        X_train_list.append(X_train)
        X_test_list.append(X_test)
        y_train_list.append(y_train)
        y_test_list.append(y_test)
    
    return X_train_list, X_test_list, y_train_list, y_test_list

if __name__ == '__main__':
    print ("This is not meant to be run individually")