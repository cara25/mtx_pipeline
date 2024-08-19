## SCRIPT TO PROCESS DATA INTO K FOLDS: 5 THROUGH 10 AND LOOCV ##
## INPUT: any data that has been properly preprocessed. Only splitting occurs here ##
## OUTPUTS:
# - split_data subdirectory with kfold and loocv train and test split: "5_fold, 10_fold" and "loocv"
# - these split_data subdirectories each have data combo subdirectories within them with corresponding data
# - the train and test indices for k-fold data *can* be saved in linear subdir
## To ensure kfold data was split correctly, see split_data/kfold_split_validation.py (currently outdated) ##

def main(data_combo):
    data_combo_dict = {"lin_prot_demo_met":"lin_proteomics_demographics_metabolomics.csv",
                       "log2_prot_demo_met":"log_proteomics_demographics_metabolomics.csv",
                       "lin_prot_met":"lin_proteomics_metabolomics.csv",
                       "log2_prot_met":"log_proteomics_metabolomics.csv",
                       "lin_prot_demo":"lin_proteomics_demographics.csv",
                       "log2_prot_demo":"log_proteomics_demographics.csv",
                       "demo_met":"demographics_metabolomics.csv",
                       "linear":"linear_proteins.csv",
                       "log2":"log_proteins.csv",
                       "metabolomics":"metabolomics.csv",
                       "demographics":"demographics.csv"}
    
    input_file = data_combo_dict[data_combo]
    
    # paths to load full data
    data_path = os.path.join("..", "..", "processed", "full_data", input_file)
    
    data = pd.read_csv(data_path)
    
    # data.head()
    
    # split into X and y
    y = data["mtx_binary"]
    X = data.drop(["mtx_binary"], axis=1)
    
    
    ## Section 2: KFold Splitting ##
    
    # Base output directory for splitting data
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    grandparent_dir = os.path.dirname(parent_dir)
    output_dir = os.path.join(grandparent_dir, 'processed', 'split_data')
    
    # Loop through the range of splits
    # can do range(5, 11) for more values of k
    for k in [5, 10]:
        # Initialize stratified k-fold cross-validator
        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42) # INITIAL STATE = 42
        
        # Create main directory for the current number of splits
        kfold_dir = os.path.join(output_dir, f"{k}_fold")
        os.makedirs(kfold_dir, exist_ok=True)
            
        # Create subdirectories based on the dataset for each fold
        subdir = os.path.join(kfold_dir, data_combo)
        os.makedirs(subdir, exist_ok=True)
    
        i = 0
        # Loop through the folds and save data
        for fold, (train_index, test_index) in enumerate(kf.split(X, y), start=1):
            # Split the data
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # Combine X_train and y_train for saving to CSV
            train_df = pd.concat([X_train, y_train], axis=1)
            test_df = pd.concat([X_test, y_test], axis=1)

            if i == 0:
            # Save indices to text files just once.
            # All the data across sets should be split the same for number of folds.
                # save in the larger kfold dir because it is not specific to any fold (should be same)
                train_indices_file = os.path.join(kfold_dir, f'train_indices_fold_{fold}.txt')
                test_indices_file = os.path.join(kfold_dir, f'test_indices_fold_{fold}.txt')
            
                np.savetxt(train_indices_file, train_index, fmt='%d')
                np.savetxt(test_indices_file, test_index, fmt='%d')
            i += 1
            # Save train and test datasets to CSV files
            train_df.to_csv(os.path.join(subdir, f'train_{fold}.csv'), index=True)
            test_df.to_csv(os.path.join(subdir, f'test_{fold}.csv'), index=True)
            
    ## Section 3: Repeat for LOOCV ##
    loo = LeaveOneOut()
    loo.get_n_splits(X)
    
    # Create main directory for the current number of splits
    loocv_dir = os.path.join(output_dir, "loocv")
    os.makedirs(loocv_dir, exist_ok=True)
    # Create subdirectory again matching the type of data
    loocv_subdir = os.path.join(loocv_dir, data_combo)
    os.makedirs(loocv_subdir, exist_ok=True)
        
    for fold, (train_index, test_index) in enumerate(loo.split(X, y), start=1):
        # Split the data
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
        # Combine X_train and y_train for saving to CSV
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
            
        # Don't need to save indices-- unseen test obs will be the index of the file name minus one (because of python numbering)
        # Save train and test datasets to CSV files for data
        train_df.to_csv(os.path.join(loocv_subdir, f'train_{fold}.csv'), index=True)
        test_df.to_csv(os.path.join(loocv_subdir, f'test_{fold}.csv'), index=True)

if __name__ == "__main__":
    ## import packages ##
    import sys
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import StratifiedKFold, train_test_split, LeaveOneOut
    import os
    import time
    
    start_time = time.time()
    # Ensure we have the correct number of arguments
    # commenting out because this was being weird before
    # if len(sys.argv) != 1:
    #     print("Usage: python split_data.py <data_combo>")
    #     sys.exit(1)

    # Parse command-line arguments
    data_combo = sys.argv[1]
    
    main(data_combo)
    
    end_time = time.time()
    # print total time
    elapsed_time = end_time - start_time
    # Convert elapsed time to minutes and seconds
    elapsed_minutes = int(elapsed_time // 60)
    elapsed_seconds = int(elapsed_time % 60)
    # Print the elapsed time in minutes and seconds
    print(f"Total time taken to run splitting script: {elapsed_minutes} minutes and {elapsed_seconds} seconds")