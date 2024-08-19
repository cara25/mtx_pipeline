def main(feat_sel, data_combo, splits, sel_dir, model_penalty, seed, threshold=0.005):
    # threshold set somewhat arbitrarily from visualizations in ridge_feature_selection_v2.ipynb.
    if "log" in data_combo:
        threshold = 0.0001
    print(f"Seed: {seed}, Feature Selection: {feat_sel}, Datasets: {data_combo}, Splits: {splits}")
    # retrieve paths for specified number of splits and type of data
    path_list = path_from_inputs(splits=splits, data_combo=data_combo, seed=seed)
    seed_dir = f'seed_{seed}'
    if splits >= 5 and splits <= 10:
        split_dir = f'{splits}_fold'
    else:
        split_dir = 'loocv'
    # Ensure the output directory exists
    # Constructing a directory path using os.path.join() with the string
    output_dir = os.path.join(os.getcwd(), seed_dir, split_dir, data_combo, sel_dir) # go into the correct fold --> linear or log2 --> lasso or ridge dir
    os.makedirs(output_dir, exist_ok=True)
    # initialize to store features picked during each fold training
    selected_feats_list = []
    # Perform on the folds
    for i in range(1, splits + 1):
        X_train, X_test, y_train, y_test = split_from_path(path_list[i-1])
        Cs = np.logspace(-1, 1)
        # First, train with cv
        # L1 penalty makes it lambda
        # L2 is alpha technically but whatever
        model_cv = LogisticRegressionCV(Cs=Cs, cv=10, penalty=model_penalty, solver='liblinear', random_state=seed)
        model_cv.fit(X_train, y_train)
        # Uses the best penalty val now
        best_lambda = model_cv.C_[0]
        # Refit model with best penalty val
        if feat_sel == "L1":
            model = LogisticRegression(penalty=model_penalty, C=best_lambda, solver='liblinear', random_state=seed)
            model.fit(X_train, y_train)
        # FOR RIDGE ONLY:
        # Get the coefficients and zero out those below the threshold
        elif feat_sel == "L2":
            # needs RidgeClassifier specifically, was not working on log data earlier
            model = RidgeClassifier(random_state=seed, alpha=best_lambda)
            model.fit(X_train, y_train)
            coefficients = model.coef_
            coefficients[np.abs(coefficients) < threshold] = 0
            # Assign the non-zero coefficients back to the model
            model.coef_ = coefficients
        # predict on test
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        # Identify non-zero coefficients
        non_zero_coefs = model.coef_[0] != 0
        selected_features = X_train.columns[non_zero_coefs]
        # Subset the original DataFrame to include only the selected features
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]
        # # calculate metrics - not run for pipeline purposes--can be done in separate scripts
        # the reason is because the same metrics will get reported for fs + logistic regression
        selected_feats_list.append(list(selected_features))
        train_selected = pd.concat([X_train_selected, y_train], axis=1)
        test_selected = pd.concat([X_test_selected, y_test], axis=1)
        train_selected.to_csv(os.path.join(output_dir, f'train_{i}.csv'), index=False)
        test_selected.to_csv(os.path.join(output_dir, f'test_{i}.csv'), index=False)
    
    # unpack the list of all the selected features
    flat_feat_sel_list = [
        x
        for xs in selected_feats_list
        for x in xs
    ]
    
    feat_dict = {x:0 for x in flat_feat_sel_list}
    for x in flat_feat_sel_list:
        feat_dict[x] += 1
    
    X_train_selected_list = []
    
    subset_paths = path_from_inputs(splits=splits, data_combo=data_combo, seed=seed, feat_sel=feat_sel)
    for i in range(splits):
        X_train_selected, X_test, y_train, y_test = split_from_path(subset_paths[i], drop_cols=[])
        X_train_selected_list.append(X_train_selected)
    
    # Dictionary to store the number of splits each feature is missing from
    missing_splits_dict = {feature: [] for feature in feat_dict.keys()}
    
    # Update the occurrence and missing splits for each feature
    for i, X_train_selected in enumerate(X_train_selected_list):
        current_features = X_train_selected.columns
        for feature in feat_dict.keys():
            if feature not in current_features:
                missing_splits_dict[feature].append(f'Split {i+1}')
    
    # Create a dataframe to store summaries of feature selections
    results = []
    for feature, num_occurrences in feat_dict.items():
        missing_splits = missing_splits_dict[feature]
        if not missing_splits:
            missing_splits = 'None'
        else:
            missing_splits = ', '.join(missing_splits)
        results.append([feature, num_occurrences, missing_splits])
    results_df = pd.DataFrame(results, columns=['Feature', 'Num_Occurrences', 'Missing_Splits'])
    results_df = results_df.sort_values(by='Num_Occurrences', ascending=False)
    results_df.to_csv(os.path.join(output_dir, 'feature_occurrences.csv'), index=False)

if __name__ == "__main__":
     ## import packages ##
    import sys
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.model_selection import KFold, StratifiedKFold
    from sklearn.linear_model import LogisticRegression, Lasso, LassoCV, LogisticRegressionCV
    import os
    from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score, \
    roc_curve, auc, precision_score, recall_score, confusion_matrix, precision_recall_curve
    from data_load_functions import path_from_inputs, split_from_path # this is user-defined in a .py file
    import math
    import time
    start_time = time.time()
    if len(sys.argv) != 5:
        print("Usage: python lasso_ridge.py <feat_sel> <data_combo> <splits> <seed>")
        sys.exit(1)
    feat_sel = sys.argv[1]
    data_combo = sys.argv[2]
    splits = int(sys.argv[3])
    seed = int(sys.argv[4])
        
    # make the seed dir path
    seed_dir = f'seed_{seed}'
    
    if feat_sel == "L1":
        sel_dir = "lasso_subsets"
        model_penalty = "l1"
    elif feat_sel == "L2":
        sel_dir = "ridge_subsets"
        model_penalty = "l2"

    # Call the main function
    main(feat_sel, data_combo, splits, sel_dir, model_penalty, seed)
    end_time = time.time()
    # print total time
    elapsed_time = end_time - start_time
    # Convert elapsed time to minutes and seconds
    elapsed_minutes = int(elapsed_time // 60)
    elapsed_seconds = int(elapsed_time % 60)
    # Print the elapsed time in minutes and seconds
    print(f"Total time taken to run training script: {elapsed_minutes} minutes and {elapsed_seconds} seconds")
