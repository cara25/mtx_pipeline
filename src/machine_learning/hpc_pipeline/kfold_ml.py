## ML for K-fold data ##
## to be run in the shell script "run_ml.sh" ##
## define the main way to predict on kfold data and report statistic averages ##
# Next steps: debug XGBoost and CatBoost (or remove) for cluster or handle with try/excepts.
def main(feat_sel, data_combo, splits, model_str, seed, output_txt):
    print(f"Seed: {seed}, Feature Selection: {feat_sel}, Datasets: {data_combo}, Splits: {splits}, Model Name: {model_str}")
    # map classifier name to the model itself
    classifiers = {
    "logistic": LogisticRegression(),
    "lin_svc": LinearSVC(random_state=seed, dual="auto"),
    "svc": SVC(random_state=seed, probability=True),
    "sgd_logistic": SGDClassifier(random_state=seed, loss="log_loss"),
    "rf": RandomForestClassifier(random_state=seed),
    "ada": AdaBoostClassifier(random_state=seed, algorithm='SAMME'),
    "xgb": xgb.XGBClassifier(objective='binary:logistic', random_state=seed, use_label_encoder=False),
    "cat": CatBoostClassifier(random_state=seed, verbose=0),
    "vote_hard": VotingClassifier(estimators=[("svc", SVC(random_state=seed)), ("logistic", LogisticRegression()), ("rf", RandomForestClassifier(random_state=seed))], voting='hard'),
        "vote_soft": VotingClassifier(estimators=[("svc", SVC(random_state=seed, probability=True)), ("logistic", LogisticRegression()), ("rf", RandomForestClassifier(random_state=seed))], voting='soft')}
    model_name = classifiers[model_str] # gets the right model from the input string given in the function calls
    if feat_sel != 'none':
        cols_to_drop = ['mtx_binary'] # for the X_train when features have been selected already (no original index or id column)
    else:
        cols_to_drop = ['mtx_binary', 'Unnamed: 0', 'EAC_ID'] # for the X_train where features were not selected (has original index and eac id column)
    dir_paths = path_from_inputs(splits=splits, data_combo=data_combo, seed=seed, feat_sel=feat_sel)
 
    # initialize lists to store metrics
    acc_list = []
    prec_list = []
    tpr_list = []
    tnr_list = []
    fpr_list = []
    fnr_list = []
    npv_list = []
    auc_list = []

    for i in range(1, splits + 1):
        X_train, X_test, y_train, y_test = split_from_path(dir_paths[i-1], drop_cols=cols_to_drop)
        model = model_name
        model.fit(X_train, y_train)
        # predict on test
        y_pred = model.predict(X_test)
        
        y_pred_proba = None
        if model_str == "vote_soft":
            # soft voting, y_pred is already the avged probabilities
            y_pred_proba = y_pred
        elif model_str != "vote_hard":
            # handle models without predict_proba
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            except AttributeError:
                try:
                    y_pred_proba = model.decision_function(X_test)
                except AttributeError:
                    pass  # No probability or decision function available
    
        # from confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    
        # calculate ROC AUC if applicable
        roc_auc = None
        if y_pred_proba is not None:
            try:
                fprs, tprs, thresholds = roc_curve(y_test, y_pred_proba)
                roc_auc = auc(fprs, tprs)
                auc_list.append(roc_auc)
            except ValueError:
                pass  # Handle case where ROC AUC cannot be calculated
    
        # Calculate metrics, handling division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            acc = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else np.nan
            prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
            tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            tnr = tn / (tn + fp) if (tn + fp) > 0 else np.nan
            fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
            fnr = fn / (fn + tp) if (fn + tp) > 0 else np.nan
            npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    
        # Append metrics to lists
        acc_list.append(acc)
        prec_list.append(prec)
        tpr_list.append(tpr)
        tnr_list.append(tnr)
        fpr_list.append(fpr)
        fnr_list.append(fnr)
        if not np.isnan(npv):
            npv_list.append(npv)
        
    # write it all out into separate folder
    output_txt.write(f'{splits},{data_combo},{feat_sel},{model_str},{seed},')
    output_txt.write(f'{np.mean(acc_list)},{np.std(acc_list)},')
    output_txt.write(f'{np.mean(prec_list)},{np.std(prec_list)},')
    output_txt.write(f'{np.mean(tpr_list)},{np.std(tpr_list)},')
    output_txt.write(f'{np.mean(tnr_list)},{np.std(tnr_list)},')
    output_txt.write(f'{np.mean(fpr_list)},{np.std(fpr_list)},')
    output_txt.write(f'{np.mean(fnr_list)},{np.std(fnr_list)},')

    try:
        output_txt.write(f'{np.mean(npv_list)},{np.std(npv_list)},')
    except TypeError:
        output_txt.write(f'{npv_list},{npv_list},')

    try:
        output_txt.write(f'{np.mean(auc_list)},{np.std(auc_list)}')
    except TypeError:
        output_txt.write(f'{auc_list},{auc_list}')
    # create a new line
    output_txt.write('\n')
    

if __name__ == "__main__":
    ## import packages ##
    import sys
    import pandas as pd
    import numpy as np
    import os
    from data_load_functions import path_from_inputs, split_from_path # this is user-defined in a .py file
    from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score, \
    roc_curve, auc, precision_score, recall_score, confusion_matrix, precision_recall_curve
    import math
    import statistics
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.svm import LinearSVC, SVC
    import xgboost as xgb
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
    from catboost import CatBoostClassifier
    import time
    start_time = time.time()
    # Ensure we have the correct number of arguments
    if len(sys.argv) != 5:
        print("Usage: python kfold_ml.py <feat_sel> <data_combo> <splits> <seed>")
        sys.exit(1)

    # Parse command-line arguments
    feat_sel = sys.argv[1]
    data_combo = sys.argv[2]
    splits = int(sys.argv[3])
    seed = int(sys.argv[4])

    # create output directory for all the folders
    current_dir = os.getcwd() # pipeline
    parent_dir = os.path.dirname(current_dir) # machine_learning
    grandparent_dir = os.path.dirname(parent_dir) # src
    greatgrandparent_dir = os.path.dirname(grandparent_dir) # base
    output_dir = os.path.join(greatgrandparent_dir, 'results', f'seed_{seed}')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{splits}_fold_{feat_sel}_{data_combo}_results.csv') # make a new results file
    output_txt = open(output_file,'w') # open it and clear any existing results
    output_txt.write( # put the headers in for the eventual file
        'Fold,'
        'Data_combo2,'
        'Feature Selection,'
        'Model,'
        'Seed,'
        'ACC_average,ACC_stdev,'
        'PREC_average,PREC_stdev,'
        'TPR_average,TPR_stdev,'
        'TNR_average,TNR_stdev,'
        'FPR_average,FPR_stdev,'
        'FNR_average,FNR_stdev,'
        'NPV_average,NPV_stdev,'
        'AUC_average,AUC_stdev\n'
    )
    # define models to try
    # run through every model here line by line
    main(feat_sel, data_combo, splits, "logistic", seed, output_txt)
    main(feat_sel, data_combo, splits, "lin_svc", seed, output_txt)
    main(feat_sel, data_combo, splits, "svc", seed, output_txt)
    main(feat_sel, data_combo, splits, "sgd_logistic", seed, output_txt)
    main(feat_sel, data_combo, splits, "rf", seed, output_txt)
    main(feat_sel, data_combo, splits, "ada", seed, output_txt)
    main(feat_sel, data_combo, splits, "xgb", seed, output_txt)
    main(feat_sel, data_combo, splits, "cat", seed, output_txt)
    main(feat_sel, data_combo, splits, "vote_hard", seed, output_txt)
    main(feat_sel, data_combo, splits, "vote_soft", seed, output_txt)
    output_txt.close()
    end_time = time.time()
    # print total time
    elapsed_time = end_time - start_time
    # Convert elapsed time to minutes and seconds
    elapsed_minutes = int(elapsed_time // 60)
    elapsed_seconds = int(elapsed_time % 60)
    # Print the elapsed time in minutes and seconds
    print(f"Total time taken to run training script: {elapsed_minutes} minutes and {elapsed_seconds} seconds")