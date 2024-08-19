## ML for K-fold data ##
## to be run in the shell script "run_ml.sh" ##
## define the main way to predict on kfold data and report statistic averages ##
def main(feat_sel, data_combo, splits, model_str, seed, output_txt):
    print(f"Seed: {seed}, Feature Selection: {feat_sel}, Datasets: {data_combo}, Splits: {splits}, Model Name: {model_str}")
    # map classifier name to the model itself
    classifiers = {
    "logistic": LogisticRegression(),
    "lin_svc": LinearSVC(random_state=seed, dual="auto"),
    "svc": SVC(random_state=seed),
    "sgd_logistic": SGDClassifier(random_state=seed, loss="log_loss"),
    "rf": RandomForestClassifier(random_state=seed),
    "ada": AdaBoostClassifier(random_state=seed, algorithm='SAMME'),
    "xgb": xgb.XGBClassifier(objective='binary:logistic', random_state=seed, use_label_encoder=False),
    "cat": CatBoostClassifier(random_state=seed, verbose=0),
    "vote_hard": VotingClassifier(estimators=[("svc", SVC(random_state=seed)), ("logistic", LogisticRegression()), ("rf", RandomForestClassifier(random_state=seed))], voting='hard'),
        "vote_soft": VotingClassifier(estimators=[("svc", SVC(random_state=seed, probability=True)), ("logistic", LogisticRegression()), ("rf", RandomForestClassifier(random_state=seed))], voting='soft')}
    model_name = classifiers[model_str]
    if feat_sel != 'none':
        cols_to_drop = ['mtx_binary'] # for the X_train when features have been selected already (no original index or id column)
    else:
        cols_to_drop = ['mtx_binary', 'Unnamed: 0', 'EAC_ID'] # for the X_train where features were not selected (has original index and eac id column)
    dir_paths = path_from_inputs(splits=splits, data_combo=data_combo, seed=seed, feat_sel=feat_sel)

    # lists to store each test prediction and each actual prediction (y_test)
    y_preds = []
    y_trues = []

    for i in range(1, splits + 1):
        X_train, X_test, y_train, y_test = split_from_path(dir_paths[i-1], drop_cols=cols_to_drop)
        y_test = y_test.iloc[0] # distinct to LOOCV, extract the true value itself and append to a separate array
        model = model_name
        model.fit(X_train, y_train)
        # predict on test
        y_pred = model.predict(X_test)
        y_preds.append(y_pred[0])
        y_trues.append(y_test)
        # turn predictions into array, stick to naming conventions of other scripts
        y_pred = np.array(y_preds)
        y_pred = np.transpose(y_pred)
        # turn true values into array
        y_test = np.array(y_trues)
        y_test = np.transpose(y_test)
        # calculate metrics
        # not all of these are getting reported in this case for feature selection, but I wanted to have my version of this code implemented in a model
    try: # this code is adapted from hurben RA_ACPA_multiomics/src/machine_learning/classification_5fold.2class.opti.withMCC.py
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        acc = (tp + tn) / (tp + fp + tn + fn)
        prec = tp / (tp + fp)
        tpr = tp / (tp + fn)
        tnr = tn / (tn + fp)
        fpr = fp / (fp + tn)
        fnr = fn / (fn + tp)
        if (tn + fn) != 0:
            npv = tn / (tn + fn)
        else:
            npv = None
    except ZeroDivisionError:
        print("Error ACC: %s %s %s %s %s" % (tp, tn, fp, tn, fn))
        
    # write it all out into separate folder
    output_txt.write(f'{splits},{data_combo},{feat_sel},{model_str},{seed},')
    output_txt.write(f'{acc},')
    output_txt.write(f'{prec},')
    output_txt.write(f'{tpr},')
    output_txt.write(f'{tnr},')
    output_txt.write(f'{fpr},')
    output_txt.write(f'{fnr},')

    try:
        output_txt.write(f'{npv},')
    except TypeError:
        output_txt.write("N/A")
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
    if len(sys.argv) != 4:
        print("Usage: python loocv_ml.py <feat_sel> <data_combo> <splits>")
        sys.exit(1)

    # Parse command-line arguments
    feat_sel = sys.argv[1]
    data_combo = sys.argv[2]
    seed = int(sys.argv[3])
    
    splits = 60 # constant for loocv

    # create output directory for all the folders
    output_dir = f'results/seed_{seed}'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'loocv_{feat_sel}_{data_combo}_results.csv') # make a new results file
    output_txt = open(output_file,'w') # open it and clear any existing results
    output_txt.write( # put the headers in for the eventual file
        'Fold,'
        'Data_combo,'
        'Feature Selection,'
        'Model,'
        'Seed,'
        'ACC,'
        'PREC,'
        'TPR,'
        'TNR,'
        'FPR,'
        'FNR,'
        'NPV\n'
    )
    # define models to try
    # run through every model here line by line
    main(feat_sel, data_combo, 60, "logistic", seed, output_txt)
    main(feat_sel, data_combo, 60, "lin_svc", seed, output_txt)
    main(feat_sel, data_combo, 60, "svc", seed, output_txt)
    main(feat_sel, data_combo, 60, "sgd_logistic", seed, output_txt)
    main(feat_sel, data_combo, 60, "rf", seed, output_txt)
    main(feat_sel, data_combo, 60, "ada", seed, output_txt)
    main(feat_sel, data_combo, 60, "xgb", seed, output_txt)
    main(feat_sel, data_combo, 60, "cat", seed, output_txt)
    output_txt.close()
    end_time = time.time()
    # print total time
    elapsed_time = end_time - start_time
    # Convert elapsed time to minutes and seconds
    elapsed_minutes = int(elapsed_time // 60)
    elapsed_seconds = int(elapsed_time % 60)
    # Print the elapsed time in minutes and seconds
    print(f"Total time taken to run training script: {elapsed_minutes} minutes and {elapsed_seconds} seconds")