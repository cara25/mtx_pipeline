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

## INPUTS SPECIFIED BY USER ##
splits = int(input("number of folds (int): ")) # as an int, number of folds
if splits >= 5 and splits <= 10:
    split_dir = f'{splits}_fold'
else:
    split_dir = 'loocv'
lin_log2 = input("linear or log2 data?: ") # linear or log2

# retrieve paths for specified number of splits and type of data
path_list = path_from_inputs(splits, lin_log2)

# Ensure the output directory exists
# Constructing a directory path using os.path.join() with the string
output_dir = os.path.join(os.getcwd(), split_dir, lin_log2, 'lasso_subsets') # go into the correct fold --> linear or log2 --> lasso dir

# Ensure the directory exists, create if it doesn't
os.makedirs(output_dir, exist_ok=True)

# initialize lists for metrics
acc_list = []
auc_list = []
prec_list = []
tpr_list = []
fpr_list = []
selected_feats_list = []
seed = 42

# Perform on the folds
for i in range(1, splits + 1):
    X_train, X_test, y_train, y_test = split_from_path(path_list[i-1])
    Cs = np.logspace(-1, 1)
    # First, train the lasso with cv 
    # L1 penalty makes it lambda
    model_cv = LogisticRegressionCV(Cs=Cs, cv=10, penalty='l1', solver='liblinear', random_state=seed)
    model_cv.fit(X_train, y_train)
    # Uses the best lambda now
    best_lambda = model_cv.C_[0]
    # Refit lasso model with best lambda
    model = LogisticRegression(penalty='l1', C=best_lambda, solver='liblinear', random_state=seed)
    model.fit(X_train, y_train)
    # predict on test
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    # Identify non-zero coefficients
    non_zero_coefs = model.coef_[0] != 0
    selected_features = X_train.columns[non_zero_coefs]
    # Subset the original DataFrame to include only the selected features
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    
    # calculate metrics
    # not all of these are getting reported in this case for feature selection, but I wanted to have my version of this code implemented in a model
    try: # this code is adapted from hurben RA_ACPA_multiomics/src/machine_learning/classification_5fold.2class.opti.withMCC.py
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        fprs, tprs, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fprs, tprs)
        acc = (tp + tn) / (tp + fp + tn + fn)
        prec = tp / (tp + fp)
        tpr = tp / (tp + fn)
        tnr = tn / (tn + fp)
        fpr = fp / (fp + tn)
        fnr = fn / (fn + tp)
    
        try:
            negative_predictive_value = tn / (tn + fn)
        except ZeroDivisionError:
            negative_predictive_value = "nan"
        
        try:
            num = (tn * tp) - (fn * fp)
            denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
            denom = math.sqrt(denom)
            matthews_corr_coef = num / denom
        except ZeroDivisionError:
            matthews_corr_coef = "nan"
    except ZeroDivisionError:
        print("Error ACC: %s %s %s %s %s" % (tp, tn, fp, tn, fn))
    # append metrics for averages
    acc_list.append(acc)
    auc_list.append(roc_auc)
    prec_list.append(prec)
    tpr_list.append(tpr)
    fpr_list.append(fpr)
    selected_feats_list.append(list(selected_features))
    # Combine subsets X_train and y_train for saving to CSV
    train_selected = pd.concat([X_train_selected, y_train], axis=1)
    test_selected = pd.concat([X_test_selected, y_test], axis=1)
    train_selected.to_csv(os.path.join(output_dir, f'train_{i}.csv'), index=False)
    test_selected.to_csv(os.path.join(output_dir, f'test_{i}.csv'), index=False)
    
    print(f"Split {i} - AUC: {roc_auc}, Accuracy: {acc}, Precision: {prec}, TPR: {tpr}, FPR: {fpr}")

# Report avg metrics
print("Mean AUC: ", np.mean(auc_list))
print("Mean Accuracy: ", np.mean(acc_list))
print("Mean Precision: ", np.mean(prec_list))
print("Mean Recall (TPR): ", np.mean(tpr_list))
print("Mean FPR: ", np.mean(fpr_list))

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
lasso_paths = path_from_inputs(splits, lin_log2, 'L1')
for i in range(splits):
    X_train_selected, X_test, y_train, y_test = split_from_path(lasso_paths[i], drop_cols=[])
    X_train_selected_list.append(X_train_selected)

# Dictionary to store the number of splits each feature is missing from
missing_splits_dict = {feature: [] for feature in feat_dict.keys()}

# Update the occurrence and missing splits for each feature
for i, X_train_selected in enumerate(X_train_selected_list):
    current_features = X_train_selected.columns
    for feature in feat_dict.keys():
        if feature not in current_features:
            missing_splits_dict[feature].append(f'Split {i+1}')

# Create a dataframe to store the result
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
results_df