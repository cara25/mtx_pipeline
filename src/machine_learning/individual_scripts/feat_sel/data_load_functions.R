# THIS IS THE R EQUIVALENT TO PYTHON FUNCTIONS IN data_load_functions.py

## This function will retrieve the paths of X_train, X_test, y_train, y_test in a list for specified split ##
## inputs: split (as int or 'loocv'), 'linear' or 'log', and optional feat_sel that is 'L1', 'L2', or 'enet' ##
## outputs: list of tuples with [('.../train_1.csv', '.../test_1.csv')... ]##
path_from_inputs <- function(splits, lin_log2, feat_sel='none') {
  feat_sel_type <- c('none' = '', 'L1' = 'lasso_subsets', 'L2' = 'ridge_subsets', 'enet' = 'elastic_net_subsets')
  base_dir <- getwd()
  
  if (splits >= 5 && splits <= 10) {
    split_str <- paste0(splits, "_fold")
  } else {
    split_str <- 'loocv'
  }
  
  if (feat_sel != "none") {
    dir_path <- file.path(base_dir, split_str, lin_log2, feat_sel_type[feat_sel])
  } else {
    dir_path <- file.path(base_dir, split_str, lin_log2)
  }
  
  dir_list <- lapply(1:splits, function(i) {
    train_path <- file.path(dir_path, paste0('train_', i, '.csv'))
    test_path <- file.path(dir_path, paste0('test_', i, '.csv'))
    list(train_path = train_path, test_path = test_path)
  })
  
  return(dir_list)
}

## This function will unpack X_train, X_test, y_train, y_test from tuples created in path_from_inputs function ##
## This allows for files to be unpacked one at a time in a loop to be more memory efficient ##
## inputs: tuple of paths to train, test data ('.../train_1.csv', '.../test_1.csv')##
## outputs: X_train, X_test, y_train, y_test ##
split_from_path <- function(tup_path, drop_cols=c('mtx_binary', 'Unnamed: 0', 'EAC_ID'), target='mtx_binary') {
  train_file <- tup_path$train_path
  test_file <- tup_path$test_path
  
  train_df <- read.csv(train_file)
  X_train <- train_df[, !names(train_df) %in% drop_cols]
  y_train <- train_df[[target]]
  
  test_df <- read.csv(test_file)
  X_test <- test_df[, !names(test_df) %in% drop_cols]
  y_test <- test_df[[target]]
  
  return(list(X_train = X_train, X_test = X_test, y_train = y_train, y_test = y_test))
}