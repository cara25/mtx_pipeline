## NOT CURRENTLY USED ##
## ELASTIC NET SCRIPT ##
library(glmnet)
library(dplyr)
library(caret)
library(stringr)
source("data_load_functions.R") # equivalent to the .py file for path_from_inputs and split_from_path functions

# test the data load functions here...
lin_log2 <- 'linear'
splits <- 5
seed <- 42

path_list <- path_from_inputs(5, 'linear')

main <- function(lin_log2, splits, sel_dir, model_penalty, seed) {
  print(paste("Seed:", seed, "Feature Selection:", 'enet', "Linear/Log:", lin_log2, "Splits:", splits))
  
  # Retrieve paths
  path_list <- path_from_inputs(splits, lin_log2)
  
  # Ensure the output directory exists
  base_dir <- getwd()
  seed_dir <- paste0('seed_', seed)
  output_dir <- file.path(base_dir, seed_dir, lin_log2, 'enet_subsets')
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  counter <- 0
  # Perform on the folds
  for (i in 1:splits) {
    set.seed(seed)
    tup_path <- path_list[[i]]
    data <- split_from_path(tup_path)
    
    X_train <- data$X_train
    X_test <- data$X_test
    y_train <- data$y_train
    y_test <- data$y_test
    
    y_train <- as.factor(y_train)
    y_test <- as.factor(y_test)
    
    # Make the grid for the elastic net
    train_control = trainControl(method = "cv", number = 10)
    elastic_net_model = train(y= y_train, x= X_train, 
                              method = "glmnet",
                              tuneLength = 10,
                              trControl = train_control)
    
    #Summarize output
    coefficients <- coef(elastic_net_model$finalModel,
                         s = elastic_net_model$bestTune$lambda, alpha = elastic_net_model$bestTune$alpha)
    feature_list = rownames(coefficients)
    # subset the data based on elastic net
    non_zero_indices <- which(coefficients != 0, arr.ind = TRUE)
    sel_feats_train <- X_train[, non_zero_indices[, "row"]]
    sel_feats_test <- X_test[, non_zero_indices[, "row"]]
    
    # Save results
    train_selected <- cbind(sel_feats_train, y_train)
    test_selected <- cbind(sel_feats_test, y_test)
    # rename cols and drop index
    colnames(train_selected)[colnames(train_selected) == "y_train"] <- "mtx_binary" # put the target name back
    train_selected <- subset(train_selected, select = -X) # remove index
    colnames(test_selected)[colnames(test_selected) == "y_test"] <- "mtx_binary" # put the target name back
    test_selected <- subset(test_selected, select = -X) # remove index
    
    write.csv(train_selected, file.path(output_dir, paste0('train_', i, '.csv')), row.names = FALSE)
    write.csv(test_selected, file.path(output_dir, paste0('test_', i, '.csv')), row.names = FALSE)
    print("Enet subsets saved out.")
  }
  
  # Initialize lists to store selected features for each split
  X_train_selected_list <- list()
  all_features <- c()
  
  # Read the selected features from CSV files and collect all unique features
  for (i in 1:splits) {
    file_path <- paste0(file.path(output_dir, paste0('train_', i, '.csv')))
    X_train_selected <- read.csv(file_path)
    all_features <- unique(c(all_features, colnames(X_train_selected)))
  }
  
  # Initialize the feature dictionary
  feat_dict <- setNames(rep(0, length(all_features)), all_features)
  
  # Initialize a dictionary to store the number of splits each feature is missing from
  missing_splits_dict <- setNames(vector("list", length(all_features)), all_features)
  
  # Update the occurrence and missing splits for each feature
  for (i in seq_along(X_train_selected_list)) {
    current_features <- colnames(X_train_selected_list[[i]])
    for (feature in names(feat_dict)) {
      if (!(feature %in% current_features)) {
        missing_splits_dict[[feature]] <- c(missing_splits_dict[[feature]], paste('Split', i))
      } else {
        feat_dict[[feature]] <- feat_dict[[feature]] + 1
      }
    }
  }
  
  # Create a dataframe to store the result
  results <- data.frame(
    Feature = names(feat_dict),
    Num_Occurrences = unlist(feat_dict),
    Missing_Splits = sapply(missing_splits_dict, function(x) {
      if (length(x) == 0) return("None")
      return(paste(x, collapse = ', '))
    })
  )
  
  # Sort the dataframe by the number of occurrences
  results <- results %>% arrange(desc(Num_Occurrences))
  
  # Save the results to a CSV file
  write.csv(results, 'enet_subsets/feature_occurrences.csv', row.names = FALSE)}


# Call the main function
main(lin_log2, splits, sel_dir, model_penalty, seed)