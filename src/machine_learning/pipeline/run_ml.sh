#!/bin/bash

# This is to run different combinations of ML
# Assumes data has already been split on seed=42 (does not require changing seeds)

# Capture start time
start_time=$(date +%s)

# Define arrays for each parameter
feat_sel=("L1" "L2" "none")
# for data_combo: can include "lin_prot_demo_met" "log2_prot_demo_met" "lin_prot_met" "log2_prot_met" "lin_prot_demo"
# "log2_prot_demo" "demo_met" "linear" "log2" "metabolomics" "demographics"
data_combo=("lin_prot_demo_met" "lin_prot_met" "lin_prot_demo" "linear" "log2")
splits=(5 10 60)

# Define the range of seeds
start_seed=1
end_seed=5

# Specify the output directory mtx_pipeline/src/machine_learning/pipeline --> mtx_pipeline/results
output_base_path="../../../results"
# Loop through each seed
for seed in $(seq $start_seed $end_seed); do
  # Loop through each combination of parameters
  for fs in "${feat_sel[@]}"; do
    for dc in "${data_combo[@]}"; do
      for sp in "${splits[@]}"; do

        # STEP 1: FEATURE SELECTION
        if [ "$fs" != "none" ]; then
          python lasso_ridge.py "$fs" "$dc" "$sp" "$seed"
        fi

        # STEP 2: MACHINE LEARNING
        if [ "$sp" -lt "60" ]; then
          python kfold_ml.py "$fs" "$dc" "$sp" "$seed"
        else
          python loocv_ml.py "$fs" "$dc" "$seed"
        fi

      done
    done
  done
done


# Capture end time
end_time=$(date +%s)

# Calculate duration in seconds
duration=$((end_time - start_time))

# Print the duration in a human-readable format
echo "Total execution time: $(($duration / 60)) minutes and $(($duration % 60)) seconds."