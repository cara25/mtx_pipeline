#!/bin/bash

# This is to run different combinations of ML
# Assumes data has already been split on seed=42 (does not require changing seeds)

# Define arrays for each parameter
feat_sel=("L1" "L2" "none")
data_combo=("lin_prot_demo_met" "log2_prot_demo_met" "lin_prot_met" "log2_prot_met" "lin_prot_demo" "log2_prot_demo" "demo_met" "linear" "log2" "metabolomics" "demographics")
splits=(5 10 60)

# Define the range of seeds
start_seed=42
end_seed=43

# Loop through each seed
for seed in $(seq $start_seed $end_seed); do
  # Create a directory for each seed
  mkdir -p "seed_$seed"
  # Loop through each combination of parameters
  for fs in "${feat_sel[@]}"; do
    for dc in "${data_combo[@]}"; do
      for sp in "${splits[@]}"; do
      
      # STEP 1: FEATURE SELECTION
      # ignores if "$fs" is "none"--no statement needed for that case
        if [ "$fs" == "L1" ] || [ "$fs" == "L2" ]; then
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