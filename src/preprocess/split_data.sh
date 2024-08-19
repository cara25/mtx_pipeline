#!/bin/bash

# This is to split the data using different combinations of the data then to be fed to run_ml.sh

# Define arrays for each parameter
data_combo=("lin_prot_demo_met" "log2_prot_demo_met" "lin_prot_met" "log2_prot_met" "lin_prot_demo" "log2_prot_demo" "demo_met" "linear" "log2" "metabolomics" "demographics")

for dc in "${data_combo[@]}"; do
    python split_data.py "$dc"

done