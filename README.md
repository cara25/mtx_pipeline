# Predicting Methotrexate Response in Treatment-Naïve Early Arthritis Patients Through the Blood Proteome
**Authors: Cara Chang, Benjamin Hur, John M. Davis III, Jaeyun Sung**

Currently unpublished

For a more complete overview of this project, see my summer internship poster `poster.pdf`, presented at Mayo Clinic to the Summer Undergraduate Research Fellowship and Public Health Internship Group programs.
It has been a pleasure to work as part of the Sung Lab this summer.

## Introduction and Study Design

<img width="1223" alt="study_design" src="https://github.com/user-attachments/assets/ebf36b26-816a-4341-baa7-e12852b8587f">

One of the first treatments that arthritis patients receive after diagnosis is methotrexate (MTX). However, this treatment can be ineffective up to 50% of the time ([Aletaha and Smolen 2002](https://pubmed.ncbi.nlm.nih.gov/12180721/)), with non-trivial side effects, as MTX surpresses autoimmune response. 
There is currently no available biomarkers that indicate patients' potential MTX response, so this research addresses this question.

Blood samples from _n = 60 patients_ who were diagnosed with early arthritis were used to analyze baseline protein relative abundance through 2,904 Olink® NPX values.
After 3-4 months of MTX treatment, patients were assessed at follow-up using DAS28CRP. The change in DAS28CRP and follow-up levels determined if patients had improved
sufficiently to MTX to count as a responder (1) or non-responder (0). Machine learning was conducted to predict if patients' baseline protein abundances would correctly predict
if patients truly responded to MTX positively. This repo contains source code (src) that generated results through the automated pipeline that follows this general structure.

<img width="906" alt="updated_pipeline" src="https://github.com/user-attachments/assets/5bec7f11-eccb-46e6-9860-f4f7c5e78df5">

To test various initializations (seeds) of the data, this code was run on a Linux-based HPC. Datasets are not included for patient privacy, but can be added after forking.
This pipeline was designed to be reusable, so the code is as modularized as possible and has been tested on multiple datasets besides proteomics (e.g. metabolomics).

## GitHub Setup

Simplified directory structure:

<img width="1123" alt="Screenshot 2024-08-22 at 11 17 22 AM" src="https://github.com/user-attachments/assets/71437b6b-26e1-4cae-966e-63ae088dfe74">

### Processing and Splitting
To set up the data, put the following into the `mtx_pipeline/raw` folder:
- Olink data (proteomics): `Q-04911_Sung_NPX_2023-02-24.csv`
- Metabolomics: `metabolon_raw_norm_preprocessed.tsv`
- Patient data: `RA_mtx_patient_info.tsv`
- Proteomics bridge (link child and parent samples): `Parent_Child_Bridge_info.csv`
- Metabolomics bridge: `RA_MTX_CLP_GLOBAL_PROFILE_child_parentID_Aug1.xlsx`

Running `mtx_pipeline/src/merging_all_data.ipynb` will preprocess and merge all data into the following 11 combinations of demographics, proteomics, and metabolomics:
- Demographics: `mtx_pipeline/processed/demographics.csv`
- Log2 scale proteomics: `mtx_pipeline/processed/log_proteins.csv`
- **Linear scale proteomics**: `mtx_pipeline/processed/linear_proteins.csv`
- Metabolomics: `mtx_pipeline/processed/metabolomics.csv`
- Log2 scale proteomics + demographics: `mtx_pipeline/processed/log_proteomics_demographics.csv`
- **Linear scale proteomics + demographics**: `mtx_pipeline/processed/lin_proteomics_demographics.csv`
- Demographics + metabolomics: `mtx_pipeline/processed/demographics_metabolomics.csv`
- Log2 scale proteomics + metabolomics: `mtx_pipeline/processed/log_proteomics_metabolomics.csv`
- Linear scale proteomics + metabolomics: `mtx_pipeline/processed/lin_proteomics_metabolomics.csv`
- Log2 proteomics + demographics + metabolomics: `mtx_pipeline/processed/log_proteomics_demographics_metabolomics.csv`
- Linear proteomics + demographics + metabolomics: `mtx_pipeline/processed/lin_proteomics_demographics_metabolomics.csv`

Then, running `mtx_pipeline/src/split_data.sh` will use `split_data.py` to create 5-fold, 10-fold, and loocv splits using random_state=24 of all the datasets in
`5_fold`, `10_fold`, and `loocv`.
The subdirectories of each of these (to access the type of data being trained on) include (for example):
- Demographics: `mtx_pipeline/processed/split_data/5_fold/demographics`
- Log2 scale proteomics: `mtx_pipeline/processed/split_data/5_fold/log2`
- **Linear scale proteomics**: `mtx_pipeline/processed/split_data/5_fold/linear`
- Metabolomics: `mtx_pipeline/processed/split_data/5_fold/metabolomics`
- Log2 scale proteomics + demographics: `mtx_pipeline/processed/split_data/5_fold/log2_prot_demo`
- **Linear scale proteomics + demographics**: `mtx_pipeline/processed/split_data/5_fold/lin_prot_demo`
- Demographics + metabolomics: `mtx_pipeline/processed/split_data/5_fold/demo_metabolomics`
- Log2 scale proteomics + metabolomics: `mtx_pipeline/processed/split_data/5_fold/log2_prot_met`
- Linear scale proteomics + metabolomics: `mtx_pipeline/processed/split_data/5_fold/lin_prot_met`
- Log2 proteomics + demographics + metabolomics: `mtx_pipeline/processed/split_data/5_fold/log2_prot_demo_met`
- Linear proteomics + demographics + metabolomics: `mtx_pipeline/processed/split_data/5_fold/lin_prot_demo_met`
Each subdir has `train_1.csv`, `test_1.csv` up to the max number of folds (5 for 5-fold, 10 for 10-fold, 60 for loocv).

### Machine Learning
Each of the algorithms can be individually run in `mtx_pipeline/src/machine_learning/individual_scripts` to further customize/analyze performance. There is
a subdir for `feat_sel` to explore Lasso, Ridge, and Elastic Net implementation (elastic net did not converge to the initial proteomics and was thus removed from the pipeline).
There is another subdir for `models` for each of the models used in the pipeline. These are not the most up-to-date versions, as I wrote this code before I implemented
the pipeline, so apologies for any incompatibility in data loading/training.

**More importantly, there is the `mtx_pipeline/src/machine_learning/pipeline`** that contains `run_ml.sh` which uses `lasso_ridge.py`, `kfold_ml.py`, and `loocv_ml.py`
to automate combinations of the train/test splits, feature selection, and machine learning over a given number of seeds. Results appear in `mtx_pipeline/results/seed_42` by data combination, and similar dirs will be created when pipeline is implemented going through multiple seeds.
