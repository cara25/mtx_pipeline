# Predicting Methotrexate Response in Treatment-Naïve Early Arthritis Patients Through the Blood Proteome
**Authors: Cara Chang, Benjamin Hur, John M. Davis III, Jaeyun Sung**

**Currently unpublished**

For a more complete overview of this project, see my summer internship poster, presented at Mayo Clinic to the Summer Undergraduate Research Fellowship and Public Health Internship Group programs.
It has been a pleasure to work as part of the Sung Lab this summer.

**Introduction and Study Design:**
(methods picture)

One of the first treatments that arthritis patients receive after diagnosis is methotrexate (MTX). However, this treatment can be ineffective up to 50% of the time, with non-trivial
side effects, as MTX surpresses autoimmune response. There is currently no available biomarkers that indicate patients' potential MTX response, so this research addresses this question.

Blood samples from _n = 60 patients_ who were diagnosed with early arthritis were used to analyze baseline protein relative abundance through 2,904 Olink® NPX values.
After 3-4 months of MTX treatment, patients were assessed at follow-up using DAS28CRP. The change in DAS28CRP and follow-up levels determined if patients had improved
sufficiently to MTX to count as a responder (1) or non-responder (0). Machine learning was conducted to predict if patients' baseline protein abundances would correctly predict
if patients truly responded to MTX positively. This repo contains source code (src) that generated results through the automated pipeline that follows this general structure.
(pipeline picture)
To test various initializations (seeds) of the data, this code was run on a Linux-based HPC. Datasets are not included for patient privacy, but can be added after forking.
This pipeline was designed to be reusable, so the code is as modularized as possible and has been tested on multiple datasets besides proteomics (e.g. metabolomics).

Directory structure:
<img width="1103" alt="mtx_workflow" src="https://github.com/user-attachments/assets/f4cd236f-6666-496d-9261-e5a5b5ecd210">
