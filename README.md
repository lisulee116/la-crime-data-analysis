# LA Crime Data Analysis & Classification

End-to-end data science project on the City of Los Angeles' public crime dataset — from raw data cleaning through exploratory analysis to a multi-class machine learning model that predicts crime type.

## Overview

| | |
|---|---|
| **Dataset** | [Crime Data from 2020 to Present](https://data.lacity.org/Public-Safety/Crime-Data-from-2020-to-Present/2nrs-mtv8), LAPD / data.lacity.org (open data) |
| **Task** | Data cleaning → EDA → multi-class classification (top 15 crime types) |
| **Tools** | Python, pandas, NumPy, matplotlib/seaborn, scikit-learn, imbalanced-learn (SMOTE) |

## Key results
Three models were tuned (`GridSearchCV`/`RandomizedSearchCV`) and evaluated across three class-balancing strategies:

| Model | Resampling | Accuracy | F1 (macro) | F1 (weighted) | ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest** | **Original** | **0.587** | 0.473 | **0.544** | **0.942** |
| Random Forest | SMOTE | 0.569 | **0.483** | 0.543 | 0.940 |
| SVM (RBF) | SMOTE | 0.517 | 0.446 | 0.500 | 0.929 |
| Logistic Regression | Undersampling | 0.473 | 0.393 | 0.448 | 0.915 |

**Random Forest on the original, unbalanced data performed best overall.** SMOTE improved fairness across rare crime types (higher macro F1) at a small cost to overall accuracy — a real trade-off, not a clear winner either way.


## Repository structure

```
.
├── project1_data_cleaning_eda/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda_visualization.ipynb
│   └── data/
│       └── MoCodes.csv
├── project2_crime_classification/
│   ├── 01_feature_selection_and_baseline_models.ipynb
│   └── 02_hyperparameter_tuning_and_evaluation.ipynb
├── docs/
│   └── MoCodes.pdf
└── requirements.txt
```

## Project 1 — Data Cleaning & EDA

Cleaned ~790K raw LAPD incident records (missing values, duplicates, date/time parsing, MO-code mapping) and explored crime frequency by area, month, and victim demographics.

Four feature-selection methods (variance filter, correlation, chi-square, mutual information) identified premise type, weapon, and victim descriptors as the strongest predictors of crime type — temporal features (hour, day of week) turned out to matter far less than initially expected. Four baseline models were compared; **Random Forest led on every metric**, confirming the underlying relationship is non-linear.

## Project 2 — Crime Type Classification

Built on Project 1's cleaned data to tune and compare Logistic Regression, SVM (RBF), and Random Forest, then evaluated all three against three class-balancing strategies (original / SMOTE / random undersampling) — 9 combinations in total.

**Engineering notes:**
- `liblinear` (originally planned solver) doesn't support multi-class problems with 3+ classes — caught during testing and switched to `saga`.
- SVM training time scales poorly with dataset size; tuning and final fits use a stratified subsample, with the trade-off documented and quantified against the full-data baseline.
- Fixed a non-deterministic `mutual_info_classif` call (no `random_state`) that could silently change feature-importance rankings between runs.

## Data
The raw dataset isn't included (too large for git). Download it from [data.lacity.org](https://data.lacity.org/Public-Safety/Crime-Data-from-2020-to-Present/2nrs-mtv8) and save as `project1_data_cleaning_eda/data/CrimeData.csv`, then run the notebooks in order.

## Setup
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```