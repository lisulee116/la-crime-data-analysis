# LA Crime Data Analysis & Classification

End-to-end data science project on the City of Los Angeles' public crime dataset — from raw data cleaning through exploratory analysis to a multi-class machine learning model that predicts crime type.

## Overview

| | |
|---|---|
| **Dataset** | [Crime Data from 2020 to Present](https://data.lacity.org/Public-Safety/Crime-Data-from-2020-to-Present/2nrs-mtv8), LAPD / data.lacity.org (open data) |
| **Task** | Data cleaning → EDA → multi-class classification (top 15 crime types) |
| **Tools** | Python, pandas, NumPy, matplotlib/seaborn, scikit-learn, imbalanced-learn (SMOTE) |

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
Missing values, duplicates, date/time parsing, and MO-code mapping, followed by exploratory analysis of crime frequency by area/month and victim demographics.

## Project 2 — Crime Type Classification
Feature selection (variance filter, correlation, chi-square, mutual information) narrowed down predictive features, then three models — Logistic Regression, SVM (RBF), and Random Forest — were tuned via GridSearchCV/RandomizedSearchCV and compared across three class-balancing strategies (original, SMOTE, undersampling). **Random Forest on the original (unbalanced) data performed best overall** (Weighted F1 ≈ 0.54), consistent with its strength on baseline comparisons in Project 1.

## Data
The raw dataset isn't included (too large for git). Download it from [data.lacity.org](https://data.lacity.org/Public-Safety/Crime-Data-from-2020-to-Present/2nrs-mtv8) and save as `project1_data_cleaning_eda/data/CrimeData.csv`, then run the notebooks in order.

## Setup
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```