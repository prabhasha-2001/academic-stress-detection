
# Detecting Academic Stress Levels of University Students Using Digital Behavior and Machine Learning

## Overview

This research investigates whether simple, self-reported digital behavior and academic workload indicators, collected through a low-cost Google Forms survey, can be used to predict the academic stress level of Sri Lankan university students.

Three classical machine learning algorithms are compared against a majority-class reference:

- Logistic Regression (baseline)
- Decision Tree
- Random Forest

Stress labels are generated from the **Perceived Stress Scale (PSS-10)**. The PSS-10 items are used only to create the label and are never used as model inputs.

---

# Research Question

Can Logistic Regression, Decision Tree and Random Forest classifiers predict the PSS-10 stress level (Low / Moderate / High) of Sri Lankan undergraduates from self-reported digital behavior and academic workload better than a trivial majority-class baseline?

The question is falsifiable: if the models cannot beat the baseline or each other with statistical significance, the hypothesis is not supported.

---

# Research Objective

To determine whether inexpensive survey-based behavioral indicators are enough to classify academic stress levels, and to provide a benchmark for heavier approaches such as passive sensing and deep learning.

---

# Dataset

The data was collected with an anonymous **Google Forms** survey (convenience and snowball sampling, July 2026).

| Item | Value |
|---|---|
| Valid responses | 478 undergraduates |
| Institutions | 22 (44.8% from Horizon Campus) |
| Missing values / duplicates | None after cleaning |
| PSS-10 total | Range 10–39, mean 22.9, SD 4.1 |
| PSS-10 Cronbach's alpha | 0.49 (low; reported as a limitation) |

The respondent-level data is **not published** in this repository to protect participant privacy.

### Features (15 predictors)

- Age, gender, academic year, faculty group, Horizon Campus indicator
- Daily screen time
- Daily study time
- Sleep duration
- Device use before sleep
- Phone checking while studying
- Weekly assignments
- Perceived academic workload
- Exam pressure
- Academic satisfaction
- Exercise frequency
- Social activities

> Mood-swing frequency and social-media usage frequency were planned but not collected in the final questionnaire. Phone checking, device use before sleep, exam pressure and academic satisfaction were used instead.

### Target Variable

Stress level, derived from the PSS-10 total (0–40):

| Class | PSS-10 score | Respondents |
|---|---|---|
| Low | 0–13 | 7 (1.5%) |
| Moderate | 14–26 | 395 (82.6%) |
| High | 27–40 | 76 (15.9%) |

The classes are **severely imbalanced**. A model that always predicts "Moderate" already achieves about 82.7% accuracy, so accuracy alone is not a reliable measure here.

---

# Methodology

1. Clean the survey export (duplicates, incomplete PSS-10 responses, straight-lining).
2. Compute the PSS-10 total and the 3-class label (items were already reverse-coded in the export).
3. Encode features (ordered answers to integers, gender and faculty one-hot) and standardize numeric features for Logistic Regression.
4. Split the data 80/20 (stratified); the test set is used once.
5. Tune each model with grid search and stratified 10-fold cross-validation on the training set, optimizing macro-F1, with class-balanced weights.
6. Evaluate on the held-out test set and compare models with statistical tests.
7. Run a supplementary **High vs Not-High** binary analysis, because the Low class has only 7 respondents.

All preprocessing is fitted inside each training fold to avoid data leakage.

---

# Machine Learning Models

## Logistic Regression

Baseline model. Simple, interpretable, L2-regularized.

## Decision Tree

Produces explainable decision rules, which is useful for student counsellors. Depth and leaf size are limited to reduce overfitting.

## Random Forest

An ensemble of decision trees that reduces overfitting and handles noisy self-reported data.

## Majority-Class Reference

Always predicts the most frequent class. Any useful model must do better than this.

---

# Model Evaluation

Metrics:

- Accuracy
- Precision, Recall and F1 (per class)
- Macro F1 and Weighted F1
- Matthews correlation coefficient
- ROC-AUC
- Confusion Matrix

Validation strategy:

- Stratified 80/20 train–test split
- Stratified 10-fold cross-validation on the training set
- Paired t-test, Nadeau–Bengio corrected resampled t-test and Dietterich's 5×2cv test (α = 0.05)

---

# Key Findings

- **No model beat the majority-class baseline in accuracy.** Cross-validated accuracy: Random Forest 0.79, Logistic Regression 0.54, Decision Tree 0.52, majority class 0.83.
- Random Forest reached the highest cross-validated macro-F1 (0.44), but differences between the three models were **not statistically significant**.
- Random Forest's accuracy came from predicting "Moderate" almost always. It detected only 3 of 15 High-stress test students, whereas Logistic Regression detected 11 of 15 at low precision (0.26).
- In the High vs Not-High task, cross-validated ROC-AUC was close to chance (0.50–0.54).
- Only **phone checking while studying** (ρ = 0.17) and **exam pressure** (ρ = 0.11) showed weak associations with stress. Screen time and sleep duration showed almost no association.
- In this dataset, self-reported digital behavior does not support reliable stress-level classification.

---

# Limitations

- Low PSS-10 internal consistency (α = 0.49) suggests noisy labels.
- Severe class imbalance; only 7 Low and 76 High respondents, and 15 High cases in the test set.
- Convenience sample: mostly female (64%) and 45% from one campus.
- All predictors are self-reported and coarsely banded.
- Hyperparameters and model comparisons used the same training folds, which can be optimistic.

---

# Future Work

- Collect a larger and more balanced sample.
- Verify PSS-10 data quality and coding.
- Try ordinal or regression targets instead of coarse bins.
- Add objective usage logs where ethically feasible.
- Use nested cross-validation.

---

# Project Structure

```
academic-stress-detection/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_eda.ipynb
│   └── 03_model_prototyping.ipynb
│
├── src/
│   ├── config.py
│   ├── preprocessing/
│   │   ├── clean_data.py
│   │   ├── encode_features.py
│   │   ├── compute_pss10_label.py
│   │   └── normalize_features.py
│   │
│   ├── models/
│   │   ├── train_logistic_regression.py
│   │   ├── train_decision_tree.py
│   │   ├── train_random_forest.py
│   │   └── train_majority_baseline.py
│   │
│   └── evaluation/
│       ├── cross_validate.py
│       ├── compute_metrics.py
│       └── significance_test.py
│
├── diagrams/
│   └── system_architecture.mmd
│
└── results/
    ├── figures/
    └── metrics/
```

---

# Technologies Used

- Python 3.10+
- Scikit-learn
- Pandas
- NumPy
- SciPy
- Matplotlib
- Seaborn
- Jupyter Notebook

---

# Current Progress

✔ Literature Review Completed

✔ Research Methodology Completed

✔ Google Form Designed

✔ Dataset Collection Completed (478 responses)

✔ Repository Structure Created

✔ Data Preprocessing Scripts Completed

✔ Model Training and Evaluation Completed

✔ Statistical Significance Testing Completed

- Research Paper (IEEE format) in preparation

---

# Installation

Clone the repository

```bash
git clone https://github.com/prabhasha-2001/academic-stress-detection.git
```

Move into the project directory

```bash
cd academic-stress-detection
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Authors

### M.T. Prabhasha Dilshani

mtprabhashadilshani@gmail.com

Faculty of Information Technology

Horizon Campus

---

### M.U. Sandamali Wanasinghe

Faculty of Information Technology

Horizon Campus