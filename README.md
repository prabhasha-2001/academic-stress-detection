# Detecting Academic Stress Levels of University Students Using Digital Behavior and Machine Learning

## Overview

This project is developed for the **IT41043 – Intelligent Systems** module at **Horizon Campus**.

The objective of this research is to develop a machine learning-based system capable of predicting the academic stress level of university students using digital behavior indicators collected through a Google Forms survey.

The study compares three classical machine learning algorithms:

- Logistic Regression
- Decision Tree
- Random Forest

The models are trained using behavioral and academic features such as screen time, sleep duration, study hours, assignment workload, exercise frequency, mood swings, and social media usage.

Stress labels are generated using the **Perceived Stress Scale (PSS-10)**.

---

# Research Objective

The main objective of this project is to determine whether simple digital behavior indicators can accurately predict academic stress levels among university students.

The project focuses on developing an affordable and interpretable machine learning solution suitable for the Sri Lankan higher education context.

---

# Dataset

The dataset is collected using **Google Forms**.

### Target Participants

- Undergraduate students in Sri Lankan universities
- Approximately 350 responses

### Features

- Age
- Gender
- Year of Study
- Degree Program
- Daily Screen Time
- Sleep Duration
- Assignment Load
- Daily Self Study Hours
- Social Media Usage
- Physical Exercise Frequency
- Mood Swing Frequency

### Target Variable

Stress Level

- Low Stress
- Moderate Stress
- High Stress

The target labels are calculated using the **PSS-10 questionnaire**.

---

# Machine Learning Models

The following supervised learning algorithms are compared.

## Logistic Regression

Used as the baseline model because of its simplicity and interpretability.

## Decision Tree

Provides explainable decision rules and is suitable for educational applications.

## Random Forest

An ensemble learning algorithm capable of improving prediction accuracy while reducing overfitting.

---

# Project Structure

```
academic-stress-detection/
│
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
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
│   ├── preprocessing/
│   │   ├── clean_data.py
│   │   ├── encode_features.py
│   │   ├── compute_pss10_label.py
│   │   └── normalize_features.py
│   │
│   ├── models/
│   │   ├── train_logistic_regression.py
│   │   ├── train_decision_tree.py
│   │   └── train_random_forest.py
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

# Data Preprocessing

The preprocessing pipeline includes:

- Removing duplicate responses
- Handling missing values
- PSS-10 score calculation
- Feature encoding
- Feature normalization
- Dataset preparation for machine learning

---

# Model Evaluation

The models will be evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Macro F1 Score
- Weighted F1 Score
- Confusion Matrix

Validation Strategy

- Stratified 10-Fold Cross Validation
- 80/20 Train-Test Split

---

# Technologies Used

- Python 3
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Jupyter Notebook

---

# Current Progress

✔ Literature Review Completed

✔ Research Methodology Completed

✔ Google Form Designed

✔ Repository Structure Created

✔ Data Preprocessing Scripts Prepared

✔ Machine Learning Pipeline Designed

- Dataset Collection

- Model Training

- Performance Evaluation

- Final Model Selection

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

Student ID: ITBIN-2313-0026

Faculty of Information Technology

Horizon Campus

---

### M.U. Sandamali Wanasinghe

Student ID: ITBIN-2313-0122

Faculty of Information Technology

Horizon Campus

---

# Module

**IT41043 – Intelligent Systems**

Academic Year 2026

Third Year – Second Semester

---

# License

This project is developed for academic and research purposes only.