# Detecting Academic Stress Levels of University Students Using Digital Behavior and Machine Learning

## Overview

This research project investigates whether digital behavior and academic lifestyle indicators can be used to predict the stress levels of university students using machine learning.

The study focuses on developing a machine learning-based prediction system using behavioral and academic features collected through a structured questionnaire.

Three classical machine learning algorithms are compared:

* Logistic Regression
* Decision Tree
* Random Forest

The models use features such as screen time, sleep duration, study hours, assignment workload, exercise frequency, mood swings, and social media usage.

Stress levels are determined using the **Perceived Stress Scale (PSS-10)**.

---

# Research Objective

The main objective of this research is to determine whether simple digital behavior and academic lifestyle indicators can accurately predict stress levels among university students.

The research aims to develop an affordable, interpretable, and data-driven machine learning approach that can support early identification of students experiencing higher levels of academic stress.

---

# Dataset

The dataset is collected using a structured online questionnaire.

### Target Participants

* Undergraduate university students
* Approximately 350 participants

### Features

* Age
* Gender
* Year of Study
* Degree Program
* Daily Screen Time
* Sleep Duration
* Assignment Load
* Daily Self-Study Hours
* Social Media Usage
* Physical Exercise Frequency
* Mood Swing Frequency

### Target Variable

**Stress Level**

* Low Stress
* Moderate Stress
* High Stress

The stress categories are generated based on the **PSS-10 questionnaire score**.

---

# Machine Learning Models

The following supervised learning algorithms are compared.

## Logistic Regression

Logistic Regression is used as a baseline classification model because of its simplicity, interpretability, and suitability for multi-class classification.

## Decision Tree

Decision Tree provides interpretable decision rules and can identify relationships between behavioral factors and stress levels.

## Random Forest

Random Forest is an ensemble learning algorithm that combines multiple decision trees to improve prediction performance and reduce overfitting.

---

# Research Workflow

The research workflow consists of the following stages:

1. Data Collection
2. Data Cleaning
3. PSS-10 Score Calculation
4. Feature Engineering
5. Feature Encoding
6. Feature Normalization
7. Stratified 10-Fold Cross Validation
8. Model Training
9. Model Comparison
10. Performance Evaluation
11. Stress Level Prediction

---

# Project Structure

```text
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

* Removing duplicate responses
* Handling missing values
* Calculating PSS-10 scores
* Generating stress level labels
* Feature encoding
* Feature normalization
* Preparing the dataset for machine learning

---

# Model Evaluation

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Macro F1 Score
* Weighted F1 Score
* Confusion Matrix

### Validation Strategy

* Stratified 10-Fold Cross Validation
* 80/20 Train-Test Split

The performance of the three models will be compared to identify the most suitable model for predicting student stress levels.

---

# Technologies Used

* Python 3
* Scikit-learn
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Jupyter Notebook

---

# Current Progress

* Literature Review Completed
* Research Methodology Completed
* Questionnaire Designed
* Repository Structure Created
* Data Preprocessing Scripts Prepared
* Machine Learning Pipeline Designed

### Remaining Tasks

* Dataset Collection
* Data Preprocessing
* Exploratory Data Analysis
* Model Training
* Cross-Validation
* Performance Evaluation
* Model Comparison
* Final Model Selection

---

# Installation

### Clone the Repository

```bash
git clone https://github.com/prabhasha-2001/academic-stress-detection.git
```

### Move into the Project Directory

```bash
cd academic-stress-detection
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Research Focus

This research focuses on the relationship between students' digital behavior, academic lifestyle, and perceived stress levels.

The study aims to evaluate whether commonly available behavioral indicators can provide useful information for predicting academic stress using interpretable machine learning techniques.

---

# Ethical Considerations

The research data is collected through a structured questionnaire. Participant privacy and confidentiality should be maintained throughout the data collection, processing, and analysis stages.

The machine learning predictions are intended for research purposes and should not be considered a clinical diagnosis of mental health conditions.

---

