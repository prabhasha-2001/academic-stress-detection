"""Logistic Regression - the BASELINE model (L2 regularisation, class-balanced).

Run:  python -m src.models.train_logistic_regression [--task multiclass|binary]
"""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.preprocessing.normalize_features import build_preprocessor
from src.evaluation.cross_validate import run_experiment, parse_task

NAME = "Logistic Regression"


def get_model():
    pipe = Pipeline([
        ("prep", build_preprocessor(scale=True)),          # standardise numeric features
        ("clf", LogisticRegression(class_weight="balanced", max_iter=2000)),  # default penalty = L2
    ])
    grid = {"clf__C": [0.001, 0.01, 0.1, 1, 10, 100]}      # inverse regularisation strength
    return pipe, grid


if __name__ == "__main__":
    pipe, grid = get_model()
    run_experiment(NAME, pipe, grid, parse_task())
