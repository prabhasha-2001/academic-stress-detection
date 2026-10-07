"""Decision Tree - depth and leaf size are limited by grid search to avoid over-fitting.

Run:  python -m src.models.train_decision_tree [--task multiclass|binary]
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from src.config import SEED
from src.preprocessing.normalize_features import build_preprocessor
from src.evaluation.cross_validate import run_experiment, parse_task

NAME = "Decision Tree"


def get_model():
    pipe = Pipeline([
        ("prep", build_preprocessor(scale=False)),          # trees do not need scaling
        ("clf", DecisionTreeClassifier(class_weight="balanced", random_state=SEED)),
    ])
    grid = {
        "clf__max_depth": [2, 3, 4, 5, 6, 8],
        "clf__min_samples_leaf": [5, 10, 20, 30],
        "clf__criterion": ["gini", "entropy"],
    }
    return pipe, grid


if __name__ == "__main__":
    pipe, grid = get_model()
    run_experiment(NAME, pipe, grid, parse_task())
