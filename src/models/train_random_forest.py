"""Random Forest - ensemble of decision trees (majority vote).

Run:  python -m src.models.train_random_forest [--task multiclass|binary]
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from src.config import SEED
from src.preprocessing.normalize_features import build_preprocessor
from src.evaluation.cross_validate import run_experiment, parse_task

NAME = "Random Forest"


def get_model():
    pipe = Pipeline([
        ("prep", build_preprocessor(scale=False)),
        ("clf", RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1)),
    ])
    grid = {
        "clf__n_estimators": [100, 300, 500],
        "clf__max_depth": [3, 5, 8, None],
        "clf__min_samples_leaf": [1, 3, 5, 10],
    }
    return pipe, grid


if __name__ == "__main__":
    pipe, grid = get_model()
    run_experiment(NAME, pipe, grid, parse_task())
