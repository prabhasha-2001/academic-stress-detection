"""Majority-class reference (always predicts the most frequent class).

Run:  python -m src.models.train_majority_baseline [--task multiclass|binary]
"""
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from src.preprocessing.normalize_features import build_preprocessor
from src.evaluation.cross_validate import run_experiment, parse_task

NAME = "Majority Class"

if __name__ == "__main__":
    pipe = Pipeline([("prep", build_preprocessor(scale=True)),
                     ("clf", DummyClassifier(strategy="most_frequent"))])
    run_experiment(NAME, pipe, {}, parse_task())
