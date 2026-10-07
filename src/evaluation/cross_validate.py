"""Shared experiment protocol used by ALL three classifiers 

Same data, same stratified 80/20 split, same stratified 10-fold CV, same metrics.

Hyper-parameters are tuned by grid search with stratified 10-fold CV on the TRAINING
partition (macro-F1). 
The held-out test partition is used once for the final evaluation.
"""
import argparse
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.config import (DATA_LABELED, RESULTS, METRICS, MODELS_DIR, SEED, TEST_SIZE, N_SPLITS, TARGET)
from src.preprocessing.encode_features import build_feature_frame
from src.evaluation.compute_metrics import compute_metrics, plot_confusion_matrix

# Rare classes (e.g. only a handful of "Low" respondents) trigger a harmless stratification warning
warnings.filterwarnings("ignore", message="The least populated class")

POSITIVE = "High"


def parse_task():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["multiclass", "binary"], default="multiclass",
                    help="multiclass = Low/Moderate/High ; binary = High vs NotHigh (supplementary)")
    return ap.parse_args().task


def load_dataset(task: str = "multiclass"):
    df = pd.read_csv(DATA_LABELED)
    X, y = build_feature_frame(df)
    if task == "binary":
        y = y.where(y == POSITIVE, "NotHigh")
    return X, y


def make_split(X, y):
    """Stratified 80/20 split. Always stratified by the original 3-class label, so the
    multiclass and binary tasks use exactly the same train/test partition."""
    strata = pd.read_csv(DATA_LABELED)[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=strata, random_state=SEED)


def cv_splitter():
    return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)


def tune(pipeline, grid, X_train, y_train):
    if not grid:
        return clone(pipeline).fit(X_train, y_train), {}
    gs = GridSearchCV(pipeline, grid, cv=cv_splitter(), scoring="f1_macro", n_jobs=-1)
    gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_params_


def fold_scores(estimator, X_train, y_train):
    """Per-fold accuracy / macro-F1 (/ AUC for binary) of the tuned configuration."""
    acc, f1, auc = [], [], []
    classes = sorted(set(y_train))
    for tr, va in cv_splitter().split(X_train, y_train):
        m = clone(estimator).fit(X_train.iloc[tr], y_train.iloc[tr])
        pred = m.predict(X_train.iloc[va])
        acc.append(accuracy_score(y_train.iloc[va], pred))
        f1.append(f1_score(y_train.iloc[va], pred, average="macro"))
        if len(classes) == 2 and hasattr(m, "predict_proba"):
            try:
                auc.append(roc_auc_score(y_train.iloc[va] == POSITIVE,
                                         m.predict_proba(X_train.iloc[va])[:, list(m.classes_).index(POSITIVE)]))
            except ValueError:
                auc.append(0.5)
    out = {"accuracy": acc, "macro_f1": f1}
    if auc:
        out["roc_auc"] = auc
    return out


def run_experiment(name: str, pipeline, grid: dict, task: str = "multiclass"):
    X, y = load_dataset(task)
    X_tr, X_te, y_tr, y_te = make_split(X, y)
    est, best = tune(pipeline, grid, X_tr, y_tr)
    folds = fold_scores(est, X_tr, y_tr)
    labels = ["NotHigh", "High"] if task == "binary" else None
    metrics = compute_metrics(est, X_te, y_te, labels=labels, positive=POSITIVE)

    summary = {
        "model": name, "task": task, "best_params": {k: (v if not isinstance(v, np.generic) else v.item()) for k, v in best.items()},
        "n_train": len(X_tr), "n_test": len(X_te),
        "cv_accuracy_mean": float(np.mean(folds["accuracy"])), "cv_accuracy_sd": float(np.std(folds["accuracy"], ddof=1)),
        "cv_macro_f1_mean": float(np.mean(folds["macro_f1"])), "cv_macro_f1_sd": float(np.std(folds["macro_f1"], ddof=1)),
        "test": metrics,
    }
    tag = f"{name.lower().replace(' ', '_')}_{task}"
    METRICS.mkdir(parents=True, exist_ok=True); MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS / f"metrics_{tag}.json").write_text(json.dumps(summary, indent=2, default=float))
    (METRICS / f"fold_scores_{tag}.json").write_text(json.dumps(folds, indent=2))
    joblib.dump(est, MODELS_DIR / f"{tag}.joblib")
    plot_confusion_matrix(metrics["confusion_matrix"], metrics["labels"], name, f"cm_{tag}.png")

    print(f"\n=== {name} ({task}) ===")
    print("Best params:", summary["best_params"])
    print(f"CV accuracy {summary['cv_accuracy_mean']:.3f} +/- {summary['cv_accuracy_sd']:.3f} | "
          f"CV macro-F1 {summary['cv_macro_f1_mean']:.3f} +/- {summary['cv_macro_f1_sd']:.3f}")
    print(f"Test accuracy {metrics['accuracy']:.3f} | macro-F1 {metrics['macro_f1']:.3f} | "
          f"weighted-F1 {metrics['weighted_f1']:.3f} | AUC {metrics.get('roc_auc')}")
    print("Confusion matrix (rows=actual):", metrics["labels"], "\n", np.array(metrics["confusion_matrix"]))
    return est, summary
