"""Evaluation metrics (Table 2 of Milestone 2) and confusion-matrix plot."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, f1_score,
                             confusion_matrix, matthews_corrcoef, roc_auc_score)
from src.config import CLASS_ORDER, FIGURES


def compute_metrics(estimator, X_test, y_test, labels=None, positive=None) -> dict:
    """Accuracy, per-class P/R/F1, macro/weighted F1, MCC, ROC-AUC and confusion matrix.

    labels   : class order for reporting (default Low/Moderate/High)
    positive : positive class name for binary tasks (used for AUC)
    """
    labels = labels or [c for c in CLASS_ORDER if c in set(y_test) or c in estimator.classes_]
    y_pred = estimator.predict(X_test)
    p, r, f, s = precision_recall_fscore_support(y_test, y_pred, labels=labels, zero_division=0)
    out = {
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro", labels=labels, zero_division=0),
        "weighted_f1": f1_score(y_test, y_pred, average="weighted", labels=labels, zero_division=0),
        "mcc": matthews_corrcoef(y_test, y_pred),
        "per_class": {l: {"precision": float(p[i]), "recall": float(r[i]),
                          "f1": float(f[i]), "support": int(s[i])} for i, l in enumerate(labels)},
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels).tolist(),
        "labels": labels,
    }
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X_test)
        classes = list(estimator.classes_)
        try:
            if len(classes) == 2:
                pos = positive or classes[1]
                out["roc_auc"] = float(roc_auc_score(y_test == pos, proba[:, classes.index(pos)]))
            else:
                out["roc_auc"] = float(roc_auc_score(y_test, proba, multi_class="ovr", labels=classes))
        except ValueError:
            out["roc_auc"] = None
    return out


def plot_confusion_matrix(cm, labels, title, filename):
    FIGURES.mkdir(parents=True, exist_ok=True)
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=300)
    plt.close(fig)
