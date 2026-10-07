"""Statistical significance of differences between models 

Run (after the three model scripts):  python -m src.evaluation.significance_test
"""
import json
import itertools
import numpy as np
from scipy import stats
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import joblib

from src.config import METRICS, MODELS_DIR, N_SPLITS
from src.evaluation.cross_validate import load_dataset, make_split

MODELS = ["logistic_regression", "decision_tree", "random_forest"]


def paired_t(a, b):
    t, p = stats.ttest_rel(a, b)
    return float(t), float(p)


def corrected_resampled_t(a, b, n_splits=N_SPLITS):
    """Nadeau & Bengio (2003): variance inflated by (1/k + n_test/n_train)."""
    d = np.asarray(a) - np.asarray(b)
    k = len(d)
    n_test_over_train = 1.0 / (n_splits - 1)
    var = d.var(ddof=1)
    if var == 0:
        return float("nan"), float("nan")
    t = d.mean() / np.sqrt((1.0 / k + n_test_over_train) * var)
    return float(t), float(2 * stats.t.sf(abs(t), k - 1))


def five_by_two_cv(est_a, est_b, X, y, seed=0):
    """Dietterich (1998) 5x2cv paired t-test on accuracy."""
    variances, first_diff = [], None
    for i in range(5):
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed + i)
        diffs = []
        for tr, va in skf.split(X, y):
            pa = clone(est_a).fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[va])
            pb = clone(est_b).fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[va])
            diffs.append(accuracy_score(y.iloc[va], pa) - accuracy_score(y.iloc[va], pb))
        m = np.mean(diffs)
        variances.append((diffs[0] - m) ** 2 + (diffs[1] - m) ** 2)
        if i == 0:
            first_diff = diffs[0]
    t = first_diff / np.sqrt(np.mean(variances))
    return float(t), float(2 * stats.t.sf(abs(t), 5))


def run(task="multiclass", metric="macro_f1"):
    folds = {m: json.loads((METRICS / f"fold_scores_{m}_{task}.json").read_text()) for m in MODELS}
    X, y = load_dataset(task)
    X_tr, _, y_tr, _ = make_split(X, y)
    rows = []
    for a, b in itertools.combinations(MODELS, 2):
        fa, fb = folds[a][metric], folds[b][metric]
        t1, p1 = paired_t(fa, fb)
        t2, p2 = corrected_resampled_t(fa, fb)
        ea, eb = (joblib.load(MODELS_DIR / f"{m}_{task}.joblib") for m in (a, b))
        t3, p3 = five_by_two_cv(ea, eb, X_tr, y_tr)
        rows.append({"comparison": f"{a} vs {b}", "mean_diff": float(np.mean(fa) - np.mean(fb)),
                     "paired_t_p": p1, "corrected_t_p": p2, "5x2cv_accuracy_p": p3})
        print(f"{a:>20} vs {b:<20} diff={np.mean(fa) - np.mean(fb):+.3f} | paired p={p1:.4f} | "
              f"corrected p={p2:.4f} | 5x2cv(acc) p={p3:.4f}")
    (METRICS / f"significance_{task}_{metric}.json").write_text(json.dumps(rows, indent=2))
    print("alpha = 0.05; report exact p-values, not just 'significant / not significant'.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["multiclass", "binary"], default="multiclass")
    ap.add_argument("--metric", choices=["macro_f1", "accuracy", "roc_auc"], default="macro_f1")
    args = ap.parse_args()
    run(args.task, args.metric)
