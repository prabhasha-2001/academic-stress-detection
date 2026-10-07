"""Step 2 - compute the PSS-10 total and the 3-class stress label.

Run:  python -m src.preprocessing.compute_pss10_label
"""
import pandas as pd
from src.config import (DATA_CLEAN, DATA_LABELED, PSS_ITEMS, REVERSE_ITEMS,
                        BANDS, CLASS_ORDER, TARGET)


def pss10_total(df: pd.DataFrame, reverse_items=REVERSE_ITEMS) -> pd.Series:
    items = df[PSS_ITEMS].copy()
    for i in reverse_items:                     # reverse-score positively worded items (0-4 scale)
        items[f"pss{i}"] = 4 - items[f"pss{i}"]
    return items.sum(axis=1)


def band(score: float) -> str:
    for lo, hi, name in BANDS:
        if lo < score <= hi:
            return name
    raise ValueError(f"PSS-10 score out of range: {score}")


def cronbach_alpha(items: pd.DataFrame) -> float:
    k = items.shape[1]
    return k / (k - 1) * (1 - items.var(ddof=1).sum() / items.sum(axis=1).var(ddof=1))


def add_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pss_total"] = pss10_total(df)
    df[TARGET] = pd.Categorical(df["pss_total"].map(band), categories=CLASS_ORDER, ordered=True).astype(str)
    return df


def self_test():
    """Hand-calculated cases (the two-person check described in Milestone 2, Section 1.4)."""
    assert band(0) == "Low" and band(13) == "Low"
    assert band(14) == "Moderate" and band(26) == "Moderate"
    assert band(27) == "High" and band(40) == "High"
    demo = pd.DataFrame([[2] * 10], columns=PSS_ITEMS)
    assert pss10_total(demo).iloc[0] == 20 - 0   # no reversal when REVERSE_ITEMS = []
    demo_rev = pss10_total(demo, reverse_items=[4, 5, 7, 8]).iloc[0]
    assert demo_rev == 20                         # 2 -> 4-2 = 2, unchanged
    print("Label self-test passed.")


if __name__ == "__main__":
    self_test()
    df = add_label(pd.read_csv(DATA_CLEAN))
    print(f"PSS-10 Cronbach's alpha: {cronbach_alpha(df[PSS_ITEMS]):.3f}")
    print(df[TARGET].value_counts().reindex(CLASS_ORDER))
    DATA_LABELED.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_LABELED, index=False)
    print(f"Saved {DATA_LABELED}")
