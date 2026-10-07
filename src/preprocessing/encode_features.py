"""Step 3 - feature encoding.

"""
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from src.config import ORDINAL_MAPS, FEATURES, TARGET, DATA_LABELED, CATEGORICAL_FEATURES


def encode_ordinals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, order in ORDINAL_MAPS.items():
        mapping = {v: i for i, v in enumerate(order)}
        encoded = out[col].map(mapping)
        if encoded.isna().any():
            bad = out.loc[encoded.isna(), col].unique()
            raise ValueError(f"Unexpected category in '{col}': {bad}")
        out[col] = encoded
    out["is_horizon"] = (out["university"] == "Horizon Campus").astype(int)
    return out


def build_feature_frame(df: pd.DataFrame):
    """Return (X, y) with ordinal encoding applied. PSS items are NOT features."""
    enc = encode_ordinals(df)
    return enc[FEATURES], enc[TARGET]


def make_onehot() -> OneHotEncoder:
    return OneHotEncoder(handle_unknown="ignore")


if __name__ == "__main__":
    X, y = build_feature_frame(pd.read_csv(DATA_LABELED))
    print(X.head())
    print(f"Feature matrix: {X.shape}; categorical (one-hot later): {CATEGORICAL_FEATURES}")
