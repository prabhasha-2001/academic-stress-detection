"""Step 4 - normalisation and the shared preprocessing transformer.

"""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from src.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES
from src.preprocessing.encode_features import make_onehot


def build_preprocessor(scale: bool = True) -> ColumnTransformer:
    numeric = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", make_onehot(), CATEGORICAL_FEATURES),
    ])


if __name__ == "__main__":
    import pandas as pd
    from src.config import DATA_LABELED
    from src.preprocessing.encode_features import build_feature_frame
    X, _ = build_feature_frame(pd.read_csv(DATA_LABELED))
    Z = build_preprocessor(scale=True).fit_transform(X)
    print("Transformed shape:", Z.shape)
