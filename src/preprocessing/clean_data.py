"""Step 1 - clean the raw Google Forms export.

Run:  python -m src.preprocessing.clean_data
"""
import pandas as pd
from src.config import DATA_RAW, DATA_CLEAN, PSS_ITEMS


def clean(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    n0 = len(df)
    df = df.drop_duplicates().copy()
    n_dup = n0 - len(df)

    # PSS items must be complete : imputing a psychometric scale would distort the label
    n1 = len(df)
    df = df.dropna(subset=PSS_ITEMS)
    n_incomplete = n1 - len(df)

    # Straight-lining : same answer to all ten PSS items
    n2 = len(df)
    df = df[df[PSS_ITEMS].nunique(axis=1) > 1]
    n_straight = n2 - len(df)

    # Impute remaining missing behavioural values
    for col in df.columns:
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode().iloc[0])

    # Flag (do not delete) values worth a manual look : IQR rule on age
    q1, q3 = df["age"].quantile([0.25, 0.75])
    iqr = q3 - q1
    df["flag_age_outlier"] = ((df["age"] < q1 - 1.5 * iqr) | (df["age"] > q3 + 1.5 * iqr)).astype(int)

    if verbose:
        print(f"Rows in: {n0} | duplicates: {n_dup} | incomplete PSS: {n_incomplete} "
              f"| straight-lined: {n_straight} | rows out: {len(df)}")
        print(f"Age outliers flagged for review: {int(df['flag_age_outlier'].sum())}")
    return df.reset_index(drop=True)


if __name__ == "__main__":
    raw = pd.read_csv(DATA_RAW)
    cleaned = clean(raw)
    DATA_CLEAN.parent.mkdir(parents=True, exist_ok=True)  
    cleaned.to_csv(DATA_CLEAN, index=False)
    print(f"Saved {DATA_CLEAN}")
