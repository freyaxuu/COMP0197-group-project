# src/fx_forecasting/data/inspect.py

# src/fx_forecasting/data/inspect.py

import pandas as pd

def inspect_data(df: pd.DataFrame, timestamp_col: str = "timestamp"):
    """
    Minimal inspection for time-series data.
    """

    print("Shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst rows:")
    print(df.head())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    if timestamp_col in df.columns:
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")

        print("\nDate range:")
        print("Start:", df[timestamp_col].min())
        print("End:", df[timestamp_col].max())

        # check time gaps
        gaps = df[timestamp_col].diff().value_counts().sort_index()

        print("\nObserved time gaps:")
        print(gaps.head())

    print("\nNumeric summary:")
    print(df.describe())