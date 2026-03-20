import numpy as np
import pandas as pd


def inspect_data(df: pd.DataFrame, timestamp_col: str = "timestamp") -> None:
    print("Shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst rows:")
    print(df.head())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    print("\n=== NUMERIC DATA QUALITY CHECK ===")

    zero_counts = (df[numeric_cols] == 0).sum()
    print("\nZero values per column:")
    print(zero_counts[zero_counts > 0])

    non_positive = (df[numeric_cols] <= 0).sum()
    print("\nNon-positive values (<=0):")
    print(non_positive[non_positive > 0])

    inf_counts = np.isinf(df[numeric_cols].to_numpy()).sum(axis=0)
    inf_series = pd.Series(inf_counts, index=numeric_cols)
    print("\nInfinity values:")
    print(inf_series[inf_series > 0])

    nan_counts = df[numeric_cols].isna().sum()
    print("\nNaNs in numeric columns:")
    print(nan_counts[nan_counts > 0])

    if timestamp_col in df.columns:
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")

        print("\nDate range:")
        print("Start:", df[timestamp_col].min())
        print("End:", df[timestamp_col].max())

        gaps = df[timestamp_col].diff().value_counts().sort_index()
        print("\nObserved time gaps:")
        print(gaps.head())

    print("\nNumeric summary:")
    print(df.describe(include="all"))