# src/fx_forecasting/data/loading.py

import pandas as pd
from pathlib import Path


def load_fx_csv(
    path: str | Path,
    date_col: str = "Date",
) -> pd.DataFrame:
    """
    Load FX dataset CSV and standardize format.

    Returns
    -------
    pd.DataFrame
        Columns:
        timestamp + currency columns
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    if date_col not in df.columns:
        raise ValueError(f"Expected date column '{date_col}' not found.")

    # Convert to datetime
    df["timestamp"] = pd.to_datetime(df[date_col], errors="coerce")

    if date_col != "timestamp":
        df = df.drop(columns=[date_col])

    # Remove rows where timestamp failed
    df = df.dropna(subset=["timestamp"])

    # Sort by time
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Keep numeric columns only
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if len(numeric_cols) == 0:
        raise ValueError("No numeric FX columns found.")

    df = df[["timestamp"] + numeric_cols]

    return df