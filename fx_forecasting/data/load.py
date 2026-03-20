from pathlib import Path
import pandas as pd


def load_fx_csv(path: str | Path, date_col: str = "Date") -> pd.DataFrame:
    """
    Load FX dataset CSV and standardise format.

    Returns
    -------
    pd.DataFrame
        Columns: timestamp + numeric columns
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    if date_col not in df.columns:
        raise ValueError(f"Expected date column '{date_col}' not found.")

    df["timestamp"] = pd.to_datetime(df[date_col], errors="coerce")

    if date_col != "timestamp":
        df = df.drop(columns=[date_col])

    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found in dataset.")

    df = df[["timestamp"] + numeric_cols]
    return df