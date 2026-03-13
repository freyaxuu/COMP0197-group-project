from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler


# =========================================================
# 1) DATA LOADING
# =========================================================
def _read_table(path_or_url: str) -> pd.DataFrame:
    """
    Read csv/xlsx/parquet from local path or http(s) URL.
    Supports csv.zip.
    """
    is_url = path_or_url.startswith("http://") or path_or_url.startswith("https://")
    s = path_or_url.lower()

    if s.endswith(".csv.zip") or s.endswith(".zip"):
        return pd.read_csv(path_or_url, compression="zip")

    if s.endswith(".csv"):
        return pd.read_csv(path_or_url)

    if (not is_url) and (s.endswith(".xlsx") or s.endswith(".xls")):
        return pd.read_excel(path_or_url)

    if (not is_url) and s.endswith(".parquet"):
        return pd.read_parquet(path_or_url)

    # fallback: try csv
    return pd.read_csv(path_or_url)


def load_any_table(
    path_or_url: str,
    timestamp_col: str = "timestamp",
    target_col: Optional[str] = None,
    long_table: bool = False,
    long_var_col: str = "variable",
    long_value_col: str = "value",
    make_timestamp_from_ymdh: bool = False,
    y_col: str = "year",
    m_col: str = "month",
    d_col: str = "day",
    h_col: str = "hour",
    downsample_step: int = 1,
) -> Tuple[pd.DataFrame, str]:
    """
    Load a table and return a standardized dataframe with a fixed 'timestamp' column.

    Returns:
        df_wide: DataFrame with [timestamp, numeric features..., target]
        inferred_target_col
    """
    df = _read_table(path_or_url)

    if downsample_step > 1:
        df = df.iloc[::downsample_step].reset_index(drop=True)

    # -------------------------
    # HANDLE LONG FORMAT
    # -------------------------
    if long_table:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")

        df_wide = (
            df.pivot_table(
                index=timestamp_col,
                columns=long_var_col,
                values=long_value_col,
                aggfunc="mean",
            )
            .reset_index()
        )

    # -------------------------
    # HANDLE WIDE FORMAT
    # -------------------------
    else:
        df_wide = df.copy()

        if make_timestamp_from_ymdh:
            for c in [y_col, m_col, d_col, h_col]:
                if c not in df_wide.columns:
                    raise ValueError(f"make_timestamp_from_ymdh=True but missing column: {c}")

            df_wide["timestamp"] = pd.to_datetime(
                df_wide[[y_col, m_col, d_col, h_col]].rename(
                    columns={
                        y_col: "year",
                        m_col: "month",
                        d_col: "day",
                        h_col: "hour",
                    }
                ),
                errors="coerce",
            )

        else:
            if timestamp_col not in df_wide.columns:
                raise ValueError(
                    f"timestamp_col '{timestamp_col}' not found. "
                    f"If your data has year/month/day/hour, use make_timestamp_from_ymdh=True."
                )

            df_wide["timestamp"] = pd.to_datetime(df_wide[timestamp_col], errors="coerce")

    # Drop original timestamp column if name differs
    if timestamp_col != "timestamp" and timestamp_col in df_wide.columns:
        df_wide = df_wide.drop(columns=[timestamp_col])

    # -------------------------
    # SORT + CLEAN
    # -------------------------
    df_wide = (
        df_wide.dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # -------------------------
    # KEEP NUMERIC FEATURES
    # -------------------------
    numeric_cols = [
        c for c in df_wide.columns
        if c != "timestamp" and pd.api.types.is_numeric_dtype(df_wide[c])
    ]

    df_wide = df_wide[["timestamp"] + numeric_cols]

    if len(numeric_cols) < 2:
        raise ValueError("Need at least 2 numeric columns (features + target).")

    if target_col is None:
        target_col = numeric_cols[-1]

    if target_col not in df_wide.columns:
        raise ValueError(
            f"target_col '{target_col}' not in columns: {df_wide.columns.tolist()}"
        )

    return df_wide, target_col


# =========================================================
# 2) DATA INSPECTION + CLEANING
# =========================================================
def inspect_data(df: pd.DataFrame, datetime_col: Optional[str] = None) -> pd.DataFrame:
    """
    Print a quick inspection summary.
    """
    df = df.copy()

    print("Shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing values per column:")
    print(df.isnull().sum())

    print("\nMissing value percentages:")
    print((df.isnull().mean() * 100).round(2))

    print("\nDuplicate rows:", df.duplicated().sum())

    if datetime_col and datetime_col in df.columns:
        print(f"\nConverting '{datetime_col}' to datetime...")
        df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
        print(df[[datetime_col]].head())
        print("\nDatetime nulls after conversion:", df[datetime_col].isnull().sum())

    print("\nNumeric summary:")
    print(df.describe(include=[np.number]))

    return df


def basic_clean(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    fill_method: str = "ffill",
    dropna_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Keep columns with acceptable missingness and fill remaining missing values.
    """
    df = df.copy()
    non_ts = [c for c in df.columns if c != timestamp_col]
    missing_ratio = df[non_ts].isna().mean()

    keep_cols = [c for c in non_ts if missing_ratio[c] <= dropna_threshold]
    df = df[[timestamp_col] + keep_cols]

    if fill_method == "ffill":
        df[keep_cols] = df[keep_cols].ffill().bfill()
    elif fill_method == "bfill":
        df[keep_cols] = df[keep_cols].bfill().ffill()
    elif fill_method == "interpolate":
        df[keep_cols] = df[keep_cols].interpolate().ffill().bfill()
    else:
        raise ValueError("fill_method must be one of: 'ffill', 'bfill', 'interpolate'")

    return df


# =========================================================
# 3) INSPECTION PLOTS
# =========================================================
def plot_missingness(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    title: str = "Dataset",
    save_path: Optional[str] = None,
):
    cols = [c for c in df.columns if c != timestamp_col]
    miss = df[cols].isna().mean().sort_values(ascending=False)

    plt.figure(figsize=(10, 4))
    plt.bar(miss.index, miss.values)
    plt.title(f"{title} - Missing Ratio Per Column")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_corr_heatmap(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    title: str = "Dataset",
    save_path: Optional[str] = None,
):
    cols = [c for c in df.columns if c != timestamp_col]
    corr = df[cols].corr().values

    plt.figure(figsize=(6, 5))
    plt.imshow(corr, aspect="auto")
    plt.colorbar()
    plt.title(f"{title} - Correlation Heatmap")
    plt.xticks(range(len(cols)), cols, rotation=45, ha="right")
    plt.yticks(range(len(cols)), cols)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_timeseries(
    df: pd.DataFrame,
    timestamp_col: str,
    cols: List[str],
    title: str = "Time Series",
    save_path: Optional[str] = None,
):
    plt.figure(figsize=(10, 4))
    for c in cols:
        plt.plot(df[timestamp_col], df[c], label=c, linewidth=1)
    plt.legend()
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def plot_time_series(df: pd.DataFrame, col: str, timestamp_col: Optional[str] = None, title: str = "Time Series"):
    plt.figure(figsize=(15, 5))
    if timestamp_col and timestamp_col in df.columns:
        plt.plot(df[timestamp_col], df[col], linewidth=0.8)
        plt.xlabel(timestamp_col)
    else:
        plt.plot(df[col], linewidth=0.8)
        plt.xlabel("Index")

    plt.title(title)
    plt.ylabel(col)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_numeric_distributions(df: pd.DataFrame, cols: Optional[List[str]] = None, bins: int = 30):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if cols is None:
        cols = numeric_cols

    for col in cols:
        plt.figure(figsize=(6, 4))
        df[col].dropna().hist(bins=bins)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()


def plot_boxplots(df: pd.DataFrame, cols: Optional[List[str]] = None):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if cols is None:
        cols = numeric_cols

    for col in cols:
        plt.figure(figsize=(6, 4))
        plt.boxplot(df[col].dropna())
        plt.title(f"Boxplot of {col}")
        plt.ylabel(col)
        plt.tight_layout()
        plt.show()


def plot_results(targets, preds, sigma=None, title="Model Evaluation"):
    """
    Supports deterministic (preds only) or probabilistic (preds + sigma) output.
    """
    targets = np.asarray(targets)
    preds = np.asarray(preds)

    plt.figure(figsize=(15, 6))
    plt.plot(targets, label="Actual", color="black", alpha=0.7)
    plt.plot(preds, label="Predicted Mean", color="red", linestyle="--")

    if sigma is not None:
        sigma = np.asarray(sigma)
        plt.fill_between(
            range(len(preds)),
            preds - 2 * sigma,
            preds + 2 * sigma,
            color="red",
            alpha=0.2,
            label="95% Confidence",
        )

    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


# =========================================================
# 4) WINDOW DATASET FOR BASELINE LSTM
# =========================================================
@dataclass
class WindowConfig:
    lookback: int = 48
    horizon: int = 1
    stride: int = 1


class WindowDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, cfg: WindowConfig):
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)
        self.cfg = cfg

        T = len(x)
        L, H, S = cfg.lookback, cfg.horizon, cfg.stride
        self.indices = list(range(0, T - (L + H) + 1, S))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i: int):
        t = self.indices[i]
        L, H = self.cfg.lookback, self.cfg.horizon
        x_win = self.x[t:t + L]          # [L, D]
        y_win = self.y[t + L:t + L + H]  # [H, 1] or [H]
        return torch.from_numpy(x_win), torch.from_numpy(y_win)


def time_split_indices(n: int, train_ratio: float = 0.7, val_ratio: float = 0.15):
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    idx_train = np.arange(0, n_train)
    idx_val = np.arange(n_train, n_train + n_val)
    idx_test = np.arange(n_train + n_val, n)

    return idx_train, idx_val, idx_test


# =========================================================
# 5) SEQUENCE PREP FOR LSTM (NO LEAKAGE)
# =========================================================
def prepare_data(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    seq_length: int,
    train_split: float = 0.8,
    batch_size: int = 64,
):
    """
    Generalized sequence creator with no data leakage.

    Returns:
        train_loader, test_loader, scaler
    """
    data = df[feature_cols].values
    target_idx = feature_cols.index(target_col)

    split_idx = int(train_split * len(data))
    train_data_raw = data[:split_idx]
    test_data_raw = data[split_idx:]

    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data_raw)
    test_data_scaled = scaler.transform(test_data_raw)

    full_data_scaled = np.vstack((train_data_scaled, test_data_scaled))

    xs, ys = [], []
    for i in range(len(full_data_scaled) - seq_length):
        xs.append(full_data_scaled[i:i + seq_length])
        ys.append(full_data_scaled[i + seq_length, target_idx])

    X = np.array(xs)
    y = np.array(ys)

    seq_split = split_idx - seq_length
    X_train = torch.FloatTensor(X[:seq_split])
    X_test = torch.FloatTensor(X[seq_split:])
    y_train = torch.FloatTensor(y[:seq_split])
    y_test = torch.FloatTensor(y[seq_split:])

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loader, scaler


# =========================================================
# 6) BASELINES
# =========================================================
def persistence_mae(y: np.ndarray) -> float:
    """
    Naive baseline: predict next value using previous value.
    """
    y = np.asarray(y).reshape(-1)
    y_true = y[1:]
    y_pred = y[:-1]
    return float(np.mean(np.abs(y_true - y_pred)))


# =========================================================
# 7) OPTIONAL END-TO-END HELPER
# =========================================================
def load_clean_inspect(
    path_or_url: str,
    timestamp_col: str = "timestamp",
    target_col: Optional[str] = None,
    fill_method: str = "ffill",
    dropna_threshold: float = 0.5,
    long_table: bool = False,
    long_var_col: str = "variable",
    long_value_col: str = "value",
    make_timestamp_from_ymdh: bool = False,
    y_col: str = "year",
    m_col: str = "month",
    d_col: str = "day",
    h_col: str = "hour",
    downsample_step: int = 1,
    inspect: bool = True,
) -> Tuple[pd.DataFrame, str]:
    """
    Full pipeline:
      1. load
      2. inspect
      3. clean
      4. return cleaned df + target_col
    """
    df, inferred_target = load_any_table(
        path_or_url=path_or_url,
        timestamp_col=timestamp_col,
        target_col=target_col,
        long_table=long_table,
        long_var_col=long_var_col,
        long_value_col=long_value_col,
        make_timestamp_from_ymdh=make_timestamp_from_ymdh,
        y_col=y_col,
        m_col=m_col,
        d_col=d_col,
        h_col=h_col,
        downsample_step=downsample_step,
    )

    if inspect:
        inspect_data(df, datetime_col=timestamp_col)

    df = basic_clean(
        df,
        timestamp_col=timestamp_col,
        fill_method=fill_method,
        dropna_threshold=dropna_threshold,
    )

    return df, inferred_target