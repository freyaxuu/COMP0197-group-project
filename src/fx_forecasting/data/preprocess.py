# src/fx_forecasting/data/preprocessing.py
# src/fx_forecasting/data/preprocessing.py

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Sequence, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def clean_fx_data(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Basic cleaning for daily FX data.
    """
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df = df.dropna(subset=[timestamp_col])
    df = df.sort_values(timestamp_col).reset_index(drop=True)

    value_cols = [c for c in df.columns if c != timestamp_col]
    df[value_cols] = df[value_cols].ffill().bfill()

    return df


def add_log_returns(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Add log return columns for price series.
    """
    df = df.copy()
    if cols is None:
        cols = [c for c in df.columns if c != timestamp_col]

    for col in cols:
        df[f"{col}_ret"] = np.log(df[col] / df[col].shift(1))

    return df


def add_moving_averages(
    df: pd.DataFrame,
    windows: Sequence[int] = (7, 30),
    timestamp_col: str = "timestamp",
    cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Add moving average columns for price series.
    """
    df = df.copy()
    if cols is None:
        cols = [
            c for c in df.columns
            if c != timestamp_col and not c.endswith("_ret")
        ]

    for col in cols:
        for w in windows:
            df[f"{col}_ma{w}"] = df[col].rolling(w).mean()

    return df


def add_rolling_volatility(
    df: pd.DataFrame,
    window: int = 30,
    timestamp_col: str = "timestamp",
    ret_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Add rolling std on return columns.
    """
    df = df.copy()
    if ret_cols is None:
        ret_cols = [c for c in df.columns if c.endswith("_ret")]

    for col in ret_cols:
        df[f"{col}_vol{window}"] = df[col].rolling(window).std()

    return df


def drop_feature_nans(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows introduced by rolling windows / returns.
    """
    return df.dropna().reset_index(drop=True)


def time_train_test_split(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-order-preserving split.
    """
    split_idx = int(len(df) * (1 - test_ratio))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


def scale_train_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    scaler_type: str = "standard",
) -> Tuple[pd.DataFrame, pd.DataFrame, object]:
    """
    Fit scaler on train only, then transform train and test.
    This avoids leakage.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    feature_cols = [c for c in train_df.columns if c != timestamp_col]

    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("scaler_type must be 'standard' or 'minmax'")

    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    return train_df, test_df, scaler


def create_windows(
    df: pd.DataFrame,
    target_col: str,
    lookback: int = 30,
    timestamp_col: str = "timestamp",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows for sequence models.
    X shape: (n_samples, lookback, n_features)
    y shape: (n_samples,)
    """
    feature_cols = [c for c in df.columns if c not in [timestamp_col, target_col]]

    X, y = [], []
    features = df[feature_cols].values
    target = df[target_col].values

    for i in range(lookback, len(df)):
        X.append(features[i - lookback:i])
        y.append(target[i])

    return np.array(X), np.array(y)


def prepare_fx_data(
    df: pd.DataFrame,
    target_col: str,
    timestamp_col: str = "timestamp",
    test_ratio: float = 0.2,
    add_returns: bool = True,
    add_ma: bool = True,
    ma_windows: Sequence[int] = (7, 30),
    add_volatility: bool = True,
    vol_window: int = 30,
    scale: bool = True,
    scaler_type: str = "standard",
    make_windows: bool = False,
    lookback: int = 30,
):
    """
    End-to-end preprocessing pipeline for FX data.

    Returns either:
    - train_df, test_df, scaler
    or
    - X_train, y_train, X_test, y_test, scaler
    """
    df = clean_fx_data(df, timestamp_col=timestamp_col)

    original_price_cols = [c for c in df.columns if c != timestamp_col]

    if add_returns:
        df = add_log_returns(df, timestamp_col=timestamp_col, cols=original_price_cols)

    if add_ma:
        df = add_moving_averages(
            df,
            windows=ma_windows,
            timestamp_col=timestamp_col,
            cols=original_price_cols,
        )

    if add_volatility:
        ret_cols = [c for c in df.columns if c.endswith("_ret")]
        if len(ret_cols) > 0:
            df = add_rolling_volatility(
                df,
                window=vol_window,
                timestamp_col=timestamp_col,
                ret_cols=ret_cols,
            )

    df = drop_feature_nans(df)

    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found after preprocessing")

    train_df, test_df = time_train_test_split(df, test_ratio=test_ratio)

    scaler = None
    if scale:
        train_df, test_df, scaler = scale_train_test(
            train_df,
            test_df,
            timestamp_col=timestamp_col,
            scaler_type=scaler_type,
        )

    if not make_windows:
        return train_df, test_df, scaler

    X_train, y_train = create_windows(
        train_df,
        target_col=target_col,
        lookback=lookback,
        timestamp_col=timestamp_col,
    )
    X_test, y_test = create_windows(
        test_df,
        target_col=target_col,
        lookback=lookback,
        timestamp_col=timestamp_col,
    )

    return X_train, y_train, X_test, y_test, scaler


def inverse_transform_target(
    values,
    scaler,
    columns,
    target_col,
):
    """
    Inverse-transform a 1D array of target values using a scaler fitted on
    multiple columns.
    """
    import numpy as np

    values = np.asarray(values).reshape(-1, 1)

    if target_col not in columns:
        raise ValueError(f"target_col '{target_col}' not found in columns")

    target_idx = columns.index(target_col)

    dummy = np.zeros((len(values), len(columns)))
    dummy[:, target_idx] = values[:, 0]

    inv = scaler.inverse_transform(dummy)

    return inv[:, target_idx]
