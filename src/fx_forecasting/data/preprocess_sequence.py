from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def load_interim_dataset(
    csv_path: str | Path,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df = df.dropna(subset=[timestamp_col]).sort_values(timestamp_col).reset_index(drop=True)
    return df


def infer_feature_columns(
    df: pd.DataFrame,
    target_col: str,
    timestamp_col: str = "timestamp",
) -> List[str]:
    return [c for c in df.columns if c not in {timestamp_col, target_col}]


def time_train_test_split(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be in (0, 1)")

    split_idx = int(len(df) * (1 - test_ratio))
    train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)
    return train_df, test_df


def make_scaler(scaler_type: str):
    if scaler_type == "standard":
        return StandardScaler()
    if scaler_type == "minmax":
        return MinMaxScaler()
    raise ValueError("scaler_type must be 'standard' or 'minmax'")


def scale_train_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    scale_target: bool = False,
    scaler_type: str = "standard",
) -> Tuple[pd.DataFrame, pd.DataFrame, object, Optional[object]]:
    train_df = train_df.copy()
    test_df = test_df.copy()

    x_scaler = make_scaler(scaler_type)
    y_scaler = make_scaler(scaler_type) if scale_target else None

    train_df[list(feature_cols)] = x_scaler.fit_transform(train_df[list(feature_cols)])
    test_df[list(feature_cols)] = x_scaler.transform(test_df[list(feature_cols)])

    if scale_target:
        train_df[[target_col]] = y_scaler.fit_transform(train_df[[target_col]])
        test_df[[target_col]] = y_scaler.transform(test_df[[target_col]])

    return train_df, test_df, x_scaler, y_scaler


def save_split_csvs(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_path: str | Path,
    test_path: str | Path,
) -> None:
    train_path = Path(train_path)
    test_path = Path(test_path)

    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)


def create_windows(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    lookback: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    features = df[list(feature_cols)].values
    target = df[target_col].values

    X, y = [], []
    for i in range(lookback, len(df)):
        X.append(features[i - lookback:i])
        y.append(target[i])

    return np.asarray(X), np.asarray(y)


def create_train_test_windows(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    lookback: int = 30,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, y_train = create_windows(
        train_df,
        feature_cols=feature_cols,
        target_col=target_col,
        lookback=lookback,
    )

    test_with_context = pd.concat(
        [train_df.tail(lookback), test_df],
        axis=0,
        ignore_index=True,
    )

    X_test, y_test = create_windows(
        test_with_context,
        feature_cols=feature_cols,
        target_col=target_col,
        lookback=lookback,
    )

    X_test = X_test[-len(test_df):]
    y_test = y_test[-len(test_df):]

    return X_train, y_train, X_test, y_test


def inverse_transform_target(
    values: np.ndarray,
    y_scaler,
) -> np.ndarray:
    if y_scaler is None:
        return np.asarray(values)

    values = np.asarray(values).reshape(-1, 1)
    return y_scaler.inverse_transform(values).ravel()


def prepare_sequences_from_csv(
    csv_path: str | Path,
    target_col: str,
    timestamp_col: str = "timestamp",
    test_ratio: float = 0.2,
    scale_features: bool = True,
    scale_target: bool = False,
    scaler_type: str = "standard",
    lookback: int = 30,
    train_csv_path: Optional[str | Path] = None,
    test_csv_path: Optional[str | Path] = None,
):
    df = load_interim_dataset(csv_path, timestamp_col=timestamp_col)
    feature_cols = infer_feature_columns(df, target_col=target_col, timestamp_col=timestamp_col)

    train_df, test_df = time_train_test_split(df, test_ratio=test_ratio)

    x_scaler, y_scaler = None, None
    if scale_features or scale_target:
        train_df, test_df, x_scaler, y_scaler = scale_train_test(
            train_df=train_df,
            test_df=test_df,
            feature_cols=feature_cols,
            target_col=target_col,
            scale_target=scale_target,
            scaler_type=scaler_type,
        )

    if train_csv_path is not None and test_csv_path is not None:
        save_split_csvs(train_df, test_df, train_csv_path, test_csv_path)

    X_train, y_train, X_test, y_test = create_train_test_windows(
        train_df=train_df,
        test_df=test_df,
        feature_cols=feature_cols,
        target_col=target_col,
        lookback=lookback,
    )

    return {
        "df": df,
        "train_df": train_df,
        "test_df": test_df,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }


if __name__ == "__main__":
    interim_csv = "data/interim/fx_interim.csv"

    results = prepare_sequences_from_csv(
        csv_path=interim_csv,
        target_col="GBP-USD_ret_fwd1",
        timestamp_col="timestamp",
        test_ratio=0.2,
        scale_features=True,
        scale_target=False,
        scaler_type="standard",
        lookback=30,
        train_csv_path="data/processed/train_scaled.csv",
        test_csv_path="data/processed/test_scaled.csv",
    )

    print("Train window shape:", results["X_train"].shape)
    print("Test window shape:", results["X_test"].shape)
    print("Target column:", results["target_col"])
    print("Num features:", len(results["feature_cols"]))