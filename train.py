# Generative AI (GenAI) Usage Statement:
# Generative AI tools were used to assist in generating annotations and
# docstrings, as well as improving code formatting and readability.
# The design, implementation, and validation of the code remain the
# authors' own work.

from __future__ import annotations

from pathlib import Path
import json
import os
import random
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor


# =========================================================
# Paths and config
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

DATA_URL = "https://raw.githubusercontent.com/freyaxuu/COMP0197-group-project/main/data/fx_merged.csv"

RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
FIGURES_DIR = PROJECT_ROOT / "training_figures"

RAW_DIR.mkdir(parents=True, exist_ok=True)
INTERIM_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

RAW_CACHE_PATH = RAW_DIR / "fx_merged.csv"
INTERIM_SELECTED_PATH = INTERIM_DIR / "interim_selected.csv"
FEATURE_METADATA_PATH = INTERIM_DIR / "selected_features_metadata.json"

TEST_RATIO = 0.2
SEED = 42


# =========================================================
# Reproducibility
# =========================================================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Seed set to: {seed}")


# =========================================================
# Data retrieval
# =========================================================
def load_raw_fx_data(data_url: str = DATA_URL, save_local_copy: bool = True) -> pd.DataFrame:
    df_raw = pd.read_csv(data_url)

    if save_local_copy:
        df_raw.to_csv(RAW_CACHE_PATH, index=False)

    return df_raw


# =========================================================
# Data preparation
# =========================================================
def clean_raw_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    if "Date" not in df.columns:
        raise ValueError("Expected 'Date' column not found in raw dataset.")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("Date").ffill().dropna().reset_index(drop=True)

    return df


def build_stationary_dataset(df: pd.DataFrame) -> pd.DataFrame:
    price_cols = [
        "GBP-USD", "GBP-CNY", "GBP-EUR", "GBP-JPY", "GBP-KRW", "GBP-CHF",
        "ftse100_Close", "gold_price", "DTWEXBGS", "DCOILBRENTEU", "vix_daily_close",
    ]
    rate_cols = ["DGS10", "glc_nominal_2", "glc_nominal_10", "ois_1", "ois_2"]

    missing = [c for c in price_cols + rate_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in raw dataset: {missing}")

    df_station = pd.DataFrame(index=df.index)
    df_station["Date"] = df["Date"]

    for col in price_cols:
        df_station[f"{col}_ret"] = np.log(df[col] / df[col].shift(1))

    for col in rate_cols:
        df_station[f"{col}_diff"] = df[col].diff()

    df_station["target"] = df_station["GBP-USD_ret"].shift(-1)
    df_station = df_station.dropna().reset_index(drop=True)

    print(f"Stationary dataset shape: {df_station.shape}")
    return df_station


def expand_features(df: pd.DataFrame, base_cols: list[str]) -> pd.DataFrame:
    df_ext = df.copy()

    for col in base_cols:
        df_ext[f"{col}_lag1"] = df[col].shift(1)
        df_ext[f"{col}_lag5"] = df[col].shift(5)
        df_ext[f"{col}_vol20"] = df[col].rolling(window=20).std().shift(1)
        df_ext[f"{col}_ma20"] = df[col].rolling(window=20).mean().shift(1)

    return df_ext.dropna().reset_index(drop=True)


def time_split(df: pd.DataFrame, test_ratio: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(df) * (1 - test_ratio))
    df_train = df.iloc[:split_idx].copy().reset_index(drop=True)
    df_test = df.iloc[split_idx:].copy().reset_index(drop=True)
    return df_train, df_test


# =========================================================
# Plot helpers
# =========================================================
def save_corr_heatmap(
    df: pd.DataFrame,
    feature_cols: list[str],
    save_path: Path,
    title: str,
) -> None:
    corr = df[feature_cols].corr().values

    plt.figure(figsize=(12, 8))
    im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im)
    plt.xticks(range(len(feature_cols)), feature_cols, rotation=90)
    plt.yticks(range(len(feature_cols)), feature_cols)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_mi_barplot(mi_results: pd.Series, save_path: Path, top_k: int = 15) -> None:
    plt.figure(figsize=(10, 6))
    mi_results.head(top_k).sort_values(ascending=True).plot(kind="barh")
    plt.title("Top 15 Features by Mutual Information Score")
    plt.xlabel("Information Gain")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_correlation_heatmap(
    df: pd.DataFrame,
    timestamp_col: str,
    save_path: Path,
    title: str = "Correlation Heatmap",
) -> None:
    value_cols = [c for c in df.columns if c != timestamp_col]
    corr = df[value_cols].corr()

    plt.figure(figsize=(8, 6))
    im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im)
    plt.xticks(range(len(value_cols)), value_cols, rotation=45, ha="right")
    plt.yticks(range(len(value_cols)), value_cols)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_all_series(
    df: pd.DataFrame,
    timestamp_col: str,
    save_path: Path,
    title: str = "Exchange Rate Time Series",
) -> None:
    value_cols = [c for c in df.columns if c != timestamp_col]

    plt.figure(figsize=(12, 6))
    for col in value_cols:
        plt.plot(df[timestamp_col], df[col], label=col)

    plt.xlabel("Date")
    plt.ylabel("Exchange Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_target_series(
    df: pd.DataFrame,
    target_col: str,
    timestamp_col: str,
    save_path: Path,
    title: str = "Target Over Time",
) -> None:
    plt.figure(figsize=(12, 5))
    plt.plot(df[timestamp_col], df[target_col])
    plt.xlabel("Date")
    plt.ylabel(target_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# =========================================================
# Feature selection
# =========================================================
def correlation_filter(
    df_train: pd.DataFrame,
    feature_cols: list[str],
    threshold: float = 0.85,
) -> tuple[list[str], list[str]]:
    corr_matrix = df_train[feature_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    final_features = [f for f in feature_cols if f not in to_drop]
    return final_features, to_drop


def mutual_information_filter(
    df_train: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "target",
    threshold: float = 0.001,
) -> tuple[list[str], pd.Series]:
    X = df_train[feature_cols]
    y = df_train[target_col]

    mi_scores = mutual_info_regression(X, y, random_state=SEED)
    mi_results = pd.Series(mi_scores, index=feature_cols).sort_values(ascending=False)
    effective_features = mi_results[mi_results > threshold].index.tolist()

    return effective_features, mi_results


def random_forest_noise_benchmark_filter(
    df_train: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "target",
) -> tuple[list[str], pd.Series, float]:
    X_final = df_train[feature_cols].copy()
    y = df_train[target_col]

    rng = np.random.default_rng(SEED)
    X_final["RANDOM_NOISE"] = rng.normal(0, 1, size=len(X_final))

    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=SEED,
    )
    rf.fit(X_final, y)

    importances = pd.Series(rf.feature_importances_, index=X_final.columns).sort_values(ascending=False)
    benchmark = float(importances["RANDOM_NOISE"])

    production_features = importances[importances > benchmark].index.tolist()
    production_features = [f for f in production_features if f != "RANDOM_NOISE"]

    return production_features, importances, benchmark


# =========================================================
# Main pipeline up to interim_selected.csv
# =========================================================
def build_interim_selected_dataset() -> tuple[pd.DataFrame, list[str], pd.DataFrame, pd.DataFrame]:
    """
    Build the intermediate selected dataset used by downstream training.

    This function performs the full feature-engineering pipeline:
    1. load and clean the raw FX dataset,
    2. apply stationary transformations,
    3. expand lag/rolling features,
    4. split the data chronologically into train and test parts,
    5. apply correlation filtering, mutual-information filtering, and
    random-noise benchmark filtering,
    6. save the final selected dataset, metadata, and diagnostic figures.

    Returns
    -------
    tuple[pd.DataFrame, list[str], pd.DataFrame, pd.DataFrame]
        Final selected full dataset, list of production feature names,
        selected training split, and selected test split.
    """
    # -------------------------
    # 1. Load Data
    # -------------------------
    df_raw = load_raw_fx_data(DATA_URL, save_local_copy=True)
    df = clean_raw_data(df_raw)

    # -------------------------
    # 2. Stationary transforms
    # -------------------------
    df_station = build_stationary_dataset(df)

    # -------------------------
    # 3. Expand features
    # -------------------------
    primary_drivers = [
        "GBP-USD_ret",
        "DGS10_diff",
        "vix_daily_close_ret",
        "glc_nominal_10_diff",
    ]

    df_features = expand_features(df_station, primary_drivers)

    # Add all features except Date/target
    all_features = [c for c in df_features.columns if c not in ["Date", "target"]]

    # -------------------------
    # 4. Train/Test split
    # -------------------------
    df_train, df_test = time_split(df_features, test_ratio=TEST_RATIO)

    print("train rows:", len(df_train), "test rows:", len(df_test))
    print("train date range:", df_train["Date"].min(), "->", df_train["Date"].max())
    print("test date range:", df_test["Date"].min(), "->", df_test["Date"].max())

    # -------------------------
    # 5. Correlation filter
    # -------------------------
    final_features, to_drop = correlation_filter(
        df_train=df_train,
        feature_cols=all_features,
        threshold=0.85,
    )
    print(f"Dropping {len(to_drop)} redundant features...")

    save_corr_heatmap(
        df=df_train,
        feature_cols=final_features,
        save_path=FIGURES_DIR / "filtered_feature_correlation.png",
        title="Filtered Feature Correlation",
    )

    # -------------------------
    # 6. Mutual information filter
    # -------------------------
    effective_features, mi_results = mutual_information_filter(
        df_train=df_train,
        feature_cols=final_features,
        target_col="target",
        threshold=0.001,
    )
    print(f"Features remaining after MI filter: {len(effective_features)}")

    save_mi_barplot(
        mi_results=mi_results,
        save_path=FIGURES_DIR / "top15_mutual_information.png",
        top_k=15,
    )

    # -------------------------
    # 7. Random-noise benchmark filter
    # -------------------------
    production_features, importances, benchmark = random_forest_noise_benchmark_filter(
        df_train=df_train,
        feature_cols=effective_features,
        target_col="target",
    )

    print(f"Final Production Feature Set ({len(production_features)}):")
    print(production_features)

    selected_cols = ["Date", "target"] + production_features
    train_selected = df_train[selected_cols].copy()
    test_selected = df_test[selected_cols].copy()

    print("Selected feature count:", len(production_features))
    print("Train selected shape:", train_selected.shape)
    print("Test selected shape:", test_selected.shape)

    # -------------------------
    # 8. Save plots for selected training data
    # -------------------------
    plot_correlation_heatmap(
        df=train_selected,
        timestamp_col="Date",
        save_path=FIGURES_DIR / "train_selected_correlation_heatmap.png",
        title="Correlation Heatmap",
    )

    plot_all_series(
        df=train_selected,
        timestamp_col="Date",
        save_path=FIGURES_DIR / "train_selected_all_series.png",
        title="Exchange Rate Time Series",
    )

    plot_target_series(
        df=train_selected,
        target_col="target",
        timestamp_col="Date",
        save_path=FIGURES_DIR / "train_selected_target_over_time.png",
        title="Target Over Time",
    )

    # -------------------------
    # 9. Final dataset used later
    # -------------------------
    keep_cols = ["Date", "target"] + production_features
    df_final = df_features[keep_cols].copy()
    df_final.to_csv(INTERIM_SELECTED_PATH, index=False)

    metadata = {
        "seed": SEED,
        "data_url": DATA_URL,
        "raw_cache_path": str(RAW_CACHE_PATH),
        "interim_selected_path": str(INTERIM_SELECTED_PATH),
        "test_ratio": TEST_RATIO,
        "primary_drivers": primary_drivers,
        "all_features": all_features,
        "correlation_filtered_features": final_features,
        "effective_features_after_mi": effective_features,
        "production_features": production_features,
        "random_noise_benchmark": benchmark,
        "selected_columns": keep_cols,
        "train_rows": len(df_train),
        "test_rows": len(df_test),
    }
    with open(FEATURE_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    return df_final, production_features, df_train, df_test





# =========================================================
# Sequence preparation for LSTM training
# =========================================================
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from fx_forecasting.training.train_model import train_model, train_model_gaussian


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_SCALED_PATH = PROCESSED_DIR / "train_scaled.csv"
TEST_SCALED_PATH = PROCESSED_DIR / "test_scaled.csv"

BEST_BASELINE_MODEL_PATH = PROJECT_ROOT / "best_baseline_model.pt"
BEST_GAUSSIAN_MODEL_PATH = PROJECT_ROOT / "best_gaussian_model.pt"

# notebook hyperparameters
HIDDEN_DIM = 32
NUM_LAYERS = 2
LR = 0.0005
EPOCHS = 100
BATCH_SIZE = 64
WEIGHT_DECAY = 5e-5
DROPOUT = 0.4
MC_SAMPLES = 200
GAUSSIAN_DROPOUT = 0.10


def load_interim_dataset(
    csv_path: str | Path,
    timestamp_col: str = "Date",
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df = df.dropna(subset=[timestamp_col]).sort_values(timestamp_col).reset_index(drop=True)
    return df


def infer_feature_columns(
    df: pd.DataFrame,
    target_col: str,
    timestamp_col: str = "Date",
) -> list[str]:
    return [c for c in df.columns if c not in {timestamp_col, target_col}]


def time_train_test_split(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    feature_cols: list[str],
    target_col: str,
    scale_target: bool = False,
    scaler_type: str = "standard",
):
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
    feature_cols: list[str],
    target_col: str,
    lookback: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
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
    feature_cols: list[str],
    target_col: str,
    lookback: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def prepare_sequences_from_csv(
    csv_path: str | Path,
    target_col: str,
    timestamp_col: str = "Date",
    test_ratio: float = 0.2,
    scale_features: bool = True,
    scale_target: bool = False,
    scaler_type: str = "standard",
    lookback: int = 30,
    train_csv_path: str | Path | None = None,
    test_csv_path: str | Path | None = None,
):
    """
    Load an intermediate CSV dataset, split it chronologically, optionally
    scale features/targets, and convert the tabular data into rolling input
    windows for sequence models.

    Parameters
    ----------
    csv_path : str | Path
        Path to the intermediate dataset.
    target_col : str
        Name of the prediction target column.
    timestamp_col : str, default="Date"
        Name of the timestamp column.
    test_ratio : float, default=0.2
        Fraction of the dataset reserved for the held-out split.
    scale_features : bool, default=True
        Whether to scale feature columns.
    scale_target : bool, default=False
        Whether to scale the target column.
    scaler_type : str, default="standard"
        Scaling method, either "standard" or "minmax".
    lookback : int, default=30
        Number of past time steps used in each input sequence.

    Returns
    -------
    dict
        Dictionary containing split DataFrames, feature names, scalers, and
        NumPy arrays for X_train, y_train, X_test, and y_test.
    """
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
        save_split_csvs(train_df, test_df, train_csv_path, test_path=test_csv_path)

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


# =========================================================
# Baseline models
# =========================================================
class DeterministicLSTM(nn.Module):
    """
    Baseline deterministic LSTM for time series forecasting.
    Predicts a single next-step value.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.dropout_rate = dropout

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        output = self.fc(last_hidden)
        return output


class GaussianLSTM(nn.Module):
    """
    LSTM with Gaussian output for heteroscedastic regression.
    Returns mean and log-variance.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        min_log_var: float = -14.0,
        max_log_var: float = -10.0,
    ):
        super().__init__()

        self.dropout_rate = dropout
        self.min_log_var = min_log_var
        self.max_log_var = max_log_var

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(p=dropout)
        self.mean_head = nn.Linear(hidden_dim, 1)
        self.raw_log_var_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        last_hidden = self.dropout(last_hidden)

        mean = self.mean_head(last_hidden)
        raw_log_var = self.raw_log_var_head(last_hidden)

        log_var = self.min_log_var + (self.max_log_var - self.min_log_var) * torch.sigmoid(raw_log_var)
        return mean, log_var


def plot_training_history(history, metric="loss", save_path: Path | None = None):
    train_key = f"train_{metric}"
    val_key = f"val_{metric}"

    if train_key not in history or val_key not in history:
        raise ValueError(
            f"Metric '{metric}' not found in history. "
            f"Available keys: {list(history.keys())}"
        )

    train_values = history[train_key]
    val_values = history[val_key]
    epochs = range(1, len(train_values) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_values, label=f"Train {metric.upper()}")
    plt.plot(epochs, val_values, label=f"Validation {metric.upper()}")
    plt.xlabel("Epoch")
    plt.ylabel(metric.upper())
    plt.title(f"Training History ({metric.upper()})")
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# =========================================================
# Baseline training pipeline
# =========================================================
def run_baseline_training():
    """
    Train the baseline LSTM models on the selected intermediate dataset.

    This function prepares scaled sequence data, builds dataloaders, trains
    the deterministic baseline LSTM and the Gaussian baseline LSTM, saves the
    best checkpoints, and exports training-history plots.

    Returns
    -------
    dict
        Dictionary containing prepared data objects, dataloaders, and the
        recorded training histories for both baseline variants.
    """
    interim_path = INTERIM_SELECTED_PATH
    interim_df = pd.read_csv(interim_path)

    results = prepare_sequences_from_csv(
        csv_path=interim_path,
        target_col="target",
        timestamp_col="Date",
        test_ratio=0.2,
        scale_features=True,
        scale_target=False,
        scaler_type="standard",
        lookback=30,
        train_csv_path=TRAIN_SCALED_PATH,
        test_csv_path=TEST_SCALED_PATH,
    )

    X_train = results["X_train"]
    y_train = results["y_train"]
    X_test = results["X_test"]
    y_test = results["y_test"]
    train_df = results["train_df"]
    test_df = results["test_df"]
    feature_cols = results["feature_cols"]

    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_test :", X_test.shape)
    print("y_test :", y_test.shape)

    # convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    # datasets
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_test_t, y_test_t)

    # dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # -------------------------
    # Deterministic baseline
    # -------------------------
    model = DeterministicLSTM(
        input_dim=X_train.shape[-1],
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.SmoothL1Loss(beta=0.008)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=EPOCHS,
        device=device,
        save_path=str(BEST_BASELINE_MODEL_PATH),
        early_stopping_patience=5,
    )

    plot_training_history(
        history,
        metric="loss",
        save_path=FIGURES_DIR / "deterministic_training_history_loss.png",
    )

    checkpoint = torch.load(
    str(BEST_BASELINE_MODEL_PATH),
    map_location=device,
    weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print("Loaded best model from epoch:", checkpoint["epoch"])
    print("Best validation loss:", checkpoint["val_loss"])


    # -------------------------
    # Gaussian baseline
    # -------------------------
    gaussian_model = GaussianLSTM(
        input_dim=X_train.shape[-1],
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=GAUSSIAN_DROPOUT,
        min_log_var=-14.0,
        max_log_var=-10.0,
    )

    gaussian_optimizer = optim.Adam(
        gaussian_model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    gaussian_history = train_model_gaussian(
        model=gaussian_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=gaussian_optimizer,
        epochs=EPOCHS,
        device=device,
        save_path=str(BEST_GAUSSIAN_MODEL_PATH),
        early_stopping_patience=5,
    )

    plot_training_history(
        gaussian_history,
        metric="loss",
        save_path=FIGURES_DIR / "gaussian_training_history_loss.png",
    )

    return {
        "interim_df": interim_df,
        "results": results,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "deterministic_history": history,
        "gaussian_history": gaussian_history,
    }




# =========================================================
# Dual-Branch LSTM (Deterministic + Gaussian)
# =========================================================
from fx_forecasting.data.feature_groups import (
    build_feature_groups,
    groups_to_indices,
    print_group_summary,
)
from fx_forecasting.models.dual_branch import DualBranchLSTM, GaussianDualBranchLSTM
from fx_forecasting.models.dc_lstm import DCLSTM, GaussianDCLSTM


DUAL_HIDDEN_DIM = 48
DUAL_NUM_LAYERS = 1
DUAL_LR = 1e-3
DUAL_EPOCHS = 50
DUAL_BATCH_SIZE = 64
DUAL_WEIGHT_DECAY = 1e-6
DUAL_DROPOUT = 0.4
DUAL_GAUSSIAN_DROPOUT = 0.05
DUAL_LOOKBACK = 30
DUAL_EARLY_STOPPING = 5

BEST_DUAL_BRANCH_MODEL_PATH = PROJECT_ROOT / "best_dual_branch_model.pt"
BEST_GAUSSIAN_DUAL_MODEL_PATH = PROJECT_ROOT / "best_gaussian_dual_model.pt"


class DualBranchWrapper(nn.Module):
    """
    Routes (B, T, D_total) to DualBranchLSTM by slicing
    market and macro feature indices separately.
    """
    def __init__(self, market_idx, macro_idx, hidden_dim, num_layers, dropout):
        super().__init__()
        self.market_idx = market_idx
        self.macro_idx = macro_idx
        self.core = DualBranchLSTM(
            market_input_dim=len(market_idx),
            macro_input_dim=len(macro_idx),
            hidden_dim_market=hidden_dim,
            hidden_dim_macro=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            output_dim=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.core(x[:, :, self.market_idx], x[:, :, self.macro_idx])


class GaussianDualWrapper(nn.Module):
    """
    Routes (B, T, D_total) to GaussianDualBranchLSTM.
    Returns (mean, log_var).
    """
    def __init__(
        self,
        market_idx,
        macro_idx,
        hidden_dim,
        num_layers,
        dropout,
        min_log_var=-14.0,
        max_log_var=-10.0,
    ):
        super().__init__()
        self.market_idx = market_idx
        self.macro_idx = macro_idx
        self.core = GaussianDualBranchLSTM(
            market_input_dim=len(market_idx),
            macro_input_dim=len(macro_idx),
            hidden_dim_market=hidden_dim,
            hidden_dim_macro=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            min_log_var=min_log_var,
            max_log_var=max_log_var,
        )

    def forward(self, x: torch.Tensor):
        return self.core(x[:, :, self.market_idx], x[:, :, self.macro_idx])


def run_dual_branch_training(
    interim_csv: Path = INTERIM_SELECTED_PATH,
    save_det: Path = BEST_DUAL_BRANCH_MODEL_PATH,
    save_gauss: Path = BEST_GAUSSIAN_DUAL_MODEL_PATH,
):
    """
    Train deterministic and Gaussian dual-branch LSTM models.
    Randomness setup follows the teammate version closely.
    """
    import random as _random
    import os as _os
    import numpy as _np

    _random.seed(SEED)
    _os.environ["PYTHONHASHSEED"] = str(SEED)
    _np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    results = prepare_sequences_from_csv(
        csv_path=interim_csv,
        target_col="target",
        timestamp_col="Date",
        test_ratio=TEST_RATIO,
        scale_features=True,
        scale_target=False,
        scaler_type="standard",
        lookback=DUAL_LOOKBACK,
        train_csv_path=PROCESSED_DIR / "train_scaled.csv",
        test_csv_path=PROCESSED_DIR / "test_scaled.csv",
    )

    X_train = results["X_train"]
    y_train = results["y_train"]
    X_test = results["X_test"]
    y_test = results["y_test"]
    feature_cols = results["feature_cols"]

    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")

    groups = build_feature_groups(
        feature_cols,
        target_col="target",
        timestamp_col="Date",
        allow_other=True,
    )
    print_group_summary(groups)

    idx_map = groups_to_indices(groups, feature_cols)
    market_idx = idx_map["market_idx"] + idx_map["other_idx"]
    macro_idx = idx_map["macro_idx"]

    print(f"market dim: {len(market_idx)} | macro dim: {len(macro_idx)}")

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        ),
        batch_size=DUAL_BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32),
        ),
        batch_size=DUAL_BATCH_SIZE,
        shuffle=False,
    )

    # -------------------------
    # Deterministic Dual-Branch
    # -------------------------
    print("\n" + "=" * 60)
    print("Training: Deterministic Dual-Branch LSTM")
    print("=" * 60)

    det_model = DualBranchWrapper(
        market_idx=market_idx,
        macro_idx=macro_idx,
        hidden_dim=DUAL_HIDDEN_DIM,
        num_layers=DUAL_NUM_LAYERS,
        dropout=DUAL_DROPOUT,
    )

    det_history = train_model(
        model=det_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optim.Adam(
            det_model.parameters(),
            lr=DUAL_LR,
            weight_decay=DUAL_WEIGHT_DECAY,
        ),
        criterion=nn.SmoothL1Loss(beta=0.008),
        epochs=DUAL_EPOCHS,
        device=device,
        save_path=str(save_det),
        early_stopping_patience=DUAL_EARLY_STOPPING,
    )

    plot_training_history(
        det_history,
        metric="loss",
        save_path=FIGURES_DIR / "dual_branch_training_history_loss.png",
    )

    print(f"\nDeterministic dual-branch model saved -> {save_det}")

    # -------------------------
    # Gaussian Dual-Branch
    # -------------------------
    print("\n" + "=" * 60)
    print("Training: Gaussian Dual-Branch LSTM")
    print("=" * 60)

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

    gauss_model = GaussianDualWrapper(
        market_idx=market_idx,
        macro_idx=macro_idx,
        hidden_dim=DUAL_HIDDEN_DIM,
        num_layers=DUAL_NUM_LAYERS,
        dropout=DUAL_GAUSSIAN_DROPOUT,
        min_log_var=-14.0,
        max_log_var=-10.0,
    )

    gauss_history = train_model_gaussian(
        model=gauss_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optim.Adam(
            gauss_model.parameters(),
            lr=DUAL_LR,
            weight_decay=DUAL_WEIGHT_DECAY,
        ),
        epochs=DUAL_EPOCHS,
        device=device,
        save_path=str(save_gauss),
        early_stopping_patience=DUAL_EARLY_STOPPING,
    )

    plot_training_history(
        gauss_history,
        metric="loss",
        save_path=FIGURES_DIR / "gaussian_dual_branch_training_history_loss.png",
    )

    print(f"\nGaussian dual-branch model saved -> {save_gauss}")

    return {
        "results": results,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "deterministic_history": det_history,
        "gaussian_history": gauss_history,
        "market_idx": market_idx,
        "macro_idx": macro_idx,
    }




# =========================================================
# DC-LSTM (Deterministic + Gaussian)
# =========================================================
DC_HIDDEN_MARKET = 64
DC_HIDDEN_MACRO = 48
DC_HIDDEN_COUPLING = 48
DC_LAYERS_BRANCH = 1
DC_LAYERS_COUPLING = 1
DC_DROPOUT = 0.35
DC_GAUSSIAN_DROPOUT = 0.10
DC_LR = 1e-3
DC_WEIGHT_DECAY = 1e-5
DC_EPOCHS = 50
DC_BATCH_SIZE = 64
DC_EARLY_STOPPING = 5
DC_LOOKBACK = 30

BEST_DC_LSTM_MODEL_PATH = PROJECT_ROOT / "best_dc_lstm_model.pt"
BEST_GAUSSIAN_DC_LSTM_MODEL_PATH = PROJECT_ROOT / "best_gaussian_dc_lstm_model.pt"


class DCWrapper(nn.Module):
    """
    Routes (B, T, D_total) to DCLSTM by slicing
    market and macro feature indices separately.
    """
    def __init__(
        self,
        market_idx,
        macro_idx,
        hidden_dim_market,
        hidden_dim_macro,
        hidden_dim_coupling,
        num_layers_branch,
        num_layers_coupling,
        dropout,
    ):
        super().__init__()
        self.market_idx = market_idx
        self.macro_idx = macro_idx
        self.core = DCLSTM(
            market_input_dim=len(market_idx),
            macro_input_dim=len(macro_idx),
            hidden_dim_market=hidden_dim_market,
            hidden_dim_macro=hidden_dim_macro,
            hidden_dim_coupling=hidden_dim_coupling,
            num_layers_branch=num_layers_branch,
            num_layers_coupling=num_layers_coupling,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.core(x[:, :, self.market_idx], x[:, :, self.macro_idx])


class GaussianDCWrapper(nn.Module):
    """
    Routes (B, T, D_total) to GaussianDCLSTM.
    Returns (mean, log_var).
    """
    def __init__(
        self,
        market_idx,
        macro_idx,
        hidden_dim_market,
        hidden_dim_macro,
        hidden_dim_coupling,
        num_layers_branch,
        num_layers_coupling,
        dropout,
    ):
        super().__init__()
        self.market_idx = market_idx
        self.macro_idx = macro_idx
        self.core = GaussianDCLSTM(
            market_input_dim=len(market_idx),
            macro_input_dim=len(macro_idx),
            hidden_dim_market=hidden_dim_market,
            hidden_dim_macro=hidden_dim_macro,
            hidden_dim_coupling=hidden_dim_coupling,
            num_layers_branch=num_layers_branch,
            num_layers_coupling=num_layers_coupling,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor):
        return self.core(x[:, :, self.market_idx], x[:, :, self.macro_idx])


def run_dc_lstm_training(
    interim_csv: Path = INTERIM_SELECTED_PATH,
    save_det: Path = BEST_DC_LSTM_MODEL_PATH,
    save_gauss: Path = BEST_GAUSSIAN_DC_LSTM_MODEL_PATH,
):
    """
    Train deterministic and Gaussian DC-LSTM models.
    Randomness setup is expanded to match the teammate style.
    """
    import random as _random
    import os as _os
    import numpy as _np

    _random.seed(SEED)
    _os.environ["PYTHONHASHSEED"] = str(SEED)
    _np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    results = prepare_sequences_from_csv(
        csv_path=interim_csv,
        target_col="target",
        timestamp_col="Date",
        test_ratio=TEST_RATIO,
        scale_features=True,
        scale_target=False,
        scaler_type="standard",
        lookback=DC_LOOKBACK,
        train_csv_path=PROCESSED_DIR / "train_scaled.csv",
        test_csv_path=PROCESSED_DIR / "test_scaled.csv",
    )

    X_train = results["X_train"]
    y_train = results["y_train"]
    X_test = results["X_test"]
    y_test = results["y_test"]
    feature_cols = results["feature_cols"]

    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")

    groups = build_feature_groups(
        feature_cols,
        target_col="target",
        timestamp_col="Date",
        allow_other=True,
    )
    print_group_summary(groups)

    idx_map = groups_to_indices(groups, feature_cols)
    market_idx = idx_map["market_idx"] + idx_map["other_idx"]
    macro_idx = idx_map["macro_idx"]

    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        ),
        batch_size=DC_BATCH_SIZE,
        shuffle=True,
        generator=g,
        num_workers=0,
    )

    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32),
        ),
        batch_size=DC_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # -------------------------
    # Deterministic DC-LSTM
    # -------------------------
    print("\n" + "=" * 60)
    print("Training: Deterministic DC-LSTM")
    print("=" * 60)

    det_model = DCWrapper(
        market_idx=market_idx,
        macro_idx=macro_idx,
        hidden_dim_market=DC_HIDDEN_MARKET,
        hidden_dim_macro=DC_HIDDEN_MACRO,
        hidden_dim_coupling=DC_HIDDEN_COUPLING,
        num_layers_branch=DC_LAYERS_BRANCH,
        num_layers_coupling=DC_LAYERS_COUPLING,
        dropout=DC_DROPOUT,
    )

    det_history = train_model(
        model=det_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optim.Adam(
            det_model.parameters(),
            lr=DC_LR,
            weight_decay=DC_WEIGHT_DECAY,
        ),
        criterion=nn.SmoothL1Loss(beta=0.008),
        epochs=DC_EPOCHS,
        device=device,
        save_path=str(save_det),
        early_stopping_patience=DC_EARLY_STOPPING,
    )

    plot_training_history(
        det_history,
        metric="loss",
        save_path=FIGURES_DIR / "dc_lstm_training_history_loss.png",
    )

    print(f"\nDeterministic DC-LSTM saved -> {save_det}")

    # -------------------------
    # Gaussian DC-LSTM
    # -------------------------
    print("\n" + "=" * 60)
    print("Training: Gaussian DC-LSTM")
    print("=" * 60)

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

    gauss_model = GaussianDCWrapper(
        market_idx=market_idx,
        macro_idx=macro_idx,
        hidden_dim_market=DC_HIDDEN_MARKET,
        hidden_dim_macro=DC_HIDDEN_MACRO,
        hidden_dim_coupling=DC_HIDDEN_COUPLING,
        num_layers_branch=DC_LAYERS_BRANCH,
        num_layers_coupling=DC_LAYERS_COUPLING,
        dropout=DC_GAUSSIAN_DROPOUT,
    )

    gauss_history = train_model_gaussian(
        model=gauss_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optim.Adam(
            gauss_model.parameters(),
            lr=DC_LR,
            weight_decay=DC_WEIGHT_DECAY,
        ),
        epochs=DC_EPOCHS,
        device=device,
        save_path=str(save_gauss),
        early_stopping_patience=DC_EARLY_STOPPING,
    )

    plot_training_history(
        gauss_history,
        metric="loss",
        save_path=FIGURES_DIR / "gaussian_dc_lstm_training_history_loss.png",
    )

    print(f"\nGaussian DC-LSTM saved -> {save_gauss}")

    return {
        "results": results,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "deterministic_history": det_history,
        "gaussian_history": gauss_history,
        "market_idx": market_idx,
        "macro_idx": macro_idx,
    }




# =========================================================
# BiLSTM (Deterministic + Gaussian)
# =========================================================
from fx_forecasting.models.BiLSTM import BiLSTM
from fx_forecasting.models.GaussianBiLSTM import GaussianBiLSTM
from fx_forecasting.data.preprocess_sequence import prepare_sequences_from_csv as prepare_sequences_from_csv_src

BILSTM_LOOKBACK = 30
BILSTM_BATCH_SIZE = 64
BILSTM_EPOCHS = 50
BILSTM_EARLY_STOPPING = 5
BILSTM_TEST_RATIO = 0.2
BILSTM_HIDDEN_DIM = 32
BILSTM_NUM_LAYERS = 2
BILSTM_DROPOUT = 0.4
BILSTM_WEIGHT_DECAY = 5e-5
BILSTM_BASELINE_LR = 0.001
BILSTM_GAUSSIAN_LR = 0.0005
BILSTM_SEED = 37

BEST_BILSTM_MODEL_PATH = PROJECT_ROOT / "best_bilstm_model.pt"
BEST_GAUSSIAN_BILSTM_MODEL_PATH = PROJECT_ROOT / "best_gaussian_bilstm_model.pt"


def set_bilstm_seed(seed: int = BILSTM_SEED) -> None:
    """
    Keep the teammate's BiLSTM seed setup style.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True


def run_bilstm_training(
    interim_csv: Path = INTERIM_SELECTED_PATH,
    save_baseline: Path = BEST_BILSTM_MODEL_PATH,
    save_gauss: Path = BEST_GAUSSIAN_BILSTM_MODEL_PATH,
):
    """
    Train deterministic BiLSTM and Gaussian BiLSTM models.
    Adapted to fit the main project train.py structure.
    """
    set_bilstm_seed(BILSTM_SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    interim_path = Path(interim_csv)
    if not interim_path.exists():
        raise FileNotFoundError(f"Data file not found: {interim_path}")

    results = prepare_sequences_from_csv_src(
        csv_path=interim_path,
        target_col="target",
        timestamp_col="Date",
        test_ratio=BILSTM_TEST_RATIO,
        scale_features=True,
        scale_target=False,
        scaler_type="standard",
        lookback=BILSTM_LOOKBACK,
        train_csv_path=PROJECT_ROOT / "data" / "processed" / "train_scaled.csv",
        test_csv_path=PROJECT_ROOT / "data" / "processed" / "test_scaled.csv",
    )

    X_train = results["X_train"]
    y_train = results["y_train"]
    X_test = results["X_test"]
    y_test = results["y_test"]
    feature_cols = results["feature_cols"]

    input_dim = len(feature_cols)
    print(f"Input dimension: {input_dim}")
    print(f"Training set size: {X_train.shape} | Test set size: {X_test.shape}")

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=BILSTM_BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_test_t, y_test_t),
        batch_size=BILSTM_BATCH_SIZE,
        shuffle=False,
    )

    # -------------------------
    # Deterministic BiLSTM
    # -------------------------
    print("\n" + "=" * 50)
    print("--- Training Baseline BiLSTM ---")

    baseline_model = BiLSTM(
        input_dim=input_dim,
        hidden_dim=BILSTM_HIDDEN_DIM,
        num_layers=BILSTM_NUM_LAYERS,
        dropout=BILSTM_DROPOUT,
    ).to(device)

    baseline_optimizer = optim.Adam(
        baseline_model.parameters(),
        lr=BILSTM_BASELINE_LR,
        weight_decay=BILSTM_WEIGHT_DECAY,
    )
    criterion = nn.SmoothL1Loss(beta=0.008)

    bilstm_history = train_model(
        model=baseline_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=baseline_optimizer,
        criterion=criterion,
        epochs=BILSTM_EPOCHS,
        device=device,
        save_path=str(save_baseline),
        early_stopping_patience=BILSTM_EARLY_STOPPING,
    )

    plot_training_history(
        bilstm_history,
        metric="loss",
        save_path=FIGURES_DIR / "bilstm_training_history_loss.png",
    )

    print(f"Baseline model saved -> {save_baseline}")

    # -------------------------
    # Gaussian BiLSTM
    # -------------------------
    print("\n" + "=" * 50)
    print("--- Training Gaussian BiLSTM ---")

    gauss_model = GaussianBiLSTM(
        input_dim=input_dim,
        hidden_dim=BILSTM_HIDDEN_DIM,
        num_layers=BILSTM_NUM_LAYERS,
        dropout=BILSTM_DROPOUT,
        min_log_var=-14.0,
        max_log_var=-10.0,
    ).to(device)

    gauss_optimizer = optim.Adam(
        gauss_model.parameters(),
        lr=BILSTM_GAUSSIAN_LR,
        weight_decay=BILSTM_WEIGHT_DECAY,
    )

    gaussian_bilstm_history = train_model_gaussian(
        model=gauss_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=gauss_optimizer,
        epochs=BILSTM_EPOCHS,
        device=device,
        save_path=str(save_gauss),
        early_stopping_patience=BILSTM_EARLY_STOPPING,
    )

    plot_training_history(
        gaussian_bilstm_history,
        metric="loss",
        save_path=FIGURES_DIR / "gaussian_bilstm_training_history_loss.png",
    )

    print(f"Gaussian model saved -> {save_gauss}")
    print("\nAll BiLSTM training completed successfully.")

    return {
        "results": results,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "deterministic_history": bilstm_history,
        "gaussian_history": gaussian_bilstm_history,
        "input_dim": input_dim,
        "feature_cols": feature_cols,
    }




# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    set_seed(SEED)

    # Stage 1: feature engineering
    print("\n" + "=" * 60)
    print("Stage 1: Feature Engineering & Selection")
    print("=" * 60)
    df_final, production_features, df_train, df_test = build_interim_selected_dataset()

    print("\nSaved files:")
    print("Raw cache:", RAW_CACHE_PATH)
    print("Interim selected CSV:", INTERIM_SELECTED_PATH)
    print("Feature metadata:", FEATURE_METADATA_PATH)
    print("Figures folder:", FIGURES_DIR)

    print("\nFinal selected dataset shape:", df_final.shape)
    print("Final production feature count:", len(production_features))

    # Stage 2: baseline models
    print("\n" + "=" * 60)
    print("Stage 2: Baseline LSTM Training")
    print("=" * 60)
    baseline_outputs = run_baseline_training()

    # Stage 3: dual-branch models
    print("\n" + "=" * 60)
    print("Stage 3: Dual-Branch LSTM Training")
    print("=" * 60)
    dual_branch_outputs = run_dual_branch_training()

    # Stage 4: DC-LSTM models
    print("\n" + "=" * 60)
    print("Stage 4: DC-LSTM Training")
    print("=" * 60)
    dc_lstm_outputs = run_dc_lstm_training()

    # Stage 5: BiLSTM models
    print("\n" + "=" * 60)
    print("Stage 5: BiLSTM Training")
    print("=" * 60)
    bilstm_outputs = run_bilstm_training()

    print("\n" + "=" * 60)
    print("All training complete.")
    print(f"  Baseline det.      : {BEST_BASELINE_MODEL_PATH}")
    print(f"  Baseline Gaussian  : {BEST_GAUSSIAN_MODEL_PATH}")
    print(f"  Dual det.          : {BEST_DUAL_BRANCH_MODEL_PATH}")
    print(f"  Dual Gaussian      : {BEST_GAUSSIAN_DUAL_MODEL_PATH}")
    print(f"  DC det.            : {BEST_DC_LSTM_MODEL_PATH}")
    print(f"  DC Gaussian        : {BEST_GAUSSIAN_DC_LSTM_MODEL_PATH}")
    print(f"  BiLSTM det.        : {BEST_BILSTM_MODEL_PATH}")
    print(f"  BiLSTM Gaussian    : {BEST_GAUSSIAN_BILSTM_MODEL_PATH}")
    print("=" * 60)