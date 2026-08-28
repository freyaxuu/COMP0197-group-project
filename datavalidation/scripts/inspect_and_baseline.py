from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# 1) Data loading: local path OR remote URL
def _read_table(path_or_url: str) -> pd.DataFrame:
    """Read csv/xlsx/parquet from local path or http(s) URL. Supports csv.zip."""
    is_url = path_or_url.startswith("http://") or path_or_url.startswith("https://")

    # Decide by suffix (works for URLs too)
    s = path_or_url.lower()
    if s.endswith(".csv.zip") or s.endswith(".zip"):
        # assume zipped csv
        return pd.read_csv(path_or_url, compression="zip")
    if s.endswith(".csv"):
        return pd.read_csv(path_or_url)
    if (not is_url) and (s.endswith(".xlsx") or s.endswith(".xls")):
        return pd.read_excel(path_or_url)
    if (not is_url) and s.endswith(".parquet"):
        return pd.read_parquet(path_or_url)

    # fallback: try csv (some raw URLs don't end with .csv)
    return pd.read_csv(path_or_url)


def load_any_table(
    path_or_url: str,
    timestamp_col: str = "timestamp",
    target_col: Optional[str] = None,
    long_table: bool = False,
    long_var_col: str = "variable",
    long_value_col: str = "value",
    # build timestamp from separate columns (e.g., year/month/day/hour)
    make_timestamp_from_ymdh: bool = False,
    y_col: str = "year",
    m_col: str = "month",
    d_col: str = "day",
    h_col: str = "hour",
    downsample_step: int = 1,   # e.g., 6 for 10-min -> hourly
) -> Tuple[pd.DataFrame, str]:
    """
    Returns:
      df_wide: columns = [timestamp, numeric features..., target]
      inferred_target_col
    """
    df = _read_table(path_or_url)

    # optional downsample by row step (simple but effective for fixed-frequency data)
    if downsample_step > 1:
        df = df.iloc[::downsample_step].reset_index(drop=True)

    if long_table:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df_wide = df.pivot_table(
            index=timestamp_col, columns=long_var_col, values=long_value_col, aggfunc="mean"
        ).reset_index()
    else:
        df_wide = df.copy()

        # build timestamp if needed
        if make_timestamp_from_ymdh:
            for c in [y_col, m_col, d_col, h_col]:
                if c not in df_wide.columns:
                    raise ValueError(f"make_timestamp_from_ymdh=True but missing column: {c}")
            df_wide[timestamp_col] = pd.to_datetime(
                df_wide[[y_col, m_col, d_col, h_col]].rename(
                    columns={y_col: "year", m_col: "month", d_col: "day", h_col: "hour"}
                )
            )

        # parse existing timestamp column
        if timestamp_col not in df_wide.columns:
            raise ValueError(
                f"timestamp_col '{timestamp_col}' not found. "
                f"If your data has year/month/day/hour, use --make_timestamp_from_ymdh."
            )
        df_wide[timestamp_col] = pd.to_datetime(df_wide[timestamp_col])

    df_wide = df_wide.sort_values(timestamp_col).reset_index(drop=True)

    # keep numeric columns only (except timestamp)
    numeric_cols = [
        c for c in df_wide.columns
        if c != timestamp_col and pd.api.types.is_numeric_dtype(df_wide[c])
    ]
    df_wide = df_wide[[timestamp_col] + numeric_cols]

    if len(numeric_cols) < 2:
        raise ValueError("Need at least 2 numeric columns (features + target).")

    if target_col is None:
        target_col = numeric_cols[-1]  # default: last numeric column

    if target_col not in df_wide.columns:
        raise ValueError(f"target_col '{target_col}' not in columns: {df_wide.columns.tolist()}")

    return df_wide, target_col


def basic_clean(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    fill_method: str = "ffill",
    dropna_threshold: float = 0.5,
) -> pd.DataFrame:
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
        raise ValueError("fill_method must be ffill/bfill/interpolate")

    return df


# 2) Inspection plots
def plot_missingness(df: pd.DataFrame, timestamp_col: str, title: str, save_path: str):
    cols = [c for c in df.columns if c != timestamp_col]
    miss = df[cols].isna().mean().sort_values(ascending=False)

    plt.figure(figsize=(10, 4))
    plt.bar(miss.index, miss.values)
    plt.title(f"{title} - missing ratio per column")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_corr_heatmap(df: pd.DataFrame, timestamp_col: str, title: str, save_path: str):
    cols = [c for c in df.columns if c != timestamp_col]
    corr = df[cols].corr().values

    plt.figure(figsize=(6, 5))
    plt.imshow(corr, aspect="auto")
    plt.colorbar()
    plt.title(f"{title} - correlation heatmap")
    plt.xticks(range(len(cols)), cols, rotation=45, ha="right")
    plt.yticks(range(len(cols)), cols)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_timeseries(df: pd.DataFrame, timestamp_col: str, cols: List[str], title: str, save_path: str):
    plt.figure(figsize=(10, 4))
    for c in cols:
        plt.plot(df[timestamp_col], df[c], label=c, linewidth=1)
    plt.legend()
    plt.title(title)
    plt.xlabel("time")
    plt.ylabel("value")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# 3) Window dataset for baseline LSTM
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
        x_win = self.x[t:t + L]           # [L, D]
        y_win = self.y[t + L:t + L + H]   # [H, 1]
        return torch.from_numpy(x_win), torch.from_numpy(y_win)


def time_split_indices(n: int, train_ratio=0.7, val_ratio=0.15):
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    idx_train = np.arange(0, n_train)
    idx_val = np.arange(n_train, n_train + n_val)
    idx_test = np.arange(n_train + n_val, n)
    return idx_train, idx_val, idx_test


# 4) Baselines: persistence & deterministic LSTM
def persistence_mae(y: np.ndarray) -> float:
    y_true = y[1:]
    y_pred = y[:-1]
    return float(np.mean(np.abs(y_true - y_pred)))


class DeterministicLSTM(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, num_layers: int = 2, horizon: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x):  # [B, L, D]
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        yhat = self.head(last).unsqueeze(-1)  # [B, H, 1]
        return yhat


def train_det_lstm(
    train_loader: DataLoader,
    val_loader: DataLoader,
    in_dim: int,
    horizon: int,
    device: str,
    epochs: int,
    lr: float = 1e-3,
):
    model = DeterministicLSTM(in_dim=in_dim, horizon=horizon).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        tr_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_losses.append(loss.item())

        model.eval()
        va_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                va_losses.append(loss.item())

        tr = float(np.mean(tr_losses)) if tr_losses else float("nan")
        va = float(np.mean(va_losses)) if va_losses else float("nan")
        print(f"[DetLSTM] epoch {ep:02d} train={tr:.6f} val={va:.6f}")

        if va < best_val:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def main():
    ap = argparse.ArgumentParser()

    # data options
    ap.add_argument("--data", type=str, required=True,
                    help="local path or http(s) URL to csv/xlsx/parquet/csv.zip")
    ap.add_argument("--timestamp_col", type=str, default="timestamp")
    ap.add_argument("--target_col", type=str, default=None)
    ap.add_argument("--long_table", action="store_true")

    # optional: build timestamp from year/month/day/hour (for some datasets like pollution.csv)
    ap.add_argument("--make_timestamp_from_ymdh", action="store_true",
                    help="create timestamp from year/month/day/hour columns")
    ap.add_argument("--y_col", type=str, default="year")
    ap.add_argument("--m_col", type=str, default="month")
    ap.add_argument("--d_col", type=str, default="day")
    ap.add_argument("--h_col", type=str, default="hour")

    # optional: downsample by row step (e.g., 6 for 10-min -> hourly)
    ap.add_argument("--downsample_step", type=int, default=1)

    # model/window
    ap.add_argument("--lookback", type=int, default=48)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--plot_cols", type=int, default=6)
    ap.add_argument("--plot_first_n", type=int, default=200)

    ap.add_argument("--out_dir", type=str, default="results")
    args = ap.parse_args()

    df, target_col = load_any_table(
        args.data,
        timestamp_col=args.timestamp_col,
        target_col=args.target_col,
        long_table=args.long_table,
        make_timestamp_from_ymdh=args.make_timestamp_from_ymdh,
        y_col=args.y_col, m_col=args.m_col, d_col=args.d_col, h_col=args.h_col,
        downsample_step=args.downsample_step,
    )
    df = basic_clean(df, timestamp_col=args.timestamp_col, fill_method="ffill", dropna_threshold=0.5)

    cols = [c for c in df.columns if c != args.timestamp_col]
    print("Columns:", cols)
    print("Target:", target_col)
    print("Datapoints:", len(df))

    # output directory
    import os, json
    os.makedirs(args.out_dir, exist_ok=True)

    # inspection plots (save)
    plot_missingness(
    df,
    args.timestamp_col,
    title="dataset",
    save_path=os.path.join(args.out_dir, "missing_ratio.png"),
    )

    # define which columns are "real variables" (exclude time index columns)
    drop_cols = {"No", "year", "month", "day", "hour"}
    cols_for_corr = [c for c in cols if (c not in drop_cols)] 

    # correlation heatmap WITHOUT No/year/month/day/hour
    plot_corr_heatmap(
        df[[args.timestamp_col] + cols_for_corr],
        args.timestamp_col,
        title="dataset (corr without time index cols)",
        save_path=os.path.join(args.out_dir, "corr_heatmap.png"),
    )

    # timeseries
    plot_timeseries(
    df,
    args.timestamp_col,
    [target_col],  # only plot target
    title=f"dataset timeseries (target only): {target_col}",
    save_path=os.path.join(args.out_dir, "timeseries_target.png"),
    )

    # Persistence
    y_raw = df[[target_col]].values
    print("Persistence MAE:", persistence_mae(y_raw))

    # Deterministic LSTM
    feature_cols = [c for c in cols if c != target_col and c not in drop_cols]
    x = df[feature_cols].values
    y = df[[target_col]].values

    # standardize
    x_mean, x_std = x.mean(axis=0, keepdims=True), x.std(axis=0, keepdims=True) + 1e-8
    y_mean, y_std = y.mean(axis=0, keepdims=True), y.std(axis=0, keepdims=True) + 1e-8
    xz = (x - x_mean) / x_std
    yz = (y - y_mean) / y_std

    cfg = WindowConfig(lookback=args.lookback, horizon=args.horizon, stride=1)
    ds = WindowDataset(xz, yz, cfg)

    if len(ds) == 0:
        raise ValueError(
            f"No windows can be formed: T={len(xz)}, lookback={args.lookback}, horizon={args.horizon}. "
            f"Reduce lookback/horizon or use a longer dataset."
        )

    idx_train, idx_val, idx_test = time_split_indices(len(ds), 0.7, 0.15)
    train_loader = DataLoader(torch.utils.data.Subset(ds, idx_train), batch_size=64, shuffle=False)
    val_loader = DataLoader(torch.utils.data.Subset(ds, idx_val), batch_size=64, shuffle=False)
    test_loader = DataLoader(torch.utils.data.Subset(ds, idx_test), batch_size=64, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = train_det_lstm(train_loader, val_loader, in_dim=xz.shape[1],
                           horizon=args.horizon, device=device, epochs=args.epochs)

    # --- predict over the whole test split (1-step from each window)
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for xb, yb in test_loader:
            pred = model(xb.to(device)).cpu().numpy()  # [B, H, 1]
            preds.append(pred[:, 0, 0])               # step-1
            trues.append(yb.numpy()[:, 0, 0])

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    # invert z-score to original scale
    preds_y = preds * y_std[0, 0] + y_mean[0, 0]
    trues_y = trues * y_std[0, 0] + y_mean[0, 0]

    # metrics
    errors = trues_y - preds_y
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))

    metrics = {"RMSE": rmse, "MAE": mae}
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("Test RMSE:", rmse)
    print("Test MAE:", mae)

    # plot: full test curve
    plt.figure(figsize=(10, 4))
    plt.plot(trues_y, label="true (test 1-step)")
    plt.plot(preds_y, label="pred (test 1-step)")
    plt.title("Deterministic LSTM - 1-step forecast over the whole test split")
    plt.legend()
    plt.tight_layout()
    p0 = os.path.join(args.out_dir, "det_test_curve.png")
    plt.savefig(p0, dpi=150)
    plt.close()
    print(f"[saved plot] {p0}")

    # plot: Actual vs Predicted (first N hours)
    N = min(args.plot_first_n, len(trues_y))
    plt.figure(figsize=(12, 4))
    plt.plot(trues_y[:N], label="Actual", linewidth=1)
    plt.plot(preds_y[:N], label="Predicted", linewidth=1, linestyle="--")
    plt.title(f"Deterministic Baseline: Actual vs Predicted (First {N} steps)")
    plt.legend()
    plt.tight_layout()
    p1 = os.path.join(args.out_dir, "det_actual_vs_pred_first200.png")
    plt.savefig(p1, dpi=150)
    plt.close()
    print(f"[saved plot] {p1}")

    # plot: Residual Distribution
    plt.figure(figsize=(8, 4))
    plt.hist(errors, bins=50)
    plt.title("Residual Distribution (Prediction Errors)")
    plt.xlabel("Error (y_true - y_pred)")
    plt.ylabel("Count")
    plt.tight_layout()
    p2 = os.path.join(args.out_dir, "det_residual_hist.png")
    plt.savefig(p2, dpi=150)
    plt.close()
    print(f"[saved plot] {p2}")


if __name__ == "__main__":
    main()