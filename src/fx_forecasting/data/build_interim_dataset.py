from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


def clean_fx_data(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Basic cleaning for daily FX / macro data.

    - parse timestamp
    - sort by time
    - forward fill missing values only
    """
    df = df.copy()

    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df = df.dropna(subset=[timestamp_col])
    df = df.sort_values(timestamp_col).reset_index(drop=True)

    value_cols = [c for c in df.columns if c != timestamp_col]
    df[value_cols] = df[value_cols].ffill()

    return df


def infer_column_groups(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
) -> Dict[str, List[str]]:
    all_cols = [c for c in df.columns if c != timestamp_col]

    fx_cols = [c for c in all_cols if c.startswith("GBP-")]
    price_like_cols = [
        c for c in all_cols
        if c in ["ftse100_Close", "gold_price", "DCOILBRENTEU", "DTWEXBGS"]
    ]
    rate_cols = [
        c for c in all_cols
        if c in ["DGS10", "glc_nominal_2", "glc_nominal_10", "ois_1", "ois_2"]
    ]
    vol_cols = [c for c in all_cols if c in ["vix_daily_close"]]

    other_cols = [
        c for c in all_cols
        if c not in set(fx_cols + price_like_cols + rate_cols + vol_cols)
    ]

    return {
        "fx": fx_cols,
        "price_like": price_like_cols,
        "rates": rate_cols,
        "vol": vol_cols,
        "other": other_cols,
    }


def add_log_returns(
    df: pd.DataFrame,
    cols: Sequence[str],
) -> pd.DataFrame:
    df = df.copy()

    for col in cols:
        s = df[col].astype(float)
        df[f"{col}_ret"] = np.log(s / s.shift(1))

    return df


def add_diff_features(
    df: pd.DataFrame,
    cols: Sequence[str],
) -> pd.DataFrame:
    df = df.copy()

    for col in cols:
        df[f"{col}_diff"] = df[col].astype(float).diff(1)

    return df


def add_lag_features(
    df: pd.DataFrame,
    cols: Sequence[str],
    lags: Sequence[int] = (1, 2, 3, 5, 10),
) -> pd.DataFrame:
    df = df.copy()

    for col in cols:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    return df


def add_rolling_mean_features(
    df: pd.DataFrame,
    cols: Sequence[str],
    windows: Sequence[int] = (5, 20, 60),
    shift_by: int = 1,
) -> pd.DataFrame:
    """
    Leakage-safe rolling means:
    feature at t uses information only up to t-1.
    """
    df = df.copy()

    for col in cols:
        for w in windows:
            df[f"{col}_ma{w}"] = df[col].rolling(w).mean().shift(shift_by)

    return df


def add_rolling_volatility_features(
    df: pd.DataFrame,
    cols: Sequence[str],
    windows: Sequence[int] = (5, 20, 60),
    shift_by: int = 1,
) -> pd.DataFrame:
    df = df.copy()

    for col in cols:
        for w in windows:
            df[f"{col}_vol{w}"] = df[col].rolling(w).std().shift(shift_by)

    return df


def add_spread_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if {"glc_nominal_10", "glc_nominal_2"}.issubset(df.columns):
        df["glc_curve_slope"] = df["glc_nominal_10"] - df["glc_nominal_2"]

    if {"ois_2", "ois_1"}.issubset(df.columns):
        df["ois_slope"] = df["ois_2"] - df["ois_1"]

    return df


def add_calendar_features(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    df = df.copy()

    dt = pd.to_datetime(df[timestamp_col])
    df["day_of_week"] = dt.dt.dayofweek
    df["month"] = dt.dt.month

    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


def add_forward_target(
    df: pd.DataFrame,
    base_target_col: str,
    horizon: int = 1,
    target_type: str = "return",
) -> Tuple[pd.DataFrame, str]:
    """
    target_type:
    - 'return': predict next-period return
    - 'level': predict next-period level
    - 'existing_series': shift an already-created series forward
    """
    df = df.copy()

    if target_type not in {"return", "level", "existing_series"}:
        raise ValueError("target_type must be one of {'return', 'level', 'existing_series'}")

    if target_type == "return":
        if base_target_col.endswith("_ret"):
            current_ret_col = base_target_col
        else:
            current_ret_col = f"{base_target_col}_ret"
            if current_ret_col not in df.columns:
                df[current_ret_col] = np.log(df[base_target_col] / df[base_target_col].shift(1))

        target_col = f"{current_ret_col}_fwd{horizon}"
        df[target_col] = df[current_ret_col].shift(-horizon)

    elif target_type == "level":
        target_col = f"{base_target_col}_fwd{horizon}"
        df[target_col] = df[base_target_col].shift(-horizon)

    else:
        target_col = f"{base_target_col}_fwd{horizon}"
        df[target_col] = df[base_target_col].shift(-horizon)

    return df, target_col

from typing import Optional


def _parse_suffix_number(col: str, token: str) -> Optional[int]:
    """
    Examples:
    col='GBP-USD_ret_lag5', token='_lag' -> 5
    col='GBP-USD_ret_ma20', token='_ma' -> 20
    col='GBP-USD_ret_vol60', token='_vol' -> 60
    """
    if token not in col:
        return None
    try:
        return int(col.split(token)[-1])
    except ValueError:
        return None


def _match_prefix(col: str, prefixes: Sequence[str]) -> Optional[str]:
    """
    Return which base prefix a column belongs to.
    """
    for p in prefixes:
        if col == p:
            return p
        if col.startswith(p + "_lag"):
            return p
        if col.startswith(p + "_ma"):
            return p
        if col.startswith(p + "_vol"):
            return p
    return None


def select_feature_columns(
    feature_cols: Sequence[str],
    feature_mode: str = "full_selected",
    keep_lags: Sequence[int] = (1, 3, 5),
    keep_mas: Sequence[int] = (5, 20),
    keep_vols: Sequence[int] = (5, 20),
    include_calendar_in_full: bool = True,
) -> List[str]:
    """
    Two-stage feature selection:

    1) Coarse filtering:
       keep only lag1/3/5, ma5/20, vol5/20

    2) Group ablation:
       - ar_only
       - ar_macro
       - ar_macro_risk
       - ar_macro_risk_fx
       - full_selected
    """

    # Define feature groups based on column name prefixes
    ar_prefixes = [
        "GBP-USD_ret",
    ]

    macro_prefixes = [
        "glc_curve_slope",
        "ois_slope",
        "DGS10_diff",
        "glc_nominal_2_diff",
        "glc_nominal_10_diff",
        "ois_1_diff",
        "ois_2_diff",
        "glc_curve_slope_diff",
        "ois_slope_diff",
    ]

    risk_prefixes = [
        "DCOILBRENTEU_ret",
        "vix_daily_close_ret",
    ]

    fx_prefixes = [
        "GBP-CNY_ret",
        "GBP-EUR_ret",
        "GBP-JPY_ret",
        "GBP-KRW_ret",
        "GBP-CHF_ret",
    ]

    market_prefixes = [
        "ftse100_Close_ret",
        "gold_price_ret",
        "DTWEXBGS_ret",
    ]

    calendar_cols = [
        "day_of_week",
        "month",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
    ]

    all_prefixes = (
        ar_prefixes
        + macro_prefixes
        + risk_prefixes
        + fx_prefixes
        + market_prefixes
    )

    group_map = {
        "ar_only": set(ar_prefixes),
        "ar_macro": set(ar_prefixes + macro_prefixes),
        "ar_macro_risk": set(ar_prefixes + macro_prefixes + risk_prefixes),
        "ar_macro_risk_fx": set(ar_prefixes + macro_prefixes + risk_prefixes + fx_prefixes),
        "full_selected": set(all_prefixes),
    }

    if feature_mode not in group_map:
        raise ValueError(
            f"Unknown feature_mode '{feature_mode}'. "
            f"Choose from {list(group_map.keys())}"
        )

    active_prefixes = group_map[feature_mode]

    selected = []

    for col in feature_cols:
        # calendar features: keep only in full_selected
        if col in calendar_cols:
            if feature_mode == "full_selected" and include_calendar_in_full:
                selected.append(col)
            continue

        matched_prefix = _match_prefix(col, all_prefixes)
        if matched_prefix is None:
            # ignore anything outside our defined groups
            continue

        if matched_prefix not in active_prefixes:
            continue

        # raw base feature
        if col == matched_prefix:
            selected.append(col)
            continue

        # lag
        lag_n = _parse_suffix_number(col, "_lag")
        if lag_n is not None:
            if lag_n in keep_lags:
                selected.append(col)
            continue

        # moving average
        ma_n = _parse_suffix_number(col, "_ma")
        if ma_n is not None:
            if ma_n in keep_mas:
                selected.append(col)
            continue

        # volatility
        vol_n = _parse_suffix_number(col, "_vol")
        if vol_n is not None:
            if vol_n in keep_vols:
                selected.append(col)
            continue

    return selected



def build_interim_dataset(
    df: pd.DataFrame,
    target_base_col: str,
    timestamp_col: str = "timestamp",
    horizon: int = 1,
    target_type: str = "return",
    return_lags: Sequence[int] = (1, 2, 3, 5, 10),
    rolling_windows: Sequence[int] = (5, 20, 60),
    include_calendar: bool = True,
    feature_mode: str = "full_selected",
    keep_lags: Sequence[int] = (1, 3, 5),
    keep_mas: Sequence[int] = (5, 20),
    keep_vols: Sequence[int] = (5, 20),
) -> Tuple[pd.DataFrame, str, List[str]]:
    """
    Build leakage-safe tabular dataset and return:
    - interim_df
    - target_col
    - feature_cols
    """
    df = clean_fx_data(df, timestamp_col=timestamp_col)
    df = add_spread_features(df)

    groups = infer_column_groups(df, timestamp_col=timestamp_col)

    ret_source_cols = groups["fx"] + groups["price_like"] + groups["vol"]
    ret_source_cols = [c for c in ret_source_cols if c in df.columns]
    if ret_source_cols:
        df = add_log_returns(df, ret_source_cols)

    diff_source_cols = [c for c in groups["rates"] if c in df.columns]
    if diff_source_cols:
        df = add_diff_features(df, diff_source_cols)

    spread_cols = [c for c in ["glc_curve_slope", "ois_slope"] if c in df.columns]
    if spread_cols:
        df = add_diff_features(df, spread_cols)

    signal_cols = []
    signal_cols += [f"{c}_ret" for c in ret_source_cols if f"{c}_ret" in df.columns]
    signal_cols += [f"{c}_diff" for c in diff_source_cols if f"{c}_diff" in df.columns]
    signal_cols += [f"{c}_diff" for c in spread_cols if f"{c}_diff" in df.columns]

    if signal_cols:
        df = add_lag_features(df, cols=signal_cols, lags=return_lags)
        df = add_rolling_mean_features(df, cols=signal_cols, windows=rolling_windows, shift_by=1)
        df = add_rolling_volatility_features(df, cols=signal_cols, windows=rolling_windows, shift_by=1)

    if include_calendar:
        df = add_calendar_features(df, timestamp_col=timestamp_col)

    df, target_col = add_forward_target(
        df,
        base_target_col=target_base_col,
        horizon=horizon,
        target_type=target_type,
    )

    exclude_cols = {timestamp_col, target_col}
    raw_level_cols_to_exclude = set(
        groups["fx"] + groups["price_like"] + groups["vol"] + groups["rates"]
    )

    feature_cols = []
    for c in df.columns:
        if c in exclude_cols:
            continue
        if c in raw_level_cols_to_exclude:
            continue
        feature_cols.append(c)

    # NEW: coarse filter + group ablation
    feature_cols = select_feature_columns(
        feature_cols=feature_cols,
        feature_mode=feature_mode,
        keep_lags=keep_lags,
        keep_mas=keep_mas,
        keep_vols=keep_vols,
    )

    keep_cols = [timestamp_col] + feature_cols + [target_col]
    interim_df = df[keep_cols].dropna().reset_index(drop=True)

    print(f"Using feature_mode: {feature_mode}")
    print(f"Selected feature count: {len(feature_cols)}")

    return interim_df, target_col, feature_cols


def save_interim_dataset(
    interim_df: pd.DataFrame,
    output_csv_path: str | Path,
) -> Path:
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    interim_df.to_csv(output_csv_path, index=False)
    return output_csv_path


if __name__ == "__main__":
    # Example script usage:
    raw_path = "data/raw/fx_merged.csv"
    output_path = "data/interim/fx_interim.csv"

    df_raw = pd.read_csv(raw_path)

    interim_df, target_col, feature_cols = build_interim_dataset(
        df=df_raw,
        target_base_col="GBP-USD",
        timestamp_col="timestamp",
        horizon=1,
        target_type="return",
        return_lags=(1, 2, 3, 5, 10),
        rolling_windows=(5, 20, 60),
        include_calendar=True,
    )

    save_interim_dataset(interim_df, output_path)

    print(f"Saved interim dataset to: {output_path}")
    print(f"Shape: {interim_df.shape}")
    print(f"Target column: {target_col}")
    print(f"Number of features: {len(feature_cols)}")