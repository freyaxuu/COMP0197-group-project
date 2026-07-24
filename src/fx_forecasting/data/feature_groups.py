from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence


# -------------------------
# 1) Hard-coded groups
# -------------------------
MARKET_FEATURES: List[str] = [
    "vix_daily_close_ret",
    "vix_daily_close_ret_ma20",
    "gold_price_ret",
    "DTWEXBGS_ret",
    "GBP-USD_ret",
    "GBP-USD_ret_vol20",
    "GBP-JPY_ret",
    "GBP-CHF_ret",
]

MACRO_FEATURES: List[str] = [
    "DGS10_diff",
    "DGS10_diff_ma20",
    "DGS10_diff_vol20",
    "glc_nominal_10_diff_ma20",
    "glc_nominal_10_diff_vol20",
    "glc_nominal_10_diff_lag1",
]


@dataclass(frozen=True)
class FeatureGroups:
    market: List[str]
    macro: List[str]
    other: List[str]


# -------------------------
# 2) Utilities
# -------------------------
def build_feature_groups(
    feature_cols: Sequence[str],
    *,
    target_col: str | None = "target",
    timestamp_col: str | None = "Date",
    allow_other: bool = True,
) -> FeatureGroups:
    """
    Given the feature_cols used for modeling (i.e. columns after preprocessing),
    return FeatureGroups with (market, macro, other).

    - Validates that all specified MARKET_FEATURES / MACRO_FEATURES exist in feature_cols.
    - If allow_other=True, any remaining feature columns are put into 'other'.
    """
    cols = list(feature_cols)

    exclude = set()
    if target_col:
        exclude.add(target_col)
    if timestamp_col:
        exclude.add(timestamp_col)

    cols_wo_exclude = [c for c in cols if c not in exclude]

    _validate_feature_presence(cols_wo_exclude, MARKET_FEATURES, group_name="market")  # warn only
    _validate_feature_presence(cols_wo_exclude, MACRO_FEATURES, group_name="macro")   # warn only

    market = [c for c in MARKET_FEATURES if c in cols_wo_exclude]
    macro = [c for c in MACRO_FEATURES if c in cols_wo_exclude]

    used = set(market + macro)
    other = [c for c in cols_wo_exclude if c not in used] if allow_other else []

    return FeatureGroups(market=market, macro=macro, other=other)


def groups_to_indices(groups: FeatureGroups, feature_cols: Sequence[str]) -> Dict[str, List[int]]:
    """
    Convert grouped column names into indices for slicing X[:, :, idx].
    feature_cols must match the feature ordering used to build X.
    """
    idx = {c: i for i, c in enumerate(feature_cols)}

    def to_idx(cols: Sequence[str]) -> List[int]:
        missing = [c for c in cols if c not in idx]
        if missing:
            raise ValueError(f"Columns not found in feature_cols: {missing}")
        return [idx[c] for c in cols]

    return {
        "market_idx": to_idx(groups.market),
        "macro_idx": to_idx(groups.macro),
        "other_idx": to_idx(groups.other),
    }


def print_group_summary(groups: FeatureGroups) -> None:
    print(f"[FeatureGroups] market ({len(groups.market)}): {groups.market}")
    print(f"[FeatureGroups] macro  ({len(groups.macro)}): {groups.macro}")
    print(f"[FeatureGroups] other  ({len(groups.other)}): {groups.other}")


# -------------------------
# 3) Internal helpers
# -------------------------
def _validate_feature_presence(
    available_cols: Sequence[str],
    required_cols: Sequence[str],
    *,
    group_name: str,
) -> None:
    missing = [c for c in required_cols if c not in available_cols]
    if missing:
        print(
            f"[FeatureGroups][WARN] Missing {group_name} features (will be ignored): {missing}"
        )