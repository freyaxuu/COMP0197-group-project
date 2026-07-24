# Generative AI (GenAI) Usage Statement:
# Generative AI tools were used to assist in generating annotations and
# docstrings, as well as improving code formatting and readability.
# The design, implementation, and validation of the code remain the
# authors' own work.

import os
import sys
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

# -----------------------------------------------------------------------------
# Seed
# -----------------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed(42)

# -----------------------------------------------------------------------------
# Evaluation imports
# -----------------------------------------------------------------------------
from fx_forecasting.evaluation.evaluate import (
    evaluate_model,
    evaluate_model_mc,
    evaluate_model_gaussian,
    evaluate_model_mc_gaussian,
)

# -----------------------------------------------------------------------------
# Model imports
# -----------------------------------------------------------------------------
from fx_forecasting.models.baseline import DeterministicLSTM, GaussianLSTM
from fx_forecasting.models.dc_lstm import DCLSTM, GaussianDCLSTM
from fx_forecasting.models.BiLSTM import BiLSTM
from fx_forecasting.models.GaussianBiLSTM import GaussianBiLSTM
from fx_forecasting.models.dual_branch import DualBranchLSTM, GaussianDualBranchLSTM

# -----------------------------------------------------------------------------
# Shared preprocessing imports
# -----------------------------------------------------------------------------
from fx_forecasting.data.preprocess_sequence import prepare_sequences_from_csv
from fx_forecasting.data.feature_groups import (
    build_feature_groups,
    groups_to_indices,
    print_group_summary,
)

# -----------------------------------------------------------------------------
# Device
# -----------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MC_SAMPLES = 200

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
raw_path = PROJECT_ROOT / "data" / "raw" / "fx_merged.csv"
interim_path = PROJECT_ROOT / "data" / "interim" / "interim_selected.csv"

df_raw = pd.read_csv(raw_path)

# -----------------------------------------------------------------------------
# Wrapper classes copied from train.py
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Prepare TWO data versions
# baseline / dc / dual -> lookback=30
# bilstm -> lookback=10
# -----------------------------------------------------------------------------
common_data_dict = prepare_sequences_from_csv(
    csv_path=interim_path,
    target_col="target",
    timestamp_col="Date",
    test_ratio=0.2,
    scale_features=True,
    scale_target=False,
    scaler_type="standard",
    lookback=30,
)

bilstm_data_dict = prepare_sequences_from_csv(
    csv_path=interim_path,
    target_col="target",
    timestamp_col="Date",
    test_ratio=0.2,
    scale_features=True,
    scale_target=False,
    scaler_type="standard",
    lookback=10,
)

# -----------------------------------------------------------------------------
# Common data (baseline / dc / dual)
# -----------------------------------------------------------------------------
X_train_common = common_data_dict["X_train"]
y_train_common = common_data_dict["y_train"]
X_test_common = common_data_dict["X_test"]
y_test_common = common_data_dict["y_test"]

train_df_common = common_data_dict["train_df"]
test_df_common = common_data_dict["test_df"]
feature_cols_common = common_data_dict["feature_cols"]

X_test_common_t = torch.tensor(X_test_common, dtype=torch.float32)
y_test_common_t = torch.tensor(y_test_common, dtype=torch.float32)

common_test_dataset = TensorDataset(X_test_common_t, y_test_common_t)
common_test_loader = DataLoader(common_test_dataset, batch_size=64, shuffle=False)

# -----------------------------------------------------------------------------
# BiLSTM data (lookback=10)
# -----------------------------------------------------------------------------
X_train_bilstm = bilstm_data_dict["X_train"]
y_train_bilstm = bilstm_data_dict["y_train"]
X_test_bilstm = bilstm_data_dict["X_test"]
y_test_bilstm = bilstm_data_dict["y_test"]

train_df_bilstm = bilstm_data_dict["train_df"]
test_df_bilstm = bilstm_data_dict["test_df"]
feature_cols_bilstm = bilstm_data_dict["feature_cols"]

X_test_bilstm_t = torch.tensor(X_test_bilstm, dtype=torch.float32)
y_test_bilstm_t = torch.tensor(y_test_bilstm, dtype=torch.float32)

bilstm_test_dataset = TensorDataset(X_test_bilstm_t, y_test_bilstm_t)
bilstm_test_loader = DataLoader(bilstm_test_dataset, batch_size=64, shuffle=False)

# -----------------------------------------------------------------------------
# Feature groups from common feature cols (used by dc / dual)
# -----------------------------------------------------------------------------
feature_groups = build_feature_groups(
    feature_cols_common,
    target_col="target",
    timestamp_col="Date",
    allow_other=True,
)
print_group_summary(feature_groups)

idx_map = groups_to_indices(feature_groups, feature_cols_common)

market_idx = idx_map["market_idx"]
macro_idx = idx_map["macro_idx"]
other_idx = idx_map.get("other_idx", [])

# Fold "other" into market branch
market_idx = market_idx + other_idx

print("market dim:", len(market_idx))
print("macro dim:", len(macro_idx))

# -----------------------------------------------------------------------------
# Instantiate all 8 models
# -----------------------------------------------------------------------------
deterministic_models = {
    "baseline_det": DeterministicLSTM(
        input_dim=X_train_common.shape[-1],
        hidden_dim=32,
        num_layers=2,
        dropout=0.4,
    ),
    "dc_det": DCWrapper(
        market_idx=market_idx,
        macro_idx=macro_idx,
        hidden_dim_market=64,
        hidden_dim_macro=48,
        hidden_dim_coupling=48,
        num_layers_branch=1,
        num_layers_coupling=1,
        dropout=0.35,
    ),
    "bilstm_det": BiLSTM(
        input_dim=X_train_bilstm.shape[-1],
        hidden_dim=32,
        num_layers=2,
        dropout=0.4,
    ),
    "dual_det": DualBranchWrapper(
        market_idx=market_idx,
        macro_idx=macro_idx,
        hidden_dim=48,
        num_layers=1,
        dropout=0.4,
    ),
}

gaussian_models = {
    "baseline_gauss": GaussianLSTM(
        input_dim=X_train_common.shape[-1],
        hidden_dim=32,
        num_layers=2,
        dropout=0.10,
        min_log_var=-14.0,
        max_log_var=-10.0,
    ),
    "dc_gauss": GaussianDCWrapper(
        market_idx=market_idx,
        macro_idx=macro_idx,
        hidden_dim_market=64,
        hidden_dim_macro=48,
        hidden_dim_coupling=48,
        num_layers_branch=1,
        num_layers_coupling=1,
        dropout=0.10,
    ),
    "bilstm_gauss": GaussianBiLSTM(
        input_dim=X_train_bilstm.shape[-1],
        hidden_dim=32,
        num_layers=2,
        dropout=0.4,
        min_log_var=-14.0,
        max_log_var=-10.0,
    ),
    "dual_gauss": GaussianDualWrapper(
        market_idx=market_idx,
        macro_idx=macro_idx,
        hidden_dim=48,
        num_layers=1,
        dropout=0.05,
        min_log_var=-14.0,
        max_log_var=-10.0,
    ),
}

# -----------------------------------------------------------------------------
# Checkpoint paths
# -----------------------------------------------------------------------------
checkpoint_paths = {
    "baseline_det": "best_baseline_model.pt",
    "dc_det": "best_dc_lstm_model.pt",
    "bilstm_det": "best_bilstm_model.pt",
    "dual_det": "best_dual_branch_model.pt",
    "baseline_gauss": "best_gaussian_model.pt",
    "dc_gauss": "best_gaussian_dc_lstm_model.pt",
    "bilstm_gauss": "best_gaussian_bilstm_model.pt",
    "dual_gauss": "best_gaussian_dual_model.pt",
}

# -----------------------------------------------------------------------------
# Helper function to load checkpoint
# -----------------------------------------------------------------------------
def load_checkpoint(model, checkpoint_path, device):
    """
    Load a saved PyTorch checkpoint into a model and move the model to the
    requested device.

    Parameters
    ----------
    model : nn.Module
        Instantiated model object.
    checkpoint_path : str | Path
        Path to the saved checkpoint file.
    device : torch.device | str
        Target device used for inference.

    Returns
    -------
    nn.Module
        Model with loaded weights, ready for evaluation mode.
    """
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Loaded {checkpoint_path}")
    if "epoch" in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if "val_loss" in checkpoint:
        print(f"  Val loss: {checkpoint['val_loss']}")
    return model, checkpoint


# -----------------------------------------------------------------------------
# Load all models
# -----------------------------------------------------------------------------
loaded_models = {}
loaded_checkpoints = {}

all_models = {}
all_models.update(deterministic_models)
all_models.update(gaussian_models)

for model_name, model in all_models.items():
    loaded_model, checkpoint = load_checkpoint(
        model=model,
        checkpoint_path=checkpoint_paths[model_name],
        device=device,
    )
    loaded_models[model_name] = loaded_model
    loaded_checkpoints[model_name] = checkpoint

print("\nAll models loaded successfully.")
print("Loaded model names:")
for name in loaded_models:
    print(f" - {name}")

# -----------------------------------------------------------------------------
# Evaluate all models and store results by architecture
# -----------------------------------------------------------------------------
results = {
    "baseline": {},
    "dc": {},
    "bilstm": {},
    "dual": {},
}

# -----------------------------------------------------------------------------
# Helper mappings for architecture-specific loaders / dfs
# -----------------------------------------------------------------------------
arch_test_loader = {
    "baseline": common_test_loader,
    "dc": common_test_loader,
    "dual": common_test_loader,
    "bilstm": bilstm_test_loader,
}

arch_test_df = {
    "baseline": test_df_common,
    "dc": test_df_common,
    "dual": test_df_common,
    "bilstm": test_df_bilstm,
}

# -------------------------
# Deterministic evaluations
# -------------------------
for arch in ["baseline", "dc", "bilstm", "dual"]:
    det_model = loaded_models[f"{arch}_det"]
    data_loader = arch_test_loader[arch]

    pred, targets, metrics = evaluate_model(
        model=det_model,
        data_loader=data_loader,
        device=device,
    )

    set_seed(42)
    mc_pred, mc_std, mc_targets, all_preds, mc_metrics = evaluate_model_mc(
        model=det_model,
        data_loader=data_loader,
        mc_samples=MC_SAMPLES,
        device=device,
    )

    results[arch]["pred"] = pred
    results[arch]["targets"] = targets
    results[arch]["metrics"] = metrics

    results[arch]["mc_pred"] = mc_pred
    results[arch]["mc_std"] = mc_std
    results[arch]["mc_lower"] = mc_pred - 1.96 * mc_std
    results[arch]["mc_upper"] = mc_pred + 1.96 * mc_std
    results[arch]["mc_targets"] = mc_targets
    results[arch]["mc_all_preds"] = all_preds
    results[arch]["mc_metrics"] = mc_metrics

# -------------------------
# Gaussian evaluations
# -------------------------
for arch in ["baseline", "dc", "bilstm", "dual"]:
    gauss_model = loaded_models[f"{arch}_gauss"]
    data_loader = arch_test_loader[arch]

    gaus_pred, gaus_std, gaus_targets, gaus_metrics = evaluate_model_gaussian(
        model=gauss_model,
        data_loader=data_loader,
        device=device,
    )

    set_seed(42)
    (
        mg_pred,
        mg_total_std,
        mg_aleatoric_std,
        mg_epistemic_std,
        mg_targets,
        mg_all_means,
        mg_all_stds,
        mg_metrics,
    ) = evaluate_model_mc_gaussian(
        model=gauss_model,
        data_loader=data_loader,
        mc_samples=MC_SAMPLES,
        device=device,
    )

    results[arch]["gaus_pred"] = gaus_pred
    results[arch]["gaus_std"] = gaus_std
    results[arch]["gaus_lower"] = gaus_pred - 1.96 * gaus_std
    results[arch]["gaus_upper"] = gaus_pred + 1.96 * gaus_std
    results[arch]["gaus_targets"] = gaus_targets
    results[arch]["gaus_metrics"] = gaus_metrics

    results[arch]["mg_pred"] = mg_pred
    results[arch]["mg_total_std"] = mg_total_std
    results[arch]["mg_aleatoric_std"] = mg_aleatoric_std
    results[arch]["mg_epistemic_std"] = mg_epistemic_std
    results[arch]["mg_lower"] = mg_pred - 1.96 * mg_total_std
    results[arch]["mg_upper"] = mg_pred + 1.96 * mg_total_std
    results[arch]["mg_targets"] = mg_targets
    results[arch]["mg_all_means"] = mg_all_means
    results[arch]["mg_all_stds"] = mg_all_stds
    results[arch]["mg_metrics"] = mg_metrics

baseline_pred = results["baseline"]["pred"]
baseline_mc_pred = results["baseline"]["mc_pred"]
baseline_gaus_pred = results["baseline"]["gaus_pred"]
baseline_mg_pred = results["baseline"]["mg_pred"]

dc_pred = results["dc"]["pred"]
dc_mc_pred = results["dc"]["mc_pred"]
dc_gaus_pred = results["dc"]["gaus_pred"]
dc_mg_pred = results["dc"]["mg_pred"]

bilstm_pred = results["bilstm"]["pred"]
bilstm_mc_pred = results["bilstm"]["mc_pred"]
bilstm_gaus_pred = results["bilstm"]["gaus_pred"]
bilstm_mg_pred = results["bilstm"]["mg_pred"]

dual_pred = results["dual"]["pred"]
dual_mc_pred = results["dual"]["mc_pred"]
dual_gaus_pred = results["dual"]["gaus_pred"]
dual_mg_pred = results["dual"]["mg_pred"]

# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
from fx_forecasting.evaluation.price_paths import compute_price_paths
from fx_forecasting.utils.plotting import plot_price_path

baseline_actual, baseline_mg_price_pred, baseline_mg_lower, baseline_mg_upper = compute_price_paths(
    pred_mean=baseline_mg_pred,
    all_preds=results["baseline"]["mg_all_means"],
    df_raw=df_raw,
    test_df=arch_test_df["baseline"],
)

dc_actual, dc_mg_price_pred, dc_mg_lower, dc_mg_upper = compute_price_paths(
    pred_mean=dc_mg_pred,
    all_preds=results["dc"]["mg_all_means"],
    df_raw=df_raw,
    test_df=arch_test_df["dc"],
)

bilstm_actual, bilstm_mg_price_pred, bilstm_mg_lower, bilstm_mg_upper = compute_price_paths(
    pred_mean=bilstm_mg_pred,
    all_preds=results["bilstm"]["mg_all_means"],
    df_raw=df_raw,
    test_df=arch_test_df["bilstm"],
)

dual_actual, dual_mg_price_pred, dual_mg_lower, dual_mg_upper = compute_price_paths(
    pred_mean=dual_mg_pred,
    all_preds=results["dual"]["mg_all_means"],
    df_raw=df_raw,
    test_df=arch_test_df["dual"],
)

# -----------------------------------------------------------------------------
# Compute anchor prices
# -----------------------------------------------------------------------------
actual = baseline_actual
price_df = df_raw[["Date", "GBP-USD"]].copy()
price_df["Date"] = pd.to_datetime(price_df["Date"])

test_dates = pd.to_datetime(arch_test_df["baseline"]["Date"]).reset_index(drop=True)

naive_pred = (
    test_dates.to_frame(name="Date")
    .merge(price_df, on="Date", how="left")["GBP-USD"]
    .to_numpy()
)

plot_price_path(
    actual=actual,
    naive_pred=naive_pred,
    model_curves=[
        {
            "name": "DC-LSTM Gaussian + MC",
            "pred": dc_mg_price_pred,
            "lower": dc_mg_lower,
            "upper": dc_mg_upper,
            "color": "red",
        }
    ],
    title="1-Step Ahead Price Prediction (DC-LSTM Gaussian + MC Dropout)",
    save_path="evaluation_results/plots/price_paths/dc_lstm_gauss_mc.png",
)

from fx_forecasting.evaluation.performance import (
    get_stats,
    get_cumulative_curve,
    evaluate_random_baseline,
    build_performance_summary,
    plot_performance_comparison,
)

OPTIMAL_SNR = 0.09

performance_results, performance_summary_df = build_performance_summary(
    results=results,
    optimal_snr=OPTIMAL_SNR,
)

# change baseline lstm architecture name
performance_summary_df["architecture"] = performance_summary_df["architecture"].replace({
    "Baseline": "LSTM"
})
market_targets = results["baseline"]["targets"]

mk_total, mk_sharpe = get_stats(market_targets)
cum_market = get_cumulative_curve(market_targets)

random_baseline = evaluate_random_baseline(market_targets)

os.makedirs("evaluation_results/tables", exist_ok=True)

performance_summary_df.to_csv(
    "evaluation_results/tables/performance_summary.csv",
    index=False,
)


benchmark_df = pd.DataFrame([
    {
        "architecture": "Benchmark",
        "variant": "Buy & Hold",
        "total_return": mk_total,
        "sharpe": mk_sharpe,
        "hit_ratio": np.nan,
        "exposure": 1.0,
    },
    {
        "architecture": "Benchmark",
        "variant": "Random",
        "total_return": random_baseline["total_return"],
        "sharpe": random_baseline["sharpe"],
        "hit_ratio": random_baseline["hit_ratio"],
        "exposure": 1.0,
    },
])

full_performance_table = pd.concat(
    [performance_summary_df, benchmark_df],
    ignore_index=True,
)

full_performance_table.to_csv(
    "evaluation_results/tables/performance_summary_with_benchmarks.csv",
    index=False,
)

print("\nPerformance summary:")
print(full_performance_table.round(4))

plot_performance_comparison(
    curves=[
        {
            "label": "DC-LSTM Gaussian + MC",
            "curve": performance_results["dc"]["mg"]["curve"],
            "hit_ratio": performance_results["dc"]["mg"]["hit_ratio"],
            "sharpe": performance_results["dc"]["mg"]["sharpe"],
            "color": "green",
        },
        {
            "label": "LSTM Baseline Gaussian + MC",
            "curve": performance_results["baseline"]["mg"]["curve"],
            "hit_ratio": performance_results["baseline"]["mg"]["hit_ratio"],
            "sharpe": performance_results["baseline"]["mg"]["sharpe"],
            "color": "red",
        },
        {
            "label": "LSTM Baseline Deterministic",
            "curve": performance_results["baseline"]["det"]["curve"],
            "hit_ratio": performance_results["baseline"]["det"]["hit_ratio"],
            "sharpe": performance_results["baseline"]["det"]["sharpe"],
            "color": "blue",
        },
    ],
    market_curve=cum_market,
    labels={
        "market_label": f"Market (Buy & Hold, Sharpe: {mk_sharpe:.2f})",
    },
    title="Performance Comparison: DC-LSTM Uncertainty-Aware vs Baseline Models",
    save_path="evaluation_results/plots/performance/dc_vs_baseline_performance.png",
)

arch_pretty = {
    "baseline": "LSTM",
    "dc": "DC-LSTM",
    "bilstm": "BiLSTM",
    "dual": "Dual-Branch",
}

for arch in ["baseline", "dc", "bilstm", "dual"]:
    plot_performance_comparison(
        curves=[
            {
                "label": f'{arch_pretty[arch]} Gaussian + MC',
                "curve": performance_results[arch]["mg"]["curve"],
                "hit_ratio": performance_results[arch]["mg"]["hit_ratio"],
                "sharpe": performance_results[arch]["mg"]["sharpe"],
                "color": "green",
            },
            {
                "label": f'{arch_pretty[arch]} Deterministic',
                "curve": performance_results[arch]["det"]["curve"],
                "hit_ratio": performance_results[arch]["det"]["hit_ratio"],
                "sharpe": performance_results[arch]["det"]["sharpe"],
                "color": "blue",
            },
        ],
        market_curve=cum_market,
        labels={
            "market_label": f"Market (Buy & Hold, Sharpe: {mk_sharpe:.2f})",
        },
        title=f"Performance Comparison: {arch_pretty[arch]}",
        save_path=f"evaluation_results/plots/performance/{arch}_performance.png",
    )

from fx_forecasting.evaluation.price_metrics import save_price_metrics_tables

all_price_metrics_df = save_price_metrics_tables(
    results=results,
    test_anchor_prices=naive_pred,
    output_dir="evaluation_results/tables/price_metrics",
    n_gaussian_samples=100,
    random_seed=42,
)

# renaming architectures
all_price_metrics_df["Architecture"] = all_price_metrics_df["Architecture"].replace({
    "baseline": "LSTM"
})
mask = all_price_metrics_df["Model"].isin(["Zero Baseline", "Random Baseline"])
all_price_metrics_df.loc[mask, "Architecture"] = "Baseline"

# Identify baseline rows
baseline_mask = all_price_metrics_df["Architecture"] == "Baseline"

# Split dataframe
baseline_df = all_price_metrics_df[baseline_mask]
non_baseline_df = all_price_metrics_df[~baseline_mask]

# Drop duplicate baseline rows based on Model
baseline_df = baseline_df.drop_duplicates(subset=["Model"], keep="first")

# Combine back
all_price_metrics_df = pd.concat([non_baseline_df, baseline_df], ignore_index=True)



print("\nCombined price metrics table:")
print(all_price_metrics_df.round(4))

from fx_forecasting.utils.plotting import plot_uncertainty_decomposition

uncertainty_plot_config = {
    "baseline": {
        "title": "LSTM Baseline Gaussian + MC Dropout: Uncertainty Decomposition",
        "save_path": "evaluation_results/plots/uncertainty/baseline_uncertainty_decomposition.png",
    },
    "dc": {
        "title": "DC-LSTM Gaussian + MC Dropout: Uncertainty Decomposition",
        "save_path": "evaluation_results/plots/uncertainty/dc_lstm_uncertainty_decomposition.png",
    },
    "bilstm": {
        "title": "BiLSTM Gaussian + MC Dropout: Uncertainty Decomposition",
        "save_path": "evaluation_results/plots/uncertainty/bilstm_uncertainty_decomposition.png",
    },
    "dual": {
        "title": "Dual-Branch Gaussian + MC Dropout: Uncertainty Decomposition",
        "save_path": "evaluation_results/plots/uncertainty/dual_branch_uncertainty_decomposition.png",
    },
}

for arch, cfg in uncertainty_plot_config.items():
    plot_uncertainty_decomposition(
        pred_mean=results[arch]["mg_pred"],
        targets=results[arch]["mg_targets"],
        aleatoric_std=results[arch]["mg_aleatoric_std"],
        epistemic_std=results[arch]["mg_epistemic_std"],
        total_std=results[arch]["mg_total_std"],
        title=cfg["title"],
        save_path=cfg["save_path"],
    )