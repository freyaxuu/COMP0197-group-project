import os
import numpy as np
import pandas as pd


def evaluate_price_performance(
    name,
    anchor_prices,
    log_returns,
    actual_log_returns,
    uncertainty_samples=None,
):
    """
    Converts log-return predictions to price and calculates metrics.

    Parameters
    ----------
    name : str
        Model name for reporting.
    anchor_prices : np.ndarray, shape (T,)
        Observed prices at time t used to predict time t+1.
    log_returns : np.ndarray, shape (T,)
        Mean predicted log returns.
    actual_log_returns : np.ndarray, shape (T,)
        Ground-truth log returns.
    uncertainty_samples : np.ndarray, shape (n_samples, T), optional
        Samples in log-return space for probabilistic models.

    Returns
    -------
    results : dict
        RMSE/MAE/PICP/MPIW metrics.
    pred_price : np.ndarray
        Predicted price path.
    """
    anchor_prices = np.asarray(anchor_prices)
    log_returns = np.asarray(log_returns)
    actual_log_returns = np.asarray(actual_log_returns)

    actual_price = anchor_prices * np.exp(actual_log_returns)
    pred_price = anchor_prices * np.exp(log_returns)

    rmse = np.sqrt(np.mean((actual_price - pred_price) ** 2))
    mae = np.mean(np.abs(actual_price - pred_price))

    results = {
        "Model": name,
        "RMSE (Price)": rmse,
        "MAE (Price)": mae,
    }

    if uncertainty_samples is not None:
        uncertainty_samples = np.asarray(uncertainty_samples)
        price_samples = anchor_prices[None, :] * np.exp(uncertainty_samples)

        lower = np.percentile(price_samples, 2.5, axis=0)
        upper = np.percentile(price_samples, 97.5, axis=0)

        covered = np.logical_and(actual_price >= lower, actual_price <= upper)
        results["PICP"] = np.mean(covered)
        results["MPIW"] = np.mean(upper - lower)
    else:
        results["PICP"] = np.nan
        results["MPIW"] = np.nan

    return results, pred_price


def build_price_metrics_table(
    results,
    arch,
    test_anchor_prices,
    n_gaussian_samples=100,
    random_seed=42,
):
    """
    Build one price-metrics table for a single architecture.

    Parameters
    ----------
    results : dict
        Main evaluation results dictionary from test.py
    arch : str
        One of: baseline, dc, bilstm, dual
    test_anchor_prices : np.ndarray
        Anchor prices aligned to test dates
    n_gaussian_samples : int
        Number of samples for Gaussian-only uncertainty approximation
    random_seed : int
        Reproducibility seed

    Returns
    -------
    df_comparison : pd.DataFrame
    """
    rng = np.random.default_rng(random_seed)
    comparison_list = []

    targets = np.asarray(results[arch]["targets"])

    # Zero baseline
    zero_preds = np.zeros_like(targets)
    res_zero, _ = evaluate_price_performance(
        "Zero Baseline",
        test_anchor_prices,
        zero_preds,
        targets,
    )
    comparison_list.append(res_zero)

    # Random baseline
    random_preds = rng.choice([-1, 1], size=len(targets)) * np.std(targets)
    res_random, _ = evaluate_price_performance(
        "Random Baseline",
        test_anchor_prices,
        random_preds,
        targets,
    )
    comparison_list.append(res_random)

    # Deterministic
    res_det, _ = evaluate_price_performance(
        "Deterministic",
        test_anchor_prices,
        results[arch]["pred"],
        results[arch]["targets"],
    )
    comparison_list.append(res_det)

    # MC Dropout
    res_mc, _ = evaluate_price_performance(
        "MC Dropout",
        test_anchor_prices,
        results[arch]["mc_pred"],
        results[arch]["mc_targets"],
        uncertainty_samples=results[arch]["mc_all_preds"],
    )
    comparison_list.append(res_mc)

    # Gaussian
    gaussian_samples = rng.normal(
        loc=np.asarray(results[arch]["gaus_pred"]),
        scale=np.asarray(results[arch]["gaus_std"]),
        size=(n_gaussian_samples, len(results[arch]["gaus_pred"])),
    )

    res_gauss, _ = evaluate_price_performance(
        "Gaussian",
        test_anchor_prices,
        results[arch]["gaus_pred"],
        results[arch]["gaus_targets"],
        uncertainty_samples=gaussian_samples,
    )
    comparison_list.append(res_gauss)

    # Gaussian + MC Dropout
    combined_samples = rng.normal(
        loc=np.asarray(results[arch]["mg_all_means"]),
        scale=np.asarray(results[arch]["mg_all_stds"]),
    )

    res_mg, _ = evaluate_price_performance(
        "Gaussian + MC Dropout",
        test_anchor_prices,
        results[arch]["mg_pred"],
        results[arch]["mg_targets"],
        uncertainty_samples=combined_samples,
    )
    comparison_list.append(res_mg)

    df_comparison = pd.DataFrame(comparison_list)
    df_comparison = df_comparison.sort_values(by="RMSE (Price)").reset_index(drop=True)
    return df_comparison


def save_price_metrics_tables(
    results,
    test_anchor_prices,
    output_dir="results/tables/price_metrics",
    n_gaussian_samples=100,
    random_seed=42,
):
    """
    Save one CSV per architecture, plus one combined CSV.
    """
    os.makedirs(output_dir, exist_ok=True)

    all_tables = []
    arch_pretty = {
        "baseline": "baseline",
        "dc": "dc_lstm",
        "bilstm": "bilstm",
        "dual": "dual_branch",
    }

    for arch in ["baseline", "dc", "bilstm", "dual"]:
        df_arch = build_price_metrics_table(
            results=results,
            arch=arch,
            test_anchor_prices=test_anchor_prices,
            n_gaussian_samples=n_gaussian_samples,
            random_seed=random_seed,
        )

        file_path = os.path.join(output_dir, f"{arch_pretty[arch]}_price_metrics.csv")
        df_arch.to_csv(file_path, index=False)

        df_arch_with_arch = df_arch.copy()
        df_arch_with_arch.insert(0, "Architecture", arch_pretty[arch])
        all_tables.append(df_arch_with_arch)

    combined_df = pd.concat(all_tables, ignore_index=True)
    combined_path = os.path.join(output_dir, "all_models_price_metrics.csv")
    combined_df.to_csv(combined_path, index=False)

    return combined_df