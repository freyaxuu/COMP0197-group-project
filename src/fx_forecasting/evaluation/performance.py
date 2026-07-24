import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_stats(rets):
    rets = np.asarray(rets)
    total = (np.exp(np.sum(rets)) - 1) * 100
    vol = np.std(rets)
    sharpe = np.sqrt(252) * (np.mean(rets) / vol) if vol > 1e-9 else 0.0
    return total, sharpe


def get_cumulative_curve(log_returns):
    return np.exp(np.cumsum(log_returns))


def get_filtered_signals(mean, std=None, snr_threshold=0.0):
    """
    If std is None, this becomes a simple sign strategy.
    If std is provided, only trade when |mean| / std >= snr_threshold.
    """
    mean = np.asarray(mean)

    if std is not None:
        std = np.asarray(std)
        snr = np.divide(np.abs(mean), std, out=np.zeros_like(mean), where=std != 0)
        return np.where(snr >= snr_threshold, np.sign(mean), 0)

    return np.sign(mean)


def evaluate_random_baseline(targets, n_trials=100, seed=42):
    rng = np.random.default_rng(seed)
    curves = []
    totals, sharpes, hits = [], [], []

    targets = np.asarray(targets)

    for _ in range(n_trials):
        random_signals = rng.choice([-1, 1], size=len(targets))
        random_returns = random_signals * targets

        total, sharpe = get_stats(random_returns)
        hit = np.mean(np.sign(random_signals) == np.sign(targets))

        totals.append(total)
        sharpes.append(sharpe)
        hits.append(hit)
        curves.append(get_cumulative_curve(random_returns))

    return {
        "total_return": float(np.mean(totals)),
        "sharpe": float(np.mean(sharpes)),
        "hit_ratio": float(np.mean(hits)),
        "curve": np.mean(curves, axis=0),
    }


def evaluate_strategy(mean, targets, std=None, snr_threshold=0.0):
    mean = np.asarray(mean)
    targets = np.asarray(targets)
    std = None if std is None else np.asarray(std)

    signals = get_filtered_signals(mean, std=std, snr_threshold=snr_threshold)
    strat_rets = signals * targets

    total, sharpe = get_stats(strat_rets)

    active_mask = np.abs(signals) > 0
    if np.any(active_mask):
        hit = np.mean(np.sign(mean[active_mask]) == np.sign(targets[active_mask]))
        exposure = np.mean(active_mask)
    else:
        hit = 0.0
        exposure = 0.0

    return {
        "signals": signals,
        "returns": strat_rets,
        "curve": get_cumulative_curve(strat_rets),
        "total_return": total,
        "sharpe": sharpe,
        "hit_ratio": float(hit),
        "exposure": float(exposure),
    }


def build_performance_summary(results, optimal_snr=0.09):
    """
    results: your big results dict from test.py
    Returns:
        performance_results: nested dict
        summary_df: pandas DataFrame
    """
    performance_results = {}

    for arch in ["baseline", "dc", "bilstm", "dual"]:
        performance_results[arch] = {}

        # Deterministic
        performance_results[arch]["det"] = evaluate_strategy(
            mean=results[arch]["pred"],
            targets=results[arch]["targets"],
            std=None,
            snr_threshold=0.0,
        )

        # MC dropout
        performance_results[arch]["mc"] = evaluate_strategy(
            mean=results[arch]["mc_pred"],
            targets=results[arch]["mc_targets"],
            std=results[arch]["mc_std"],
            snr_threshold=optimal_snr,
        )

        # Gaussian
        performance_results[arch]["gauss"] = evaluate_strategy(
            mean=results[arch]["gaus_pred"],
            targets=results[arch]["gaus_targets"],
            std=results[arch]["gaus_std"],
            snr_threshold=optimal_snr,
        )

        # Gaussian + MC
        performance_results[arch]["mg"] = evaluate_strategy(
            mean=results[arch]["mg_pred"],
            targets=results[arch]["mg_targets"],
            std=results[arch]["mg_total_std"],
            snr_threshold=optimal_snr,
        )

    rows = []
    pretty_names = {
        "baseline": "Baseline",
        "dc": "DC-LSTM",
        "bilstm": "BiLSTM",
        "dual": "Dual-Branch",
    }
    variant_names = {
        "det": "Deterministic",
        "mc": "MC Dropout",
        "gauss": "Gaussian",
        "mg": "Gaussian + MC",
    }

    for arch in ["baseline", "dc", "bilstm", "dual"]:
        for variant in ["det", "mc", "gauss", "mg"]:
            stats = performance_results[arch][variant]
            rows.append({
                "architecture": pretty_names[arch],
                "variant": variant_names[variant],
                "total_return": stats["total_return"],
                "sharpe": stats["sharpe"],
                "hit_ratio": stats["hit_ratio"],
                "exposure": stats["exposure"],
            })

    summary_df = pd.DataFrame(rows)
    return performance_results, summary_df


def plot_performance_comparison(
    curves,
    market_curve,
    labels,
    title,
    save_path=None,
    figsize=(14, 7),
):
    plt.figure(figsize=figsize)

    for item in curves:
        plt.plot(
            item["curve"],
            label=f'{item["label"]} (HR: {item["hit_ratio"]:.1%}, Sharpe: {item["sharpe"]:.2f})',
            color=item.get("color", None),
            linestyle=item.get("linestyle", "-"),
            alpha=item.get("alpha", 1.0),
        )

    plt.plot(
        market_curve,
        label=labels.get("market_label", "Market (Buy & Hold)"),
        color="black",
        alpha=0.3,
        linestyle="--",
    )

    plt.axhline(1.0, color="red", linestyle="-", alpha=0.2)
    plt.title(title, fontsize=15)
    plt.xlabel("Days (Test Period)", fontsize=12)
    plt.ylabel("Cumulative Growth", fontsize=12)
    plt.legend(loc="upper left", fontsize=11)
    plt.grid(True, alpha=0.15)
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()