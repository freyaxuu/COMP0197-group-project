import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def rmse_vs_naive(actual, pred, naive):
    model_rmse = np.sqrt(mean_squared_error(actual, pred))
    naive_rmse = np.sqrt(mean_squared_error(actual, naive))
    improvement = (naive_rmse - model_rmse) / naive_rmse * 100
    return model_rmse, naive_rmse, improvement


def plot_price_path(
    actual,
    naive_pred,
    model_curves,
    title,
    save_path=None,
    xlabel="Test Timestep (Days)",
    ylabel="GBP-USD",
    figsize=(16, 6),
):
    """
    Plot actual path, naive baseline, and one or more model predictions with bands.

    Parameters
    ----------
    actual : array-like, shape (T,)
        Ground-truth price path.
    naive_pred : array-like, shape (T,)
        Naive baseline predictions.
    model_curves : list of dict
        Each dict should contain:
            {
                "name": str,
                "pred": np.ndarray shape (T,),
                "lower": np.ndarray shape (T,),
                "upper": np.ndarray shape (T,),
                "color": str
            }
    title : str
        Plot title.
    save_path : str or None
        If given, saves figure to this path.
    """
    actual = np.asarray(actual)
    naive_pred = np.asarray(naive_pred)
    x = np.arange(len(actual))

    plt.figure(figsize=figsize)

    plt.plot(
        x, actual,
        color="black",
        alpha=0.7,
        lw=1.5,
        label="Actual"
    )
    plt.plot(
        x, naive_pred,
        color="blue",
        alpha=0.8,
        lw=1.2,
        linestyle=":",
        label="Naive"
    )

    for model in model_curves:
        name = model["name"]
        pred = np.asarray(model["pred"])
        lower = np.asarray(model["lower"])
        upper = np.asarray(model["upper"])
        color = model.get("color", None)

        rmse, naive_rmse, improvement = rmse_vs_naive(actual, pred, naive_pred)

        plt.plot(
            x,
            pred,
            color=color,
            alpha=0.9,
            lw=1.5,
            linestyle="--",
            label=f"{name} (RMSE {rmse:.5f})"
        )

        plt.fill_between(
            x,
            lower,
            upper,
            color=color,
            alpha=0.2
        )

    plt.title(title, fontweight="bold")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    
    


def plot_uncertainty_decomposition(
    pred_mean,
    targets,
    aleatoric_std,
    epistemic_std,
    total_std,
    title,
    save_path=None,
    xlabel="Test Timestep",
    ylabel_top="Target",
    ylabel_bottom="Std",
):
    """
    Plot Gaussian + MC dropout forecast with uncertainty decomposition.

    Parameters
    ----------
    pred_mean : array-like, shape (T,)
        Predictive mean.
    targets : array-like, shape (T,)
        Ground-truth targets.
    aleatoric_std : array-like, shape (T,)
        Aleatoric uncertainty std.
    epistemic_std : array-like, shape (T,)
        Epistemic uncertainty std.
    total_std : array-like, shape (T,)
        Total predictive std.
    title : str
        Figure title.
    save_path : str or None
        Path to save figure.
    """
    pred_mean = np.asarray(pred_mean)
    targets = np.asarray(targets)
    aleatoric_std = np.asarray(aleatoric_std)
    epistemic_std = np.asarray(epistemic_std)
    total_std = np.asarray(total_std)

    x = np.arange(len(pred_mean))

    ale_lower_95 = pred_mean - 1.96 * aleatoric_std
    ale_upper_95 = pred_mean + 1.96 * aleatoric_std

    total_lower_95 = pred_mean - 1.96 * total_std
    total_upper_95 = pred_mean + 1.96 * total_std

    fig, axes = plt.subplots(
        2, 1,
        figsize=(16, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]}
    )

    # Top panel
    ax = axes[0]
    ax.plot(x, targets, color="black", alpha=0.7, lw=1.2, label="Actual")
    ax.plot(x, pred_mean, color="red", linestyle="--", lw=1.5, label="Predicted Mean")

    ax.fill_between(
        x,
        total_lower_95,
        total_upper_95,
        color="tab:blue",
        alpha=0.18,
        label="Total 95% Interval"
    )

    ax.fill_between(
        x,
        ale_lower_95,
        ale_upper_95,
        color="tab:orange",
        alpha=0.25,
        label="Aleatoric 95% Interval"
    )

    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel_top)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.25)

    # Bottom panel
    ax2 = axes[1]
    ax2.plot(x, aleatoric_std, label="Aleatoric Std", lw=1.5)
    ax2.plot(x, epistemic_std, label="Epistemic Std", lw=1.5)
    ax2.plot(x, total_std, label="Total Std", lw=2)

    ax2.set_title("Uncertainty Decomposition")
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel_bottom)
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.25)

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()