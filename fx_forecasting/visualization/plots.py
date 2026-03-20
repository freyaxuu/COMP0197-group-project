import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_all_series(df: pd.DataFrame, timestamp_col: str = "timestamp"):
    value_cols = [c for c in df.columns if c != timestamp_col]
    plt.figure(figsize=(12, 6))
    for col in value_cols:
        plt.plot(df[timestamp_col], df[col], label=col)
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.title("Time Series")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_target_series(df: pd.DataFrame, target_col: str, timestamp_col: str = "timestamp"):
    plt.figure(figsize=(12, 5))
    plt.plot(df[timestamp_col], df[target_col])
    plt.xlabel("Date")
    plt.ylabel(target_col)
    plt.title(f"{target_col} Over Time")
    plt.tight_layout()
    plt.show()


def plot_histogram(df: pd.DataFrame, col: str, bins: int = 30):
    plt.figure(figsize=(8, 5))
    plt.hist(df[col].dropna(), bins=bins)
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {col}")
    plt.tight_layout()
    plt.show()


def plot_boxplot(df: pd.DataFrame, cols=None, timestamp_col: str = "timestamp"):
    if cols is None:
        cols = [c for c in df.columns if c != timestamp_col]

    plt.figure(figsize=(10, 5))
    plt.boxplot([df[col].dropna() for col in cols], tick_labels=cols)
    plt.ylabel("Value")
    plt.title("Boxplot of Variables")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, timestamp_col: str = "timestamp"):
    value_cols = [c for c in df.columns if c != timestamp_col]
    corr = df[value_cols].corr()

    plt.figure(figsize=(8, 6))
    im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im)
    plt.xticks(range(len(value_cols)), value_cols, rotation=45)
    plt.yticks(range(len(value_cols)), value_cols)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()


def plot_training_history(history, metric="loss"):
    train_key = f"train_{metric}"
    val_key = f"val_{metric}"

    if train_key not in history or val_key not in history:
        raise ValueError(
            f"Metric '{metric}' not found in history. Available keys: {list(history.keys())}"
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
    plt.show()


def plot_predictions(preds, targets, train_targets=None, n_points=None):
    preds = np.asarray(preds).reshape(-1)
    targets = np.asarray(targets).reshape(-1)

    if train_targets is not None:
        train_targets = np.asarray(train_targets).reshape(-1)

    if n_points is not None:
        preds_plot = preds[-n_points:]
        targets_plot = targets[-n_points:]
    else:
        preds_plot = preds
        targets_plot = targets

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    if train_targets is not None:
        axes[0].plot(np.arange(len(train_targets)), train_targets, label="Train Actual")
        test_index = np.arange(len(train_targets), len(train_targets) + len(targets_plot))
    else:
        test_index = np.arange(len(targets_plot))

    axes[0].plot(test_index, targets_plot, label="Test Actual")
    axes[0].plot(test_index, preds_plot, label="Test Prediction")
    axes[0].legend()

    axes[1].plot(targets_plot, label="Actual", color="black", alpha=0.6)
    axes[1].plot(preds_plot, label="Prediction", color="red", linestyle="--")
    axes[1].legend()

    axes[2].plot(targets_plot - preds_plot, label="Residual")
    axes[2].axhline(0, linestyle="--")
    axes[2].legend()

    plt.tight_layout()
    plt.show()


def plot_probabilistic_predictions(
    mean_preds,
    std_preds,
    targets,
    lower=None,
    upper=None,
    title="Probabilistic Forecast",
    n_points=None,
):
    mean_preds = np.asarray(mean_preds).reshape(-1)
    std_preds = np.asarray(std_preds).reshape(-1)
    targets = np.asarray(targets).reshape(-1)

    if lower is None:
        lower = mean_preds - 1.96 * std_preds
    if upper is None:
        upper = mean_preds + 1.96 * std_preds

    if n_points is not None:
        mean_preds = mean_preds[-n_points:]
        std_preds = std_preds[-n_points:]
        targets = targets[-n_points:]
        lower = np.asarray(lower).reshape(-1)[-n_points:]
        upper = np.asarray(upper).reshape(-1)[-n_points:]

    x = np.arange(len(mean_preds))

    plt.figure(figsize=(12, 5))
    plt.plot(x, targets, label="Actual", color="black", alpha=0.7)
    plt.plot(x, mean_preds, label="Predicted Mean", color="red", linestyle="--")
    plt.fill_between(x, lower, upper, color="red", alpha=0.2, label="95% interval")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()