from pyexpat import model

import torch
import numpy as np

def predict(model, data_loader, device="cpu"):
    """
    Generate deterministic predictions for a dataset.
    """
    model.to(device)
    model.eval()

    preds = []
    targets = []

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)

            output = model(batch_x).reshape(-1)

            preds.append(output.cpu().numpy())
            targets.append(batch_y.numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    return preds, targets

def compute_metrics(preds, targets):
    """
    Compute basic regression metrics.
    """
    mse = np.mean((preds - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(preds - targets))

    metrics = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
    }

    return metrics

def evaluate_model(model, data_loader, device="cpu"):
    """
    Full deterministic evaluation pipeline.
    """
    preds, targets = predict(model, data_loader, device)
    metrics = compute_metrics(preds, targets)

    print("\nEvaluation metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")

    return preds, targets, metrics

# =========================================================
# MC DROPOUT HELPERS
# =========================================================

def enable_mc_dropout(model):
    """
    Turn on dropout layers during inference while keeping the rest of the model as-is.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()

def predict_mc_dropout(model, data_loader, mc_samples=100, device="cpu"):
    """
    Monte Carlo Dropout prediction.

    Returns
    -------
    pred_mean : np.ndarray
        Predictive mean, shape (N,)
    pred_std : np.ndarray
        Predictive std, shape (N,)
    targets : np.ndarray
        Ground truth, shape (N,)
    all_preds : np.ndarray
        All MC samples, shape (mc_samples, N)
    """
    model.to(device)
    model.train()   # enable LSTM internal dropout + explicit dropout

    all_preds = []
    targets = None

    with torch.no_grad():
        for _ in range(mc_samples):
            sample_preds = []
            current_targets = []

            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(device)

                output = model(batch_x).reshape(-1)

                sample_preds.append(output.cpu().numpy())
                current_targets.append(batch_y.numpy())

            sample_preds = np.concatenate(sample_preds)
            current_targets = np.concatenate(current_targets)

            all_preds.append(sample_preds)

            if targets is None:
                targets = current_targets

    all_preds = np.stack(all_preds, axis=0)   # (mc_samples, N)
    pred_mean = all_preds.mean(axis=0)
    pred_std = all_preds.std(axis=0)

    return pred_mean, pred_std, targets, all_preds

def compute_prediction_interval_95(pred_mean, pred_std):
    lower = pred_mean - 1.96 * pred_std
    upper = pred_mean + 1.96 * pred_std
    return lower, upper

def evaluate_model_mc(model, data_loader, mc_samples=100, device="cpu"):
    """
    Full MC Dropout evaluation pipeline.
    """
    pred_mean, pred_std, targets, all_preds = predict_mc_dropout(
        model=model,
        data_loader=data_loader,
        mc_samples=mc_samples,
        device=device,
    )

    metrics = compute_metrics(pred_mean, targets)

    print(f"\nMC Dropout evaluation metrics (mc_samples={mc_samples}):")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")

    print(f"Average predictive std: {pred_std.mean():.6f}")

    return pred_mean, pred_std, targets, all_preds, metrics

# =========================================================
# GAUSSIAN OUTPUT + MC DROPOUT HELPERS
# =========================================================

def predict_gaussian(model, data_loader, device="cpu"):
    """
    Deterministic Gaussian-output prediction.

    Returns
    -------
    pred_mean : np.ndarray
        Shape (N,)
    pred_std : np.ndarray
        Shape (N,)
    targets : np.ndarray
        Shape (N,)
    """
    model.to(device)
    model.eval()

    pred_means = []
    pred_stds = []
    targets = []

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)

            mean, log_var = model(batch_x)
            mean = mean.reshape(-1)
            log_var = log_var.reshape(-1)

            std = torch.exp(0.5 * log_var)

            pred_means.append(mean.cpu().numpy())
            pred_stds.append(std.cpu().numpy())
            targets.append(batch_y.numpy())

    pred_mean = np.concatenate(pred_means)
    pred_std = np.concatenate(pred_stds)
    targets = np.concatenate(targets)

    return pred_mean, pred_std, targets

def evaluate_model_gaussian(model, data_loader, device="cpu"):
    """
    Evaluate Gaussian-output model using mean prediction for point metrics.
    """
    pred_mean, pred_std, targets = predict_gaussian(model, data_loader, device=device)
    metrics = compute_metrics(pred_mean, targets)

    print("\nGaussian evaluation metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")
    print(f"Average predictive std (aleatoric): {pred_std.mean():.6f}")

    return pred_mean, pred_std, targets, metrics

def predict_mc_dropout_gaussian(model, data_loader, mc_samples=100, device="cpu"):
    """
    MC Dropout + Gaussian output.

    Returns
    -------
    pred_mean : np.ndarray
        Final predictive mean, shape (N,)
    total_std : np.ndarray
        Total predictive std, shape (N,)
    aleatoric_std : np.ndarray
        Aleatoric std, shape (N,)
    epistemic_std : np.ndarray
        Epistemic std, shape (N,)
    targets : np.ndarray
        Shape (N,)
    all_means : np.ndarray
        Shape (mc_samples, N)
    all_stds : np.ndarray
        Shape (mc_samples, N)
    """
    model.to(device)


    # Use eval() first, then only enable explicit Dropout layers.
    # This keeps BatchNorm/other layers stable and avoids overly strong
    # stochasticity from LSTM internal dropout.
    model.eval()
    enable_mc_dropout(model)

    all_means = []
    all_stds = []
    targets = None

    with torch.no_grad():
        for _ in range(mc_samples):
            sample_means = []
            sample_stds = []
            current_targets = []

            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(device)

                mean, log_var = model(batch_x)
                mean = mean.reshape(-1)
                log_var = log_var.reshape(-1)
                std = torch.exp(0.5 * log_var)

                sample_means.append(mean.cpu().numpy())
                sample_stds.append(std.cpu().numpy())
                current_targets.append(batch_y.numpy())

            sample_means = np.concatenate(sample_means)
            sample_stds = np.concatenate(sample_stds)
            current_targets = np.concatenate(current_targets)

            all_means.append(sample_means)
            all_stds.append(sample_stds)

            if targets is None:
                targets = current_targets

    all_means = np.stack(all_means, axis=0)   # (S, N)
    all_stds = np.stack(all_stds, axis=0)     # (S, N)

    pred_mean = all_means.mean(axis=0)

    # epistemic uncertainty = variation of means across MC passes
    epistemic_var = all_means.var(axis=0)

    # aleatoric uncertainty = mean predicted variance
    aleatoric_var = (all_stds ** 2).mean(axis=0)

    total_var = epistemic_var + aleatoric_var

    epistemic_std = np.sqrt(epistemic_var)
    aleatoric_std = np.sqrt(aleatoric_var)
    total_std = np.sqrt(total_var)

    return (
        pred_mean,
        total_std,
        aleatoric_std,
        epistemic_std,
        targets,
        all_means,
        all_stds,
    )

def evaluate_model_mc_gaussian(model, data_loader, mc_samples=100, device="cpu"):
    """
    Full evaluation for MC Dropout + Gaussian output.
    """
    (
        pred_mean,
        total_std,
        aleatoric_std,
        epistemic_std,
        targets,
        all_means,
        all_stds,
    ) = predict_mc_dropout_gaussian(
        model=model,
        data_loader=data_loader,
        mc_samples=mc_samples,
        device=device,
    )

    metrics = compute_metrics(pred_mean, targets)

    print(f"\nMC Dropout + Gaussian evaluation metrics (mc_samples={mc_samples}):")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")
    print(f"Average total std: {total_std.mean():.6f}")
    print(f"Average aleatoric std: {aleatoric_std.mean():.6f}")
    print(f"Average epistemic std: {epistemic_std.mean():.6f}")

    return (
        pred_mean,
        total_std,
        aleatoric_std,
        epistemic_std,
        targets,
        all_means,
        all_stds,
        metrics,
    )
