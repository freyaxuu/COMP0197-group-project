import numpy as np
import torch


def predict(model, data_loader, device="cpu"):
    model.to(device)
    model.eval()

    preds = []
    targets = []

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            output = model(batch_x).view(-1)

            preds.append(output.cpu().numpy())
            targets.append(batch_y.view(-1).cpu().numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    return preds, targets


def compute_metrics(preds, targets):
    mse = np.mean((preds - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(preds - targets))

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
    }


def evaluate_model(model, data_loader, device="cpu"):
    preds, targets = predict(model, data_loader, device)
    metrics = compute_metrics(preds, targets)

    print("\nEvaluation metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")

    return preds, targets, metrics

def enable_dropout_during_inference(model):
    """
    Keep dropout active at test time for Monte Carlo sampling.
    BatchNorm layers would normally stay in eval mode, but this model does not use BN.
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


def predict_mc_dropout(model, data_loader, mc_samples=100, device="cpu"):
    """
    Monte Carlo dropout prediction.

    Returns
    -------
    mean_preds : np.ndarray
    std_preds  : np.ndarray
    all_preds  : np.ndarray, shape (mc_samples, n_examples)
    targets    : np.ndarray
    """
    model.to(device)
    enable_dropout_during_inference(model)

    all_sample_preds = []
    targets = None

    with torch.no_grad():
        for _ in range(mc_samples):
            sample_preds = []
            current_targets = []

            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(device)
                output = model(batch_x).view(-1)

                sample_preds.append(output.cpu().numpy())
                current_targets.append(batch_y.view(-1).cpu().numpy())

            sample_preds = np.concatenate(sample_preds)
            current_targets = np.concatenate(current_targets)

            all_sample_preds.append(sample_preds)

            if targets is None:
                targets = current_targets

    all_sample_preds = np.stack(all_sample_preds, axis=0)  # [mc_samples, N]
    mean_preds = all_sample_preds.mean(axis=0)
    std_preds = all_sample_preds.std(axis=0)

    return mean_preds, std_preds, all_sample_preds, targets


def compute_probabilistic_metrics(mean_preds, std_preds, targets, z_value=1.96):
    """
    Basic metrics for probabilistic regression.
    """
    mse = np.mean((mean_preds - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(mean_preds - targets))

    lower = mean_preds - z_value * std_preds
    upper = mean_preds + z_value * std_preds
    coverage_95 = np.mean((targets >= lower) & (targets <= upper))

    avg_interval_width = np.mean(upper - lower)

    metrics = {
        "MSE_mean": mse,
        "RMSE_mean": rmse,
        "MAE_mean": mae,
        "Coverage_95": coverage_95,
        "Avg_Interval_Width": avg_interval_width,
    }
    return metrics


def evaluate_mc_dropout(model, data_loader, mc_samples=100, device="cpu"):
    mean_preds, std_preds, all_preds, targets = predict_mc_dropout(
        model=model,
        data_loader=data_loader,
        mc_samples=mc_samples,
        device=device,
    )

    metrics = compute_probabilistic_metrics(mean_preds, std_preds, targets)

    print("\nMC Dropout evaluation metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")

    lower = mean_preds - 1.96 * std_preds
    upper = mean_preds + 1.96 * std_preds

    return {
        "mean_preds": mean_preds,
        "std_preds": std_preds,
        "all_preds": all_preds,
        "targets": targets,
        "lower_95": lower,
        "upper_95": upper,
        "metrics": metrics,
    }