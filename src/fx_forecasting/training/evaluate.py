# src/fx_forecasting/evaluation/evaluate.py

import torch
import numpy as np


def predict(model, data_loader, device="cpu"):
    """
    Generate predictions for a dataset.
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
    Full evaluation pipeline.
    """

    preds, targets = predict(model, data_loader, device)

    metrics = compute_metrics(preds, targets)

    print("\nEvaluation metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")

    return preds, targets, metrics