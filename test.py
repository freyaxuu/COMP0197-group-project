from pathlib import Path
import sys
import json
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from fx_forecasting.data.load import load_fx_csv
from fx_forecasting.data.preprocess import prepare_fx_data, inverse_transform_target
from fx_forecasting.models.baseline import DeterministicLSTM
from fx_forecasting.models.probabilistic import MCDropoutLSTM
from fx_forecasting.training.evaluate import evaluate_model, evaluate_mc_dropout
from fx_forecasting.visualization.plots import plot_predictions, plot_probabilistic_predictions

def to_python_scalars(d):
    return {
        k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
        for k, v in d.items()
    }

def main():
    DATA_PATH = PROJECT_ROOT / "data" / "raw" / "main_with_exo_2.csv"
    TARGET = "GBP-CNY"

    LOOKBACK = 30
    HIDDEN_DIM = 64
    NUM_LAYERS = 2
    DROPOUT = 0.2
    BATCH_SIZE = 64

    MODEL_TYPE = "mc_dropout"   # "deterministic" or "mc_dropout"
    MC_SAMPLES = 100

    df = load_fx_csv(DATA_PATH, date_col="Date")
    FX_ONLY_COLS = [
    "timestamp",
    "GBP-CNY",
    "GBP-USD",
    "GBP-EUR",
    "GBP-JPY",
    "GBP-KRW",
    "GBP-CHF",
]
    df = df[FX_ONLY_COLS].copy()

    X_train, y_train, X_test, y_test, scaler, feature_columns = prepare_fx_data(
        df,
        target_col=TARGET,
        test_ratio=0.2,
        add_returns=True,
        add_ma=False,
        add_volatility=False,
        scale=True,
        scaler_type="standard",
        make_windows=True,
        lookback=LOOKBACK,
    )

    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)
    test_loader = DataLoader(
        TensorDataset(X_test_t, y_test_t),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if MODEL_TYPE == "deterministic":
        model = DeterministicLSTM(
            input_dim=X_train.shape[-1],
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
        )
        ckpt_path = PROJECT_ROOT / "outputs" / "models" / "best_deterministic_lstm.pt"

        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        preds, targets, metrics = evaluate_model(model, test_loader, device=device)

        preds_original = inverse_transform_target(
            preds, scaler=scaler, columns=feature_columns, target_col=TARGET
        )
        targets_original = inverse_transform_target(
            targets, scaler=scaler, columns=feature_columns, target_col=TARGET
        )
        y_train_original = inverse_transform_target(
            y_train, scaler=scaler, columns=feature_columns, target_col=TARGET
        )

        plot_predictions(
            preds=preds_original,
            targets=targets_original,
            train_targets=y_train_original,
        )

        out_metrics = {
            "model_type": MODEL_TYPE,
            **to_python_scalars(metrics),
        }

    elif MODEL_TYPE == "mc_dropout":
        model = MCDropoutLSTM(
            input_dim=X_train.shape[-1],
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
        )
        ckpt_path = PROJECT_ROOT / "outputs" / "models" / "best_mc_dropout_lstm.pt"

        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        results = evaluate_mc_dropout(
            model=model,
            data_loader=test_loader,
            mc_samples=MC_SAMPLES,
            device=device,
        )

        mean_preds = results["mean_preds"]
        std_preds = results["std_preds"]
        targets = results["targets"]
        lower_95 = results["lower_95"]
        upper_95 = results["upper_95"]

        # back-transform mean and target to original scale
        mean_preds_original = inverse_transform_target(
            mean_preds, scaler=scaler, columns=feature_columns, target_col=TARGET
        )
        targets_original = inverse_transform_target(
            targets, scaler=scaler, columns=feature_columns, target_col=TARGET
        )

        # back-transform interval bounds to original scale
        lower_original = inverse_transform_target(
            lower_95, scaler=scaler, columns=feature_columns, target_col=TARGET
        )
        upper_original = inverse_transform_target(
            upper_95, scaler=scaler, columns=feature_columns, target_col=TARGET
        )

        # recover predictive std on original target scale
        target_idx = feature_columns.index(TARGET)
        target_scale = scaler.scale_[target_idx]
        std_preds_original = std_preds * target_scale

        plot_probabilistic_predictions(
            mean_preds=mean_preds_original,
            std_preds=std_preds_original,
            targets=targets_original,
            lower=lower_original,
            upper=upper_original,
            title="MC Dropout Forecast with 95% Interval",
        )

        # distribution output for ALL test points
        distribution_df = {
            "target": targets_original,
            "pred_mean": mean_preds_original,
            "pred_std": std_preds_original,
            "lower_95": lower_original,
            "upper_95": upper_original,
        }

        import pandas as pd
        distribution_df = pd.DataFrame(distribution_df)

        out_metrics = {
            "model_type": MODEL_TYPE,
            **to_python_scalars(results["metrics"]),
            "Mean_Predictive_STD": float(np.mean(std_preds_original)),
            "Median_Predictive_STD": float(np.median(std_preds_original)),
            "Min_Predictive_STD": float(np.min(std_preds_original)),
            "Max_Predictive_STD": float(np.max(std_preds_original)),
            "Num_Test_Points": int(len(mean_preds_original)),
        }

    else:
        raise ValueError("MODEL_TYPE must be 'deterministic' or 'mc_dropout'")

    metrics_dir = PROJECT_ROOT / "outputs" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    if MODEL_TYPE == "mc_dropout":
        dist_path = metrics_dir / "mc_dropout_distribution.csv"
        distribution_df.to_csv(dist_path, index=False)
        print("\nSaved full distribution output to:", dist_path)
        print("Distribution output shape:", distribution_df.shape)

        print("\nFirst 5 test-point distributions:")
        print(distribution_df.head())

    out_path = metrics_dir / f"{MODEL_TYPE}_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_metrics, f, indent=2)

    print("\nSaved metrics to:", out_path)


if __name__ == "__main__":
    main()