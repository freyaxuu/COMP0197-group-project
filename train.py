from pathlib import Path
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from fx_forecasting.data.load import load_fx_csv
from fx_forecasting.data.inspect import inspect_data
from fx_forecasting.data.preprocess import prepare_fx_data
from fx_forecasting.models.baseline import DeterministicLSTM
from fx_forecasting.models.probabilistic import MCDropoutLSTM
from fx_forecasting.training.train import train_model
from fx_forecasting.visualization.plots import plot_training_history


def main():
    DATA_PATH = PROJECT_ROOT / "data" / "raw" / "main_with_exo_2.csv"
    TARGET = "GBP-CNY"

    LOOKBACK = 30
    HIDDEN_DIM = 64
    NUM_LAYERS = 2
    DROPOUT = 0.2
    LR = 1e-3
    EPOCHS = 50
    BATCH_SIZE = 64
    MODEL_TYPE = "mc_dropout"   # "deterministic" or "mc_dropout"

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
    inspect_data(df)

    from fx_forecasting.visualization.plots import (
    plot_target_series,
    plot_histogram,
    plot_correlation_heatmap,
)
    plot_target_series(df, target_col=TARGET)
    plot_histogram(df, col=TARGET)
    plot_correlation_heatmap(df)

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

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_test_t, y_test_t),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    if MODEL_TYPE == "deterministic":
        model = DeterministicLSTM(
            input_dim=X_train.shape[-1],
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
        )
        save_path = PROJECT_ROOT / "outputs" / "models" / "best_deterministic_lstm.pt"
    elif MODEL_TYPE == "mc_dropout":
        model = MCDropoutLSTM(
            input_dim=X_train.shape[-1],
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
        )
        save_path = PROJECT_ROOT / "outputs" / "models" / "best_mc_dropout_lstm.pt"
    else:
        raise ValueError("MODEL_TYPE must be 'deterministic' or 'mc_dropout'")

    save_path.parent.mkdir(parents=True, exist_ok=True)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=EPOCHS,
        device=device,
        save_path=str(save_path),
        early_stopping_patience=5,
    )

    print("\nTraining finished.")
    print("Saved model to:", save_path)
    print("Feature columns:", feature_columns)
    
    plot_training_history(history, metric="loss")
    plot_training_history(history, metric="mae")
    plot_training_history(history, metric="rmse")

if __name__ == "__main__":
    main()