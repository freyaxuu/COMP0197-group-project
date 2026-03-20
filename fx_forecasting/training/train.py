import numpy as np
import torch
from tqdm import tqdm


def _compute_regression_metrics(preds, targets):
    mae = torch.mean(torch.abs(preds - targets)).item()
    rmse = torch.sqrt(torch.mean((preds - targets) ** 2)).item()
    return mae, rmse


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    epochs,
    device="cpu",
    save_path="best_model.pt",
    early_stopping_patience=5
):
    model.to(device)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_mae": [],
        "val_mae": [],
        "train_rmse": [],
        "val_rmse": [],
    }

    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_batch_losses = []
        train_batch_mae = []
        train_batch_rmse = []

        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).view(-1)

            optimizer.zero_grad()

            output = model(batch_x).view(-1)
            loss = criterion(output, batch_y)

            loss.backward()
            optimizer.step()

            mae, rmse = _compute_regression_metrics(output, batch_y)

            train_batch_losses.append(loss.item())
            train_batch_mae.append(mae)
            train_batch_rmse.append(rmse)

        avg_train_loss = np.mean(train_batch_losses)
        avg_train_mae = np.mean(train_batch_mae)
        avg_train_rmse = np.mean(train_batch_rmse)

        model.eval()
        val_batch_losses = []
        val_batch_mae = []
        val_batch_rmse = []

        with torch.no_grad():
            for batch_x, batch_y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device).view(-1)

                output = model(batch_x).view(-1)
                loss = criterion(output, batch_y)

                mae, rmse = _compute_regression_metrics(output, batch_y)

                val_batch_losses.append(loss.item())
                val_batch_mae.append(mae)
                val_batch_rmse.append(rmse)

        avg_val_loss = np.mean(val_batch_losses)
        avg_val_mae = np.mean(val_batch_mae)
        avg_val_rmse = np.mean(val_batch_rmse)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_mae"].append(avg_train_mae)
        history["val_mae"].append(avg_val_mae)
        history["train_rmse"].append(avg_train_rmse)
        history["val_rmse"].append(avg_val_rmse)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"Train MAE: {avg_train_mae:.4f} | Val MAE: {avg_val_mae:.4f} | "
            f"Train RMSE: {avg_train_rmse:.4f} | Val RMSE: {avg_val_rmse:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            patience_counter = 0

            torch.save(
                {
                    "epoch": best_epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "history": history,
                },
                save_path,
            )

            print(f"Saved best model at epoch {best_epoch} with val loss {best_val_loss:.4f}")
        else:
            patience_counter += 1

        if early_stopping_patience is not None and patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"Best model was from epoch {best_epoch} with val loss {best_val_loss:.4f}")
    return history