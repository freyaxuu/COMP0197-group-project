import numpy as np
import torch


def _compute_regression_metrics(preds, targets):
    """
    Compute MAE and RMSE for tensors.
    """
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
    early_stopping_patience=5,
):
    """
    Train a deterministic regression model with early stopping and checkpoint
    saving.

    The function performs epoch-wise training and validation, tracks loss and
    regression metrics, saves the best-performing checkpoint based on
    validation loss, and returns the full training history.

    Returns
    -------
    dict
        Training history containing epoch-wise loss and metric values for
        both training and validation splits.
    """
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
        # -------------------------
        # Training
        # -------------------------
        model.train()
        train_batch_losses = []
        train_batch_mae = []
        train_batch_rmse = []

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            output = model(batch_x).squeeze()
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

        # -------------------------
        # Validation
        # -------------------------
        model.eval()
        val_batch_losses = []
        val_batch_mae = []
        val_batch_rmse = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                output = model(batch_x).squeeze()
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

        # -------------------------
        # Save best model
        # -------------------------
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

        # -------------------------
        # Early stopping
        # -------------------------
        if early_stopping_patience is not None and patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"Best model was from epoch {best_epoch} with val loss {best_val_loss:.4f}")
    return history


def gaussian_nll_from_logvar(mean, log_var, target):
    """
    Gaussian negative log-likelihood with predicted log-variance.
    """
    var = torch.exp(log_var)
    loss = 0.5 * (log_var + ((target - mean) ** 2) / var)
    return loss.mean()


def train_model_gaussian(
    model,
    train_loader,
    val_loader,
    optimizer,
    epochs,
    device="cpu",
    save_path="best_gaussian_model.pt",
    early_stopping_patience=5,
):
    """
    Training loop for Gaussian-output regression model.
    """
    model.to(device)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_mae": [],
        "val_mae": [],
        "train_rmse": [],
        "val_rmse": [],
        "train_avg_sigma": [],
        "val_avg_sigma": [],
    }

    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0

    for epoch in range(epochs):
        # -------------------------
        # Training
        # -------------------------
        model.train()
        train_batch_losses = []
        train_batch_mae = []
        train_batch_rmse = []
        train_batch_sigma = []

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).reshape(-1)

            optimizer.zero_grad()

            mean, log_var = model(batch_x)
            mean = mean.reshape(-1)
            log_var = log_var.reshape(-1)

            loss = gaussian_nll_from_logvar(mean, log_var, batch_y)
            loss.backward()
            optimizer.step()

            mae = torch.mean(torch.abs(mean - batch_y)).item()
            rmse = torch.sqrt(torch.mean((mean - batch_y) ** 2)).item()
            avg_sigma = torch.mean(torch.exp(0.5 * log_var)).item()

            train_batch_losses.append(loss.item())
            train_batch_mae.append(mae)
            train_batch_rmse.append(rmse)
            train_batch_sigma.append(avg_sigma)

        avg_train_loss = np.mean(train_batch_losses)
        avg_train_mae = np.mean(train_batch_mae)
        avg_train_rmse = np.mean(train_batch_rmse)
        avg_train_sigma = np.mean(train_batch_sigma)

        # -------------------------
        # Validation
        # -------------------------
        model.eval()
        val_batch_losses = []
        val_batch_mae = []
        val_batch_rmse = []
        val_batch_sigma = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device).reshape(-1)

                mean, log_var = model(batch_x)
                mean = mean.reshape(-1)
                log_var = log_var.reshape(-1)

                loss = gaussian_nll_from_logvar(mean, log_var, batch_y)

                mae = torch.mean(torch.abs(mean - batch_y)).item()
                rmse = torch.sqrt(torch.mean((mean - batch_y) ** 2)).item()
                avg_sigma = torch.mean(torch.exp(0.5 * log_var)).item()

                val_batch_losses.append(loss.item())
                val_batch_mae.append(mae)
                val_batch_rmse.append(rmse)
                val_batch_sigma.append(avg_sigma)

        avg_val_loss = np.mean(val_batch_losses)
        avg_val_mae = np.mean(val_batch_mae)
        avg_val_rmse = np.mean(val_batch_rmse)
        avg_val_sigma = np.mean(val_batch_sigma)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_mae"].append(avg_train_mae)
        history["val_mae"].append(avg_val_mae)
        history["train_rmse"].append(avg_train_rmse)
        history["val_rmse"].append(avg_val_rmse)
        history["train_avg_sigma"].append(avg_train_sigma)
        history["val_avg_sigma"].append(avg_val_sigma)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"Train MAE: {avg_train_mae:.4f} | Val MAE: {avg_val_mae:.4f} | "
            f"Train RMSE: {avg_train_rmse:.4f} | Val RMSE: {avg_val_rmse:.4f} | "
            f"Train Avg Sigma: {avg_train_sigma:.4f} | Val Avg Sigma: {avg_val_sigma:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            patience_counter = 0

            checkpoint = {
                "epoch": best_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "history": history,
            }
            torch.save(checkpoint, save_path)
            print(f"Saved best Gaussian model at epoch {best_epoch} with val loss {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if early_stopping_patience is not None and patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                print(f"Best Gaussian model was from epoch {best_epoch} with val loss {best_val_loss:.4f}")
                break

    return history