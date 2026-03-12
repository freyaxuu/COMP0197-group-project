import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

def prepare_data(df, target_col, feature_cols, seq_length, train_split=0.8, batch_size=64):
    """
    Generalized sequence creator with NO data leakage.
    """
    # 1. Extract values
    data = df[feature_cols].values
    target_idx = feature_cols.index(target_col)
    
    # 2. Chronological Split
    split_idx = int(train_split * len(data))
    train_data_raw = data[:split_idx]
    test_data_raw = data[split_idx:]
    
    # 3. Scale 
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data_raw)
    test_data_scaled = scaler.transform(test_data_raw)
    
    full_data_scaled = np.vstack((train_data_scaled, test_data_scaled))
    
    # 4. Create windows
    xs, ys = [], []
    for i in range(len(full_data_scaled) - seq_length):
        xs.append(full_data_scaled[i:(i + seq_length)])
        ys.append(full_data_scaled[i + seq_length, target_idx])
    
    X, y = np.array(xs), np.array(ys)
    
    # 5. Split sequences into Tensors
    seq_split = split_idx - seq_length
    X_train, X_test = torch.FloatTensor(X[:seq_split]), torch.FloatTensor(X[seq_split:])
    y_train, y_test = torch.FloatTensor(y[:seq_split]), torch.FloatTensor(y[seq_split:])
    
    # 6. DataLoaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, scaler


def plot_time_series(df, col, title="Time Series"):
    plt.figure(figsize=(15, 5))
    plt.plot(df[col], color='royalblue', linewidth=0.8)
    plt.title(title)
    plt.ylabel(col)
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_results(targets, preds, sigma=None, title="Model Evaluation"):
    """
    Visualizes predictions. Supports both deterministic (preds only) 
    and probabilistic (preds + sigma) plotting.
    """
    plt.figure(figsize=(15, 6))
    plt.plot(targets, label="Actual", color='black', alpha=0.7)
    plt.plot(preds, label="Predicted Mean", color='red', linestyle='--')
    
    if sigma is not None:
        # Plot 95% confidence interval (approx 2 standard deviations)
        plt.fill_between(
            range(len(preds)), 
            preds - 2*sigma, 
            preds + 2*sigma, 
            color='red', alpha=0.2, label="95% Confidence (Aleatoric)"
        )
    
    plt.title(title)
    plt.legend()
    plt.show()