# src/fx_forecasting/models/deterministic_lstm.py

import torch
import torch.nn as nn


class DeterministicLSTM(nn.Module):
    """
    Baseline deterministic LSTM for time series forecasting.
    Predicts a single value (next timestep).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        x shape: (batch, seq_len, input_dim)
        """

        lstm_out, _ = self.lstm(x)

        # take final timestep output
        last_hidden = lstm_out[:, -1, :]

        output = self.fc(last_hidden)

        return output