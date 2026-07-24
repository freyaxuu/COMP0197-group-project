import torch
import torch.nn as nn


class DeterministicLSTM(nn.Module):
    """
    Baseline deterministic LSTM for time series forecasting.
    Predicts a single value (next timestep).

    MC Dropout-compatible version:
    - keep the original LSTM backbone
    - add an explicit dropout layer on the final hidden representation
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.dropout_rate = dropout

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # NEW: explicit dropout on final hidden state
        self.dropout = nn.Dropout(p=dropout)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        x shape: (batch, seq_len, input_dim)
        """
        lstm_out, _ = self.lstm(x)

        # take final timestep output
        last_hidden = lstm_out[:, -1, :]

        # NEW: dropout before final linear layer
        last_hidden = self.dropout(last_hidden)

        output = self.fc(last_hidden)

        return output
    



class GaussianLSTM(nn.Module):
    """
    LSTM with Gaussian output for heteroscedastic regression.

    Outputs
    -------
    mean : torch.Tensor
        Predictive mean, shape (batch, 1)
    log_var : torch.Tensor
        Predictive log-variance, shape (batch, 1)

    Notes
    -----
    - Still MC Dropout-compatible because dropout stays in the network.
    - At inference, repeated stochastic forward passes can be used to estimate:
        epistemic uncertainty from variation in mean outputs,
        aleatoric uncertainty from predicted variance.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        min_log_var: float = -14.0,
        max_log_var: float = -10.0,
    ):
        super().__init__()

        self.dropout_rate = dropout
        self.min_log_var = min_log_var
        self.max_log_var = max_log_var

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(p=dropout)

        self.mean_head = nn.Linear(hidden_dim, 1)
        self.raw_log_var_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, seq_len, input_dim)

        Returns
        -------
        mean : torch.Tensor
            Shape (batch, 1)
        log_var : torch.Tensor
            Shape (batch, 1)
        """
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        last_hidden = self.dropout(last_hidden)

        mean = self.mean_head(last_hidden)
        raw_log_var = self.raw_log_var_head(last_hidden)

       # Smooth bounded mapping instead of hard clamp
        log_var = self.min_log_var + (self.max_log_var - self.min_log_var) * torch.sigmoid(raw_log_var)

        return mean, log_var