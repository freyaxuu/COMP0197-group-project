import torch
import torch.nn as nn


class DualBranchLSTM(nn.Module):
    """
    Dual-branch LSTM for exchange rate forecasting.

    Structure:
    - one LSTM branch for market-level features
    - one LSTM branch for macro-level features
    - concatenate the two final hidden representations
    - apply dropout + fully connected layer for next-step prediction

    Input:
        x_market: (batch, seq_len, market_input_dim)
        x_macro : (batch, seq_len, macro_input_dim)

    Output:
        prediction: (batch, output_dim)
    """

    def __init__(
        self,
        market_input_dim: int,
        macro_input_dim: int,
        hidden_dim_market: int = 128,
        hidden_dim_macro: int = 128,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.market_input_dim = market_input_dim
        self.macro_input_dim = macro_input_dim
        self.hidden_dim_market = hidden_dim_market
        self.hidden_dim_macro = hidden_dim_macro
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout_rate = dropout

        # Market branch
        self.market_lstm = nn.LSTM(
            input_size=market_input_dim,
            hidden_size=hidden_dim_market,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Macro branch
        self.macro_lstm = nn.LSTM(
            input_size=macro_input_dim,
            hidden_size=hidden_dim_macro,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Explicit dropout on concatenated hidden representation
        self.dropout = nn.Dropout(p=dropout)

        # Final prediction layer
        self.fc = nn.Linear(hidden_dim_market + hidden_dim_macro, output_dim)

    def forward(self, x_market: torch.Tensor, x_macro: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_market: Tensor of shape (batch, seq_len, market_input_dim)
            x_macro : Tensor of shape (batch, seq_len, macro_input_dim)

        Returns:
            output: Tensor of shape (batch, output_dim)
        """
        # Market branch
        market_out, _ = self.market_lstm(x_market)
        market_hidden = market_out[:, -1, :]   # (batch, hidden_dim_market)

        # Macro branch
        macro_out, _ = self.macro_lstm(x_macro)
        macro_hidden = macro_out[:, -1, :]     # (batch, hidden_dim_macro)

        # Concatenate branch representations
        combined_hidden = torch.cat([market_hidden, macro_hidden], dim=1)

        # Dropout + final linear layer
        combined_hidden = self.dropout(combined_hidden)
        output = self.fc(combined_hidden)

        return output
    

class GaussianDualBranchLSTM(nn.Module):
    """
    Dual-branch LSTM with Gaussian (mean, log_var) output.
    Returns:
        mean: (B, 1)
        log_var: (B, 1)
    """
    def __init__(
        self,
        market_input_dim: int,
        macro_input_dim: int,
        hidden_dim_market: int = 128,
        hidden_dim_macro: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        min_log_var: float = -14.0,
        max_log_var: float = -10.0,
    ):
        super().__init__()
        self.min_log_var = min_log_var
        self.max_log_var = max_log_var

        self.market_lstm = nn.LSTM(
            input_size=market_input_dim,
            hidden_size=hidden_dim_market,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.macro_lstm = nn.LSTM(
            input_size=macro_input_dim,
            hidden_size=hidden_dim_macro,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(p=dropout)

        fused_dim = hidden_dim_market + hidden_dim_macro
        self.mean_head = nn.Linear(fused_dim, 1)
        self.raw_log_var_head = nn.Linear(fused_dim, 1)

    def forward(self, x_market: torch.Tensor, x_macro: torch.Tensor):
        market_out, _ = self.market_lstm(x_market)
        macro_out, _ = self.macro_lstm(x_macro)

        market_hidden = market_out[:, -1, :]
        macro_hidden = macro_out[:, -1, :]

        h = torch.cat([market_hidden, macro_hidden], dim=1)
        h = self.dropout(h)

        mean = self.mean_head(h)
        raw_log_var = self.raw_log_var_head(h)

        # bounded log-variance for stability
        log_var = self.min_log_var + (self.max_log_var - self.min_log_var) * torch.sigmoid(raw_log_var)
        return mean, log_var