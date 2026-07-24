import torch
import torch.nn as nn


class DCLSTM(nn.Module):
    """
    Deep Coupled LSTM for exchange rate forecasting.

    Architecture:
    1) market features  -> market LSTM
    2) macro features   -> macro LSTM
    3) concatenate the two branch hidden sequences at each timestep
    4) feed concatenated sequence into a coupling LSTM
    5) take final hidden representation -> dropout -> fully connected -> prediction

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
        hidden_dim_market: int = 64,
        hidden_dim_macro: int = 64,
        hidden_dim_coupling: int = 64,
        num_layers_branch: int = 1,
        num_layers_coupling: int = 1,
        output_dim: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.market_input_dim = market_input_dim
        self.macro_input_dim = macro_input_dim
        self.hidden_dim_market = hidden_dim_market
        self.hidden_dim_macro = hidden_dim_macro
        self.hidden_dim_coupling = hidden_dim_coupling
        self.num_layers_branch = num_layers_branch
        self.num_layers_coupling = num_layers_coupling
        self.output_dim = output_dim
        self.dropout_rate = dropout

        # Branch 1: market-level sequence encoder
        self.market_lstm = nn.LSTM(
            input_size=market_input_dim,
            hidden_size=hidden_dim_market,
            num_layers=num_layers_branch,
            batch_first=True,
            dropout=dropout if num_layers_branch > 1 else 0.0,
        )

        # Branch 2: macro-level sequence encoder
        self.macro_lstm = nn.LSTM(
            input_size=macro_input_dim,
            hidden_size=hidden_dim_macro,
            num_layers=num_layers_branch,
            batch_first=True,
            dropout=dropout if num_layers_branch > 1 else 0.0,
        )

        # Coupling LSTM: learn deep interaction between market + macro branches
        self.coupling_lstm = nn.LSTM(
            input_size=hidden_dim_market + hidden_dim_macro,
            hidden_size=hidden_dim_coupling,
            num_layers=num_layers_coupling,
            batch_first=True,
            dropout=dropout if num_layers_coupling > 1 else 0.0,
        )

        # Explicit dropout on final coupling representation
        self.dropout = nn.Dropout(p=dropout)

        # Final prediction head
        self.fc = nn.Linear(hidden_dim_coupling, output_dim)

    def forward(self, x_market: torch.Tensor, x_macro: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_market: Tensor of shape (batch, seq_len, market_input_dim)
            x_macro : Tensor of shape (batch, seq_len, macro_input_dim)

        Returns:
            output: Tensor of shape (batch, output_dim)
        """
        assert x_market.dim() == 3, f"x_market must be 3D, got shape {x_market.shape}"
        assert x_macro.dim() == 3, f"x_macro must be 3D, got shape {x_macro.shape}"
        assert x_market.shape[-1] == self.market_input_dim, (
            f"Expected market_input_dim={self.market_input_dim}, got {x_market.shape[-1]}"
        )
        assert x_macro.shape[-1] == self.macro_input_dim, (
            f"Expected macro_input_dim={self.macro_input_dim}, got {x_macro.shape[-1]}"
        )
        assert x_market.shape[0] == x_macro.shape[0], "Batch size mismatch between market and macro inputs"
        assert x_market.shape[1] == x_macro.shape[1], "Sequence length mismatch between market and macro inputs"

        # Branch encoders
        market_seq, _ = self.market_lstm(x_market)   # (batch, seq_len, hidden_dim_market)
        macro_seq, _ = self.macro_lstm(x_macro)      # (batch, seq_len, hidden_dim_macro)

        # Deep coupling across two branches at each timestep
        coupling_input = torch.cat([market_seq, macro_seq], dim=2)
        # shape: (batch, seq_len, hidden_dim_market + hidden_dim_macro)

        coupling_seq, _ = self.coupling_lstm(coupling_input)
        # shape: (batch, seq_len, hidden_dim_coupling)

        # Sequence-to-one forecasting: use final timestep representation
        final_hidden = coupling_seq[:, -1, :]        # (batch, hidden_dim_coupling)
    

        final_hidden = self.dropout(final_hidden)
        output = self.fc(final_hidden)

        return output
    

class GaussianDCLSTM(nn.Module):
    """
    Deep Coupled LSTM with Gaussian output for heteroscedastic forecasting.

    Architecture:
    1) market features  -> market LSTM
    2) macro features   -> macro LSTM
    3) concatenate branch hidden sequences at each timestep
    4) feed concatenated sequence into a coupling LSTM
    5) take final hidden representation -> dropout
    6) two output heads:
        - mean head
        - log-variance head

    Input:
        x_market: (batch, seq_len, market_input_dim)
        x_macro : (batch, seq_len, macro_input_dim)

    Output:
        mean    : (batch, 1)
        log_var : (batch, 1)

    Notes
    -----
    - MC Dropout-compatible because dropout remains active in the network.
    - Repeated stochastic forward passes at inference can estimate:
        epistemic uncertainty from variability in mean predictions,
        aleatoric uncertainty from predicted variance.
    """

    def __init__(
        self,
        market_input_dim: int,
        macro_input_dim: int,
        hidden_dim_market: int = 64,
        hidden_dim_macro: int = 64,
        hidden_dim_coupling: int = 64,
        num_layers_branch: int = 1,
        num_layers_coupling: int = 1,
        dropout: float = 0.1,
        min_log_var: float = -14.0,
        max_log_var: float = -10.0,
    ):
        super().__init__()

        self.market_input_dim = market_input_dim
        self.macro_input_dim = macro_input_dim
        self.hidden_dim_market = hidden_dim_market
        self.hidden_dim_macro = hidden_dim_macro
        self.hidden_dim_coupling = hidden_dim_coupling
        self.num_layers_branch = num_layers_branch
        self.num_layers_coupling = num_layers_coupling
        self.dropout_rate = dropout
        self.min_log_var = min_log_var
        self.max_log_var = max_log_var

        # Branch 1: market-level sequence encoder
        self.market_lstm = nn.LSTM(
            input_size=market_input_dim,
            hidden_size=hidden_dim_market,
            num_layers=num_layers_branch,
            batch_first=True,
            dropout=dropout if num_layers_branch > 1 else 0.0,
        )

        # Branch 2: macro-level sequence encoder
        self.macro_lstm = nn.LSTM(
            input_size=macro_input_dim,
            hidden_size=hidden_dim_macro,
            num_layers=num_layers_branch,
            batch_first=True,
            dropout=dropout if num_layers_branch > 1 else 0.0,
        )

        # Coupling LSTM
        self.coupling_lstm = nn.LSTM(
            input_size=hidden_dim_market + hidden_dim_macro,
            hidden_size=hidden_dim_coupling,
            num_layers=num_layers_coupling,
            batch_first=True,
            dropout=dropout if num_layers_coupling > 1 else 0.0,
        )

        # Explicit dropout on final coupling representation
        self.dropout = nn.Dropout(p=dropout)

        # Gaussian output heads
        self.mean_head = nn.Linear(hidden_dim_coupling, 1)
        self.raw_log_var_head = nn.Linear(hidden_dim_coupling, 1)

    def forward(self, x_market: torch.Tensor, x_macro: torch.Tensor):
        """
        Args:
            x_market: Tensor of shape (batch, seq_len, market_input_dim)
            x_macro : Tensor of shape (batch, seq_len, macro_input_dim)

        Returns:
            mean    : Tensor of shape (batch, 1)
            log_var : Tensor of shape (batch, 1)
        """
        assert x_market.dim() == 3, f"x_market must be 3D, got shape {x_market.shape}"
        assert x_macro.dim() == 3, f"x_macro must be 3D, got shape {x_macro.shape}"
        assert x_market.shape[-1] == self.market_input_dim, (
            f"Expected market_input_dim={self.market_input_dim}, got {x_market.shape[-1]}"
        )
        assert x_macro.shape[-1] == self.macro_input_dim, (
            f"Expected macro_input_dim={self.macro_input_dim}, got {x_macro.shape[-1]}"
        )
        assert x_market.shape[0] == x_macro.shape[0], (
            "Batch size mismatch between market and macro inputs"
        )
        assert x_market.shape[1] == x_macro.shape[1], (
            "Sequence length mismatch between market and macro inputs"
        )

        # Branch encoders
        market_seq, _ = self.market_lstm(x_market)
        macro_seq, _ = self.macro_lstm(x_macro)

        # Deep coupling
        coupling_input = torch.cat([market_seq, macro_seq], dim=2)
        coupling_seq, _ = self.coupling_lstm(coupling_input)

        # Final timestep representation
        final_hidden = coupling_seq[:, -1, :]

        # Dropout for MC Dropout compatibility
        final_hidden = self.dropout(final_hidden)

        # Gaussian heads
        mean = self.mean_head(final_hidden)
        raw_log_var = self.raw_log_var_head(final_hidden)

        # Smooth bounded mapping instead of hard clamp
        log_var = self.min_log_var + (self.max_log_var - self.min_log_var) * torch.sigmoid(raw_log_var)

        return mean, log_var