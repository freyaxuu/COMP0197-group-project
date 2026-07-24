import torch
import torch.nn as nn

class BiLSTM(nn.Module):
    """
    Baseline bidirectional LSTM for time series forecasting.
    Predicts a single value (next timestep).

    Architecture:
    - uses a bidirectional LSTM encoder to process the input sequence
    - concatenates the final forward and backward hidden states
    - applies dropout on the combined sequence representation
    - maps the result to a single scalar prediction through a linear layer
    """
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True  
        )
        self.dropout = nn.Dropout(dropout)
        # bi → hidden_dim * 2
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        out, (h_n, _) = self.lstm(x)

        # h_n shape: (num_layers * 2, batch, hidden_dim)

        forward_last = h_n[-2]
        backward_last = h_n[-1]

        out = torch.cat((forward_last, backward_last), dim=1)
        out = self.dropout(out)
        out = self.fc(out)
        return out