import torch
import torch.nn as nn

class GaussianBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout,
                 min_log_var=-14.0, max_log_var=-10.0):
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

        self.fc_mu = nn.Linear(hidden_dim * 2, 1)
        self.fc_logvar = nn.Linear(hidden_dim * 2, 1)

        self.min_log_var = min_log_var
        self.max_log_var = max_log_var

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)

        forward_last = h_n[-2]
        backward_last = h_n[-1]

        out = torch.cat((forward_last, backward_last), dim=1)
        out = self.dropout(out)

        mu = self.fc_mu(out)
        log_var = self.fc_logvar(out)
        log_var = torch.clamp(log_var, self.min_log_var, self.max_log_var)

        return mu, log_var