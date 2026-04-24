import torch
import torch.nn as nn
from config import Config


class TemporalLSTM(nn.Module):
    """
    Временная модель:
    вход  [B, T, 512]
    выход:
      context  [B, 512]
      lstm_out [B, T, 512]
      attn_w   [B, T]
    """
    def __init__(
        self,
        input_size: int = Config.FUSION_SIZE,
        hidden_size: int = Config.LSTM_HIDDEN,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.output_size = hidden_size * 2

        self.attention = nn.Sequential(
            nn.Linear(self.output_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor):
        lstm_out, _ = self.lstm(x)  # [B, T, 512]
        attn_logits = self.attention(lstm_out).squeeze(-1)  # [B, T]
        attn_weights = torch.softmax(attn_logits, dim=1)
        context = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1)
        return context, lstm_out, attn_weights