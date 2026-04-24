import torch
import torch.nn as nn
from config import Config


class FusionLayer(nn.Module):
    """
    Объединяет:
    - yolo_feat: [B, 256]
    - ae_emb:    [B, 256]
    - ae_error:  [B] / [B, 1]
    Возвращает:
    - fused:     [B, 512]
    """
    def __init__(
        self,
        yolo_dim: int = Config.YOLO_FEATURE_CHANNELS,
        ae_dim: int = Config.AE_EMB_SIZE,
        output_dim: int = Config.FUSION_SIZE,
        dropout: float = 0.1,
    ):
        super().__init__()
        input_dim = yolo_dim + ae_dim + 1

        self.fusion = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        yolo_feat: torch.Tensor,
        ae_emb: torch.Tensor,
        ae_error: torch.Tensor
    ) -> torch.Tensor:
        if ae_error.ndim == 1:
            ae_error = ae_error.unsqueeze(1)

        fused = torch.cat([yolo_feat, ae_emb, ae_error], dim=1)
        return self.fusion(fused)