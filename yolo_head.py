import torch
import torch.nn as nn
from config import Config


class YOLOHeadWithContext(nn.Module):
    """
    Упрощённая head для последнего кадра.
    Возвращает карту предсказаний [B, num_classes*5, H, W].
    """
    def __init__(
        self,
        feature_channels: int = Config.YOLO_FEATURE_CHANNELS,
        context_dim: int = Config.FUSION_SIZE,
        ae_dim: int = Config.AE_EMB_SIZE,
        num_classes: int = 4,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.feature_channels = feature_channels

        self.context_proj = nn.Sequential(
            nn.Linear(context_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
        )

        self.ae_proj = nn.Sequential(
            nn.Linear(ae_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
        )

        self.detect_head = nn.Sequential(
            nn.Conv2d(feature_channels + 256 + 128, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes * 5, kernel_size=1),
        )

    def forward(
        self,
        features_map: torch.Tensor,
        context: torch.Tensor,
        ae_emb: torch.Tensor
    ) -> torch.Tensor:
        if features_map.ndim != 4:
            raise ValueError(
                f"YOLOHead ожидает 4D features_map, получено: {tuple(features_map.shape)}"
            )

        batch, channels, height, width = features_map.shape
        if channels != self.feature_channels:
            raise ValueError(
                f"YOLOHead ожидает {self.feature_channels} каналов от YOLO, "
                f"но получил {channels}. Проверь слой hook в complete_model_fixed.py"
            )

        context_vec = self.context_proj(context).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, height, width)
        ae_vec = self.ae_proj(ae_emb).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, height, width)

        fused = torch.cat([features_map, context_vec, ae_vec], dim=1)
        return self.detect_head(fused)