from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO

from config import Config
from fusion_layer import FusionLayer
from temporal_model import TemporalLSTM
from classifier import DefectClassifier
from yolo_head import YOLOHeadWithContext
from train_autoencoder import Autoencoder


class YoloBackboneExtractor(nn.Module):
    """
    Извлекает:
    - intermediate feature map [B, 256, H, W]
    - pooled embedding        [B, 256]

    Используется hook на предпоследнем блоке YOLO, чтобы train/val/test
    всегда видели один и тот же тип признаков.
    """
    def __init__(self, weights_path: str):
        super().__init__()

        detector = YOLO(weights_path)
        self.backbone = detector.model
        self.backbone.eval()

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.feature_map: Optional[torch.Tensor] = None

        if not hasattr(self.backbone, "model") or len(self.backbone.model) < 2:
            raise RuntimeError("Не удалось найти внутренние блоки YOLO для hook.")

        hook_module = self.backbone.model[-2]

        def hook_fn(module, inputs, output):
            candidate = None
            if isinstance(output, torch.Tensor) and output.ndim == 4:
                candidate = output
            elif isinstance(output, (list, tuple)):
                for item in output:
                    if isinstance(item, torch.Tensor) and item.ndim == 4:
                        candidate = item
                        break

            if candidate is None:
                raise RuntimeError("Hook YOLO не получил 4D feature map.")
            self.feature_map = candidate

        self._hook_handle = hook_module.register_forward_hook(hook_fn)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.feature_map = None
        _ = self.backbone(x)

        if self.feature_map is None:
            raise RuntimeError("YOLO backbone hook не вернул feature map.")

        fmap = self.feature_map
        emb = fmap.mean(dim=(2, 3))
        return fmap, emb

    def train(self, mode: bool = True):
        super().train(False)
        self.backbone.eval()
        return self


class CompleteDefectDetector(nn.Module):
    def __init__(self):
        super().__init__()

        self.yolo_backbone = YoloBackboneExtractor(Config.YOLO_PATH)
        self.autoencoder = Autoencoder(latent_dim=Config.AE_EMB_SIZE)

        ae_checkpoint = torch.load(Config.AE_PATH, map_location=Config.DEVICE)
        ae_state = (
            ae_checkpoint["model_state_dict"]
            if isinstance(ae_checkpoint, dict) and "model_state_dict" in ae_checkpoint
            else ae_checkpoint
        )
        self.autoencoder.load_state_dict(ae_state, strict=True)
        self.autoencoder.eval()

        for param in self.autoencoder.parameters():
            param.requires_grad = False

        self.fusion = FusionLayer(
            yolo_dim=Config.YOLO_FEATURE_CHANNELS,
            ae_dim=Config.AE_EMB_SIZE,
            output_dim=Config.FUSION_SIZE,
        )

        self.temporal = TemporalLSTM(
            input_size=Config.FUSION_SIZE,
            hidden_size=Config.LSTM_HIDDEN,
            num_layers=2,
            dropout=0.3,
        )

        self.yolo_head = YOLOHeadWithContext(
            feature_channels=Config.YOLO_FEATURE_CHANNELS,
            context_dim=Config.FUSION_SIZE,
            ae_dim=Config.AE_EMB_SIZE,
            num_classes=4,
        )

        self.classifier = DefectClassifier(
            input_size=Config.FUSION_SIZE,
            num_classes=Config.NUM_CLASSES,
        )

        self.to(Config.DEVICE)
        print(f"Complete model создана на {Config.DEVICE}")
        self._count_parameters()

    def _count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Параметров всего: {total:,}")
        print(f"Обучаемых: {trainable:,}")

    def train(self, mode: bool = True):
        super().train(mode)
        self.yolo_backbone.eval()
        self.autoencoder.eval()
        return self

    def _extract_ae_features(self, frames_ae: torch.Tensor):
        with torch.no_grad():
            ae_recon, ae_latent = self.autoencoder(frames_ae)
            ae_error = torch.mean((ae_recon - frames_ae) ** 2, dim=(1, 2, 3))
        return ae_recon, ae_latent, ae_error

    def forward(self, frame_sequence: torch.Tensor) -> Dict[str, torch.Tensor]:
        if frame_sequence.ndim != 5:
            raise ValueError(f"Ожидается [B, T, 3, H, W], получено: {tuple(frame_sequence.shape)}")

        batch, seq_len, c, h, w = frame_sequence.shape
        if seq_len != Config.SEQ_LENGTH:
            raise ValueError(f"Ожидалась длина последовательности {Config.SEQ_LENGTH}, получено {seq_len}")

        frames = frame_sequence.view(batch * seq_len, c, h, w)

        # YOLO признаки
        features_map_all, yolo_emb_all = self.yolo_backbone(frames)
        if yolo_emb_all.shape[1] != Config.YOLO_FEATURE_CHANNELS:
            raise ValueError(
                f"YOLO embedding имеет размер {yolo_emb_all.shape[1]}, ожидалось {Config.YOLO_FEATURE_CHANNELS}. "
                "Проверь слой hook в YoloBackboneExtractor."
            )

        _, feat_c, feat_h, feat_w = features_map_all.shape
        features_map = features_map_all.view(batch, seq_len, feat_c, feat_h, feat_w)
        yolo_emb = yolo_emb_all.view(batch, seq_len, -1)

        # AE признаки
        frames_ae = F.interpolate(
            frames,
            size=(Config.IMG_SIZE_AE, Config.IMG_SIZE_AE),
            mode="bilinear",
            align_corners=False,
        )
        _, ae_latent_all, ae_error_all = self._extract_ae_features(frames_ae)
        ae_latent = ae_latent_all.view(batch, seq_len, -1)
        ae_errors = ae_error_all.view(batch, seq_len)

        # Fusion по кадрам
        fused_seq = []
        for t in range(seq_len):
            fused_t = self.fusion(
                yolo_feat=yolo_emb[:, t],
                ae_emb=ae_latent[:, t],
                ae_error=ae_errors[:, t],
            )
            fused_seq.append(fused_t)

        fused_seq = torch.stack(fused_seq, dim=1)

        # Temporal
        context, lstm_out, attn_weights = self.temporal(fused_seq)

        # Head для последнего кадра
        last_features_map = features_map[:, -1]
        yolo_detections = self.yolo_head(
            features_map=last_features_map,
            context=context,
            ae_emb=ae_latent[:, -1],
        )

        # Классификация
        log_probs, seq_embedding = self.classifier(context)

        return {
            "log_probs": log_probs,
            "yolo_detections": yolo_detections,
            "context": context,
            "seq_embedding": seq_embedding,
            "ae_errors": ae_errors,
            "attention_weights": attn_weights,
            "lstm_out": lstm_out,
            "fused_seq": fused_seq,
        }

    @torch.no_grad()
    def infer_frame(self, frame_t: torch.Tensor, history_buffer: List[torch.Tensor]):
        if frame_t.ndim != 4 or frame_t.shape[0] != 1:
            raise ValueError(f"infer_frame ожидает [1, 3, H, W], получено {tuple(frame_t.shape)}")

        features_map, yolo_emb = self.yolo_backbone(frame_t)

        frame_ae = F.interpolate(
            frame_t,
            size=(Config.IMG_SIZE_AE, Config.IMG_SIZE_AE),
            mode="bilinear",
            align_corners=False,
        )
        _, ae_latent, ae_error = self._extract_ae_features(frame_ae)

        fused = self.fusion(
            yolo_feat=yolo_emb,
            ae_emb=ae_latent,
            ae_error=ae_error,
        )

        history_buffer.append(fused.squeeze(0))
        if len(history_buffer) > Config.SEQ_LENGTH:
            history_buffer.pop(0)

        defect_prob = 0.0
        pred_class = 4
        class_probs = torch.zeros(Config.NUM_CLASSES, device=frame_t.device)
        class_probs[4] = 1.0
        attention_weights = None

        if len(history_buffer) >= 3:
            history = torch.stack(history_buffer, dim=0).unsqueeze(0)
            context, _, attention_weights = self.temporal(history)
            log_probs, _ = self.classifier(context)
            class_probs = torch.exp(log_probs)[0]
            pred_class = int(torch.argmax(class_probs).item())
            defect_prob = float(1.0 - class_probs[4].item())

        return {
            "probability": defect_prob,
            "predicted_class": pred_class,
            "class_name": Config.CLASS_NAMES[pred_class],
            "ae_error": float(ae_error.item()),
            "history_size": len(history_buffer),
            "class_probabilities": class_probs.detach().cpu().tolist(),
            "attention_weights": (
                attention_weights.detach().cpu().tolist()[0]
                if attention_weights is not None else None
            ),
            "yolo_feature_shape": tuple(features_map.shape),
            "yolo_embedding_shape": tuple(yolo_emb.shape),
        }