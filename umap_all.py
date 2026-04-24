import argparse
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import umap
from matplotlib.lines import Line2D
from torch.utils.data import Dataset, DataLoader

from config import Config
from complete_model import CompleteDefectDetector


def natural_key(path: Path):
    s = path.name
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_images(folder: Path):
    files = (
        list(folder.glob("*.jpg")) +
        list(folder.glob("*.jpeg")) +
        list(folder.glob("*.png")) +
        list(folder.glob("*.bmp"))
    )
    return sorted(files, key=natural_key)


class UMAPSequenceDataset(Dataset):
    """
    Отдельный датасет только для анализа/Umap.
    """
    def __init__(
        self,
        root_dir: str = Config.DATA_ROOT,
        seq_length: int = Config.SEQ_LENGTH,
        defect_step: int = 1,
        normal_step: int = 16,
    ):
        self.root_dir = Path(root_dir)
        self.seq_length = seq_length
        self.defect_step = defect_step
        self.normal_step = normal_step

        self.folder_to_class = {
            "defect_connection": 0,
            "defect_foreign": 1,
            "defect_garbage": 2,
            "defect_point": 3,
            "normal": 4,
        }

        self.samples = []
        self._build_index()

    def _add_windows(self, image_paths, label, step):
        if len(image_paths) < self.seq_length:
            return

        for start in range(0, len(image_paths) - self.seq_length + 1, step):
            window = image_paths[start:start + self.seq_length]
            self.samples.append({"paths": window, "label": label})

    def _build_index(self):
        for folder_name, class_id in self.folder_to_class.items():
            if class_id == 4:
                folder = self.root_dir / "normal"
                images = list_images(folder)
                before = len(self.samples)
                self._add_windows(images, class_id, self.normal_step)
                print(f"normal: {len(images)} изображений -> {len(self.samples) - before} seq (UMAP)")
            else:
                folder = self.root_dir / folder_name / "images"
                if not folder.exists():
                    print(f"Папка {folder} не найдена")
                    continue
                images = list_images(folder)
                before = len(self.samples)
                self._add_windows(images, class_id, self.defect_step)
                print(f"{folder_name}: {len(images)} изображений -> {len(self.samples) - before} seq (UMAP)")

        print(f"\nВсего последовательностей для UMAP: {len(self.samples)}")
        for class_id, class_name in enumerate(Config.CLASS_NAMES):
            count = sum(1 for s in self.samples if s["label"] == class_id)
            print(f"  класс {class_id} ({class_name}): {count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        frames = []

        for path in sample["paths"]:
            import cv2
            img = cv2.imread(str(path))
            if img is None:
                raise ValueError(f"Не удалось прочитать изображение: {path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (Config.IMG_SIZE_YOLO, Config.IMG_SIZE_YOLO))
            img = img.astype(np.float32) / 255.0
            frames.append(torch.from_numpy(img).permute(2, 0, 1))

        sequence = torch.stack(frames, dim=0)
        label = torch.tensor(sample["label"], dtype=torch.long)
        return sequence, label


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_umap_plot(emb_2d, labels, class_names, title, out_path):
    plt.figure(figsize=(9, 7))
    plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap="tab10", s=30, alpha=0.85)
    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.grid(True, alpha=0.3)

    unique_labels = sorted(set(labels.tolist()))
    handles = []
    cmap = plt.get_cmap("tab10")
    for lab in unique_labels:
        handles.append(
            Line2D(
                [0], [0],
                marker="o",
                color="w",
                markerfacecolor=cmap(lab % 10),
                markersize=8,
                label=f"{lab}: {class_names[lab]}"
            )
        )

    plt.legend(handles=handles, loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def reduce_umap(features, n_neighbors=10, min_dist=0.15, random_state=42):
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    return reducer.fit_transform(features)


@torch.no_grad()
def extract_sequence_features(model, loader, device):
    temporal_contexts = []
    labels_seq = []

    fusion_frame_features = []
    fusion_frame_labels = []

    for seq, label in loader:
        seq = seq.to(device)
        output = model(seq)

        context = output["context"].detach().cpu().numpy()
        fused_seq = output["fused_seq"].detach().cpu().numpy()

        temporal_contexts.append(context)
        labels_seq.append(label.numpy())

        b, t, d = fused_seq.shape
        fused_seq = fused_seq.reshape(b * t, d)
        fusion_frame_features.append(fused_seq)
        fusion_frame_labels.append(np.repeat(label.numpy(), t))

    temporal_contexts = np.concatenate(temporal_contexts, axis=0)
    labels_seq = np.concatenate(labels_seq, axis=0)
    fusion_frame_features = np.concatenate(fusion_frame_features, axis=0)
    fusion_frame_labels = np.concatenate(fusion_frame_labels, axis=0)

    return temporal_contexts, labels_seq, fusion_frame_features, fusion_frame_labels


@torch.no_grad()
def extract_yolo_ae_features(model, dataset, device):
    yolo_features = []
    ae_features = []
    frame_labels = []

    for seq, label in dataset:
        seq = seq.unsqueeze(0).to(device)
        b, seq_len, c, h, w = seq.shape
        frames = seq.view(b * seq_len, c, h, w)

        _, yolo_emb = model.yolo_backbone(frames)
        yolo_emb = yolo_emb.detach().cpu().numpy()

        frames_ae = F.interpolate(
            frames,
            size=(Config.IMG_SIZE_AE, Config.IMG_SIZE_AE),
            mode="bilinear",
            align_corners=False
        )
        _, ae_latent, _ = model._extract_ae_features(frames_ae)
        ae_latent = ae_latent.detach().cpu().numpy()

        yolo_features.append(yolo_emb)
        ae_features.append(ae_latent)
        frame_labels.append(np.repeat(int(label.item()), seq_len))

    yolo_features = np.concatenate(yolo_features, axis=0)
    ae_features = np.concatenate(ae_features, axis=0)
    frame_labels = np.concatenate(frame_labels, axis=0)

    return yolo_features, ae_features, frame_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="umap_results")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--n_neighbors", type=int, default=10)
    parser.add_argument("--min_dist", type=float, default=0.15)
    parser.add_argument("--defect_step", type=int, default=1, help="Шаг окна для defect в UMAP")
    parser.add_argument("--normal_step", type=int, default=16, help="Шаг окна для normal в UMAP")
    args = parser.parse_args()

    ensure_dir(args.outdir)
    device = Config.DEVICE

    print("Загрузка модели...")
    model = CompleteDefectDetector().to(device)
    checkpoint = torch.load(Config.COMPLETE_MODEL_PATH, map_location=device)
    state = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Checkpoint loaded: missing={len(missing)}, unexpected={len(unexpected)}")
    model.eval()

    print("Загрузка датасета для UMAP...")
    dataset = UMAPSequenceDataset(
        root_dir=Config.DATA_ROOT,
        seq_length=Config.SEQ_LENGTH,
        defect_step=args.defect_step,
        normal_step=args.normal_step,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print("Извлечение temporal/fusion признаков...")
    temporal_contexts, labels_seq, fusion_feats, fusion_labels = extract_sequence_features(model, loader, device)

    print("Извлечение YOLO/AE признаков...")
    yolo_feats, ae_feats, frame_labels = extract_yolo_ae_features(model, dataset, device)

    print("UMAP: temporal context...")
    temporal_2d = reduce_umap(temporal_contexts, n_neighbors=args.n_neighbors, min_dist=args.min_dist)
    save_umap_plot(
        temporal_2d,
        labels_seq,
        Config.CLASS_NAMES,
        "UMAP: Temporal context (sequence-level)",
        str(Path(args.outdir) / "umap_temporal_context.png")
    )

    print("UMAP: fusion features...")
    fusion_2d = reduce_umap(fusion_feats, n_neighbors=args.n_neighbors, min_dist=args.min_dist)
    save_umap_plot(
        fusion_2d,
        fusion_labels,
        Config.CLASS_NAMES,
        "UMAP: Fusion features (frame-level)",
        str(Path(args.outdir) / "umap_fusion_features.png")
    )

    print("UMAP: YOLO features...")
    yolo_2d = reduce_umap(yolo_feats, n_neighbors=args.n_neighbors, min_dist=args.min_dist)
    save_umap_plot(
        yolo_2d,
        frame_labels,
        Config.CLASS_NAMES,
        "UMAP: YOLO embeddings (frame-level)",
        str(Path(args.outdir) / "umap_yolo_embeddings.png")
    )

    print("UMAP: AE latent...")
    ae_2d = reduce_umap(ae_feats, n_neighbors=args.n_neighbors, min_dist=args.min_dist)
    save_umap_plot(
        ae_2d,
        frame_labels,
        Config.CLASS_NAMES,
        "UMAP: AE latent (frame-level)",
        str(Path(args.outdir) / "umap_ae_latent.png")
    )

    print(f"Картинки сохранены в: {args.outdir}")


if __name__ == "__main__":
    main()