import random
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset

from config import Config


def _list_images(folder: Path):
    return (
        sorted(folder.glob("*.jpg")) +
        sorted(folder.glob("*.jpeg")) +
        sorted(folder.glob("*.png"))
    )


class VideoSequenceDataset(Dataset):
    """
    Формирует последовательности длиной seq_length.

    Ожидаемая структура:
      data/
        defect_connection/images/*.jpg
        defect_foreign/images/*.jpg
        defect_garbage/images/*.jpg
        defect_point/images/*.jpg
        normal/*.jpg
    """

    def __init__(
        self,
        root_dir: str = Config.DATA_ROOT,
        seq_length: int = Config.SEQ_LENGTH,
        augment: bool = False
    ):
        self.root_dir = Path(root_dir)
        self.seq_length = seq_length
        self.augment = augment

        self.folder_to_class = {
            "defect_connection": 0,
            "defect_foreign": 1,
            "defect_garbage": 2,
            "defect_point": 3,
            "normal": 4,
        }

        self.samples = []
        self._build_index()

    def _add_sequences_from_images(self, image_paths, label, step):
        if len(image_paths) < self.seq_length:
            return

        for start in range(0, len(image_paths) - self.seq_length + 1, step):
            window = image_paths[start:start + self.seq_length]
            self.samples.append({"paths": window, "label": label})

    def _build_index(self):
        for folder_name, class_id in self.folder_to_class.items():
            if class_id == 4:
                folder_path = self.root_dir / "normal"
                images = _list_images(folder_path)
                before = len(self.samples)
                self._add_sequences_from_images(images, class_id, step=self.seq_length)
                created = len(self.samples) - before
                print(f"normal: {len(images)} изображений -> {created} seq")
                continue

            folder_path = self.root_dir / folder_name / "images"
            if not folder_path.exists():
                print(f"Папка {folder_path} не найдена")
                continue

            images = _list_images(folder_path)
            before = len(self.samples)
            self._add_sequences_from_images(images, class_id, step=max(1, self.seq_length // 2))
            created = len(self.samples) - before
            print(f"{folder_name}: {len(images)} изображений -> {created} seq")

        print(f"\nВсего последовательностей: {len(self.samples)}")
        for class_id, class_name in enumerate(Config.CLASS_NAMES):
            count = sum(1 for s in self.samples if s["label"] == class_id)
            print(f"  класс {class_id} ({class_name}): {count}")

    def __len__(self):
        return len(self.samples)

    def _augment_frame(self, img_rgb: np.ndarray) -> np.ndarray:
        if not self.augment:
            return img_rgb

        if random.random() < 0.5:
            img_rgb = np.ascontiguousarray(np.fliplr(img_rgb))

        if random.random() < 0.2:
            noise = np.random.normal(0, 3, img_rgb.shape).astype(np.float32)
            img_rgb = np.clip(img_rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        if random.random() < 0.3:
            alpha = random.uniform(0.9, 1.1)
            beta = random.uniform(-8, 8)
            img_rgb = np.clip(alpha * img_rgb + beta, 0, 255).astype(np.uint8)

        return img_rgb

    def __getitem__(self, idx):
        sample = self.samples[idx]
        frames = []

        for path in sample["paths"]:
            img = cv2.imread(str(path))
            if img is None:
                raise ValueError(f"Не удалось прочитать изображение: {path}")

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (Config.IMG_SIZE_YOLO, Config.IMG_SIZE_YOLO))
            img = self._augment_frame(img)
            img = img.astype(np.float32) / 255.0
            img = torch.from_numpy(img).permute(2, 0, 1)
            frames.append(img)

        sequence = torch.stack(frames, dim=0)
        label = torch.tensor(sample["label"], dtype=torch.long)
        return sequence, label


def create_dataloaders(
    batch_size: int = Config.BATCH_SIZE,
    root_dir: str = Config.DATA_ROOT,
    seq_length: int = Config.SEQ_LENGTH,
    train_ratio: float = 0.8,
    seed: int = 42,
):
    base_dataset = VideoSequenceDataset(root_dir=root_dir, seq_length=seq_length, augment=False)

    indices_by_class = defaultdict(list)
    for idx, sample in enumerate(base_dataset.samples):
        indices_by_class[sample["label"]].append(idx)

    rng = random.Random(seed)
    train_indices, val_indices = [], []

    for class_id, class_indices in indices_by_class.items():
        rng.shuffle(class_indices)
        split = max(1, int(len(class_indices) * train_ratio)) if len(class_indices) > 1 else len(class_indices)
        train_indices.extend(class_indices[:split])
        val_indices.extend(class_indices[split:])

    train_dataset = Subset(
        VideoSequenceDataset(root_dir=root_dir, seq_length=seq_length, augment=True),
        train_indices
    )
    val_dataset = Subset(
        VideoSequenceDataset(root_dir=root_dir, seq_length=seq_length, augment=False),
        val_indices
    )

    train_labels = [base_dataset.samples[idx]["label"] for idx in train_indices]
    class_counts = [max(1, train_labels.count(i)) for i in range(Config.NUM_CLASSES)]
    sample_weights = [1.0 / class_counts[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    return train_loader, val_loader