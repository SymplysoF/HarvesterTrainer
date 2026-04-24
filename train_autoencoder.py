import os
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import Config


class NormalDataset(Dataset):
    def __init__(self, normal_dir=None, img_size=224):
        self.normal_dir = Path(normal_dir or (Path(Config.DATA_ROOT) / "normal"))
        self.images = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            self.images.extend(self.normal_dir.glob(ext))
        self.images = sorted(self.images)
        self.img_size = img_size
        if not self.images:
            raise FileNotFoundError(f"No normal images found in {self.normal_dir}")
        print(f"Loaded normal images: {len(self.images)} from {self.normal_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img, img


class Autoencoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 14 * 14 * 128),
            nn.ReLU(),
            nn.Unflatten(1, (128, 14, 14)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

    def get_embedding(self, x):
        return self.encoder(x)


def main():
    print("AUTOENCODER TRAINING")
    device = Config.DEVICE
    print(f"Using device: {device}")

    dataset = NormalDataset(img_size=Config.IMG_SIZE_AE)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

    model = Autoencoder(latent_dim=Config.AE_EMB_SIZE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    history = {"loss": []}

    for epoch in range(100):
        model.train()
        total_loss = 0.0
        for inputs, targets in tqdm(dataloader, desc=f"AE epoch {epoch + 1}"):
            inputs, targets = inputs.to(device), targets.to(device)
            reconstructed, _ = model(inputs)
            loss = criterion(reconstructed, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / max(1, len(dataloader))
        history["loss"].append(avg_loss)
        print(f"Epoch {epoch + 1:03d}: loss={avg_loss:.6f}")

    out_path = Path(Config.AE_PATH)
    torch.save({"model_state_dict": model.state_dict(), "latent_dim": Config.AE_EMB_SIZE, "history": history}, out_path)
    plt.figure(figsize=(10, 5))
    plt.plot(history["loss"])
    plt.title("Autoencoder Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(Path(Config.WORKSPACE) / "autoencoder_loss.png")
    print(f"Saved model: {out_path}")


if __name__ == "__main__":
    main()
