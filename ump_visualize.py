import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from torch.utils.data import DataLoader

from config import Config
from complete_model import CompleteDefectDetector
from dataset import VideoSequenceDataset


def main():
    device = Config.DEVICE

    model = CompleteDefectDetector().to(device)
    checkpoint = torch.load(Config.COMPLETE_MODEL_PATH, map_location=device)
    state = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state, strict=False)
    model.eval()

    dataset = VideoSequenceDataset(root_dir=Config.DATA_ROOT, seq_length=Config.SEQ_LENGTH, augment=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    embeddings = []
    labels = []

    with torch.no_grad():
        for seq, label in loader:
            seq = seq.to(device)
            output = model(seq)
            vec = output["context"].cpu().numpy()[0]   # [dim]
            embeddings.append(vec)
            labels.append(int(label.item()))

    embeddings = np.array(embeddings)
    labels = np.array(labels)

    reducer = umap.UMAP(n_components=2, random_state=42)
    emb_2d = reducer.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap="tab10")
    plt.title("UMAP of sequence embeddings")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.colorbar(scatter)
    plt.grid(True)
    plt.savefig("umap_sequences.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()