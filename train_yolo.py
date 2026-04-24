from pathlib import Path

import torch
from ultralytics import YOLO

from config import Config


def ensure_data_yaml():
    yolo_dir = Path(Config.DATA_ROOT) / "yolo_train"
    yolo_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = yolo_dir / "data.yaml"
    names = Config.YOLO_CLASS_NAMES
    content = "".join([
        f"path: {yolo_dir.resolve()}\n",
        "train: images\n",
        "val: images\n",
        f"names: {names}\n",
        f"nc: {len(names)}\n",
    ])
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


def main():
    yaml_path = ensure_data_yaml()
    model = YOLO("yolov8n.pt")
    model.train(
        data=str(yaml_path),
        epochs=100,
        imgsz=512,
        batch=16,
        device=0 if torch.cuda.is_available() else "cpu",
        project=str(Path(Config.WORKSPACE) / "runs" / "detect"),
        name="train",
        exist_ok=True,
        optimizer="Adam",
        lr0=0.001,
        augment=True,
        mosaic=1.0,
        save=True,
        plots=True,
    )
    print(f"Model saved: {Config.YOLO_PATH}")


if __name__ == "__main__":
    main()
