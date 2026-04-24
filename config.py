import os
from pathlib import Path
import torch


class Config:
    BASE_DIR = Path(__file__).resolve().parent
    WORKSPACE = Path(os.getenv("WORKSPACE_DIR", BASE_DIR / "workspace"))
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    YOLO_PATH = str(WORKSPACE / "runs" / "detect" / "train" / "weights" / "best.pt")
    AE_PATH = str(WORKSPACE / "autoencoder_model.pth")
    COMPLETE_MODEL_PATH = str(WORKSPACE / "best_complete_model_fixed.pth")

    # Data
    DATA_ROOT = str(WORKSPACE / "data")
    YOLO_DATA_DIR = Path(DATA_ROOT) / "yolo_train"
    IMG_SIZE_YOLO = 640
    IMG_SIZE_AE = 224
    SEQ_LENGTH = 16

    # Dimensions
    YOLO_FEATURE_CHANNELS = 256
    AE_EMB_SIZE = 256
    FUSION_SIZE = 512
    LSTM_HIDDEN = 256
    NUM_CLASSES = 5

    # Classes
    CLASS_NAMES = ["connection", "foreign object", "garbage", "point", "normal"]
    YOLO_CLASS_NAMES = ["connection", "foreign object", "garbage", "point"]

    # Training
    BATCH_SIZE = 4
    EPOCHS = 50
    LR = 3e-4
    WEIGHT_DECAY = 1e-5
    GRAD_CLIP = 1.0
    CLASS_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 0.3]

    # Inference
    DEFECT_THRESHOLD = 0.5
