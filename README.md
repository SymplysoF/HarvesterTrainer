# Harvester Trainer Web Project

This is a finalized working project built around your existing neural network pipeline:

1. **YOLO training**
2. **Autoencoder training**
3. **Complete temporal model training** (Fusion + LSTM + Classifier)

It adds a minimal modern web interface where the user can:
- upload images,
- split them into the required folders,
- create YOLO labels,
- launch training from the browser,
- monitor logs.

## Project structure

```text
final_project/
├─ server.py
├─ config.py
├─ train_yolo.py
├─ train_autoencoder.py
├─ train_complete.py
├─ complete_model.py
├─ dataset.py
├─ fusion_layer.py
├─ temporal_model.py
├─ classifier.py
├─ yolo_head.py
├─ pipeline_runner.py
├─ test_model.py
├─ templates/
├─ static/
└─ workspace/
   ├─ data/
   │  ├─ yolo_train/
   │  │  ├─ images/
   │  │  ├─ labels/
   │  │  └─ data.yaml
   │  ├─ normal/
   │  ├─ defect_connection/images/
   │  ├─ defect_foreign/images/
   │  ├─ defect_garbage/images/
   │  └─ defect_point/images/
   └─ logs/
```

## Installation

Create and activate a virtual environment:

### Windows
```bash
python -m venv .venv
.venv\Scriptsctivate
```

### Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the web interface

```bash
uvicorn server:app --reload
```

Open in browser:

```text
http://127.0.0.1:8000
```

## How to use

### 1) Upload YOLO images
- Open **Datasets**.
- Upload defect images into the **YOLO dataset** block.
- They will be saved to:
  - `workspace/data/yolo_train/images/`

### 2) Create labels
- Open **Label**.
- Choose an image.
- Draw boxes on the canvas.
- Select a class:
  - `connection`
  - `foreign object`
  - `garbage`
  - `point`
- Click **Save labels**.
- Label files are saved to:
  - `workspace/data/yolo_train/labels/*.txt`

### 3) Upload normal images for the autoencoder
- In **Datasets**, upload normal images to the **Autoencoder** block.
- They will be saved to:
  - `workspace/data/normal/`

### 4) Upload temporal model images
- In **Datasets**, choose the correct class folder and upload image sequences.
- Folders used by the complete model:
  - `workspace/data/defect_connection/images/`
  - `workspace/data/defect_foreign/images/`
  - `workspace/data/defect_garbage/images/`
  - `workspace/data/defect_point/images/`
  - `workspace/data/normal/`

### 5) Start training
Open **Training** and run stages in this order:

#### Stage 1 — YOLO
```bash
python train_yolo.py
```
Output weights:
```text
workspace/runs/detect/train/weights/best.pt
```

#### Stage 2 — Autoencoder
```bash
python train_autoencoder.py
```
Output weights:
```text
workspace/autoencoder_model.pth
```

#### Stage 3 — Complete temporal model
```bash
python train_complete.py
```
Output weights:
```text
workspace/best_complete_model_fixed.pth
```

## Important fixes included

- YOLO output path is standardized to `workspace/runs/detect/train/weights/best.pt`.
- Autoencoder training now reads normal images from `workspace/data/normal/`.
- `data.yaml` is generated automatically for YOLO.
- The browser interface writes YOLO `.txt` labels in normalized format.
- Training jobs are launched from the web app and logs are stored in `workspace/logs/`.

## How to test after training

### Test the full pipeline on a folder of frames
```bash
python pipeline_runner.py --input workspace/data/defect_connection/images --output workspace/pipeline_output
```

### Test separate modules
```bash
python test_model.py
```

## Notes

- The complete model expects that YOLO and autoencoder weights already exist.
- For the temporal model, each class should contain enough images to build sequences of length `16`.
- If you want faster experiments, reduce epochs in `train_yolo.py`, `train_autoencoder.py`, and `config.py`.
