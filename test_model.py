import argparse
import json
import os
from pathlib import Path
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import YOLO

from config import Config
from complete_model import CompleteDefectDetector
from train_autoencoder import Autoencoder


def load_torch_checkpoint(path, device):
    return torch.load(path, map_location=device)


def test_yolo(model_path, image_path):
    if not os.path.exists(model_path):
        print(f"YOLO модель не найдена: {model_path}")
        return
    if not os.path.exists(image_path):
        print(f"Изображение не найдено: {image_path}")
        return

    print(f"\nТестирование YOLO на {image_path}")
    model = YOLO(model_path)
    results = model(image_path, verbose=False)[0]
    img = cv2.imread(image_path)

    if results.boxes is not None:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = results.names.get(
                cls,
                Config.CLASS_NAMES[cls] if cls < len(Config.CLASS_NAMES) else str(cls)
            )
            label = f"{class_name} {conf:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img,
                label,
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    out_path = (
        image_path.replace(".jpg", "_yolo.jpg")
        .replace(".png", "_yolo.png")
        .replace(".jpeg", "_yolo.jpeg")
    )
    cv2.imwrite(out_path, img)
    print(f"Результат сохранён в {out_path}")


def test_autoencoder(model_path, image_path):
    if not os.path.exists(model_path):
        print(f"Autoencoder модель не найдена: {model_path}")
        return
    if not os.path.exists(image_path):
        print(f"Изображение не найдено: {image_path}")
        return

    print(f"\nТестирование Autoencoder на {image_path}")
    device = Config.DEVICE

    model = Autoencoder(latent_dim=Config.AE_EMB_SIZE)
    checkpoint = load_torch_checkpoint(model_path, device)
    state = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state, strict=True)
    model.eval().to(device)

    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (Config.IMG_SIZE_AE, Config.IMG_SIZE_AE))
    img_tensor = torch.from_numpy(
        img_resized.astype(np.float32) / 255.0
    ).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        recon, latent = model(img_tensor)
        error = torch.mean((recon - img_tensor) ** 2).item()

    recon_img = recon[0].cpu().permute(1, 2, 0).numpy()
    diff = np.abs(img_tensor[0].cpu().numpy() - recon[0].cpu().numpy()).mean(axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_rgb)
    axes[0].set_title("Оригинал")
    axes[0].axis("off")

    axes[1].imshow(recon_img)
    axes[1].set_title(f"Реконструкция\nMSE = {error:.6f}")
    axes[1].axis("off")

    axes[2].imshow(diff, cmap="hot")
    axes[2].set_title("Карта ошибки")
    axes[2].axis("off")

    out_path = (
        image_path.replace(".jpg", "_ae.png")
        .replace(".png", "_ae.png")
        .replace(".jpeg", "_ae.png")
    )
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Результат сохранён в {out_path}")
    plt.show()

def natural_key(path):
    s = str(path.name)
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def _load_sequence_from_folder(input_path: str):
    frames_paths = (
        sorted(Path(input_path).glob("*.jpg")) +
        sorted(Path(input_path).glob("*.jpeg")) +
        sorted(Path(input_path).glob("*.png"))
    )
    frames_paths = sorted(frames_paths, key=natural_key)
    print(f"Найдено кадров: {len(frames_paths)}")

    if len(frames_paths) == 0:
        raise ValueError("В папке нет изображений.")
    if len(frames_paths) < Config.SEQ_LENGTH:
        raise ValueError(
            f"Найдено только {len(frames_paths)} кадров. Нужно минимум {Config.SEQ_LENGTH}."
        )

    selected = frames_paths[: Config.SEQ_LENGTH]
    frames = []

    for p in selected:
        img = cv2.imread(str(p))
        if img is None:
            raise ValueError(f"Не удалось прочитать {p}")

        img = cv2.resize(img, (Config.IMG_SIZE_YOLO, Config.IMG_SIZE_YOLO))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        frames.append(torch.from_numpy(img).permute(2, 0, 1))

    seq = torch.stack(frames, dim=0).unsqueeze(0)
    return selected, seq


def test_lstm(model_path, input_path, save_json=True):
    if not os.path.exists(model_path):
        print(f"Полная модель не найдена: {model_path}")
        return
    if not os.path.exists(input_path):
        print(f"Входные данные не найдены: {input_path}")
        return

    print(f"\nТестирование полной модели на {input_path}")
    device = Config.DEVICE
    model = CompleteDefectDetector()

    checkpoint = load_torch_checkpoint(model_path, device)
    state = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Загрузка checkpoint: missing={len(missing)}, unexpected={len(unexpected)}")
    model.eval().to(device)

    if os.path.isdir(input_path):
        frame_paths, seq = _load_sequence_from_folder(input_path)
        seq = seq.to(device)

        with torch.no_grad():
            output = model(seq)

        # probs = torch.exp(output["log_probs"])[0].cpu().numpy()
        # pred_idx = int(np.argmax(probs))
        # defect_prob = float(1.0 - probs[4])
        probs = torch.exp(output["log_probs"])[0].cpu().numpy()
        normal_prob = float(probs[4])
        defect_prob = float(1.0 - normal_prob)

        if defect_prob > Config.DEFECT_THRESHOLD:
            pred_idx = int(np.argmax(probs[:4]))   # только defect classes
        else:
            pred_idx = 4        
        print("\nРезультат для последовательности:")
        print(f"Предсказанный класс: {Config.CLASS_NAMES[pred_idx]}")
        print(f"Вероятность дефекта: {defect_prob:.4f}")
        print("Вероятности классов:")
        for name, p in zip(Config.CLASS_NAMES, probs):
            print(f"      {name}: {p:.4f}")

        ae_errors = output["ae_errors"][0].detach().cpu().numpy().tolist()
        attention = output["attention_weights"][0].detach().cpu().numpy().tolist()

        print("\n   AE errors по кадрам:")
        for i, val in enumerate(ae_errors, 1):
            print(f"      frame {i:02d}: {val:.8f}")

        print("\n   Attention по кадрам:")
        for i, val in enumerate(attention, 1):
            print(f"      frame {i:02d}: {val:.6f}")

        if save_json:
            report = {
                "input_path": input_path,
                "predicted_class_idx": pred_idx,
                "predicted_class_name": Config.CLASS_NAMES[pred_idx],
                "defect_probability": defect_prob,
                "class_probabilities": {
                    name: float(p) for name, p in zip(Config.CLASS_NAMES, probs)
                },
                "ae_errors": [float(v) for v in ae_errors],
                "attention_weights": [float(v) for v in attention],
                "frames": [str(p) for p in frame_paths],
            }
            out_path = Path(input_path) / "sequence_report_fixed.json"
            out_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            print(f"\nJSON отчёт сохранён в {out_path}")
        return

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Не удалось открыть видео")
        return

    fps = max(1, int(cap.get(cv2.CAP_PROP_FPS)))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = (
        input_path.replace(".mp4", "_lstm_fixed.mp4")
        .replace(".avi", "_lstm_fixed.mp4")
    )
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    history_buffer = []
    probabilities = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_resized = cv2.resize(frame, (Config.IMG_SIZE_YOLO, Config.IMG_SIZE_YOLO))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(
            frame_rgb.astype(np.float32) / 255.0
        ).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            result = model.infer_frame(frame_tensor, history_buffer)

        prob = result["probability"]
        probabilities.append(prob)

        color = (0, 0, 255) if prob > Config.DEFECT_THRESHOLD else (0, 255, 0)
        status = f"DEFECT ({result['class_name']})" if prob > Config.DEFECT_THRESHOLD else "NORMAL"

        cv2.putText(frame, f"{status} {prob:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"AE: {result['ae_error']:.5f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(
            frame,
            f"History: {result['history_size']}/{Config.SEQ_LENGTH}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )
        out.write(frame)

    cap.release()
    out.release()

    print(f"Обработано {frame_count} кадров. Видео сохранено в {out_path}")

    plt.figure(figsize=(10, 4))
    plt.plot(probabilities)
    plt.axhline(y=Config.DEFECT_THRESHOLD, linestyle="--")
    plt.xlabel("Кадр")
    plt.ylabel("Вероятность дефекта")
    plt.title("Вероятность дефекта по кадрам")
    plt.grid(True)
    plot_path = (
        input_path.replace(".mp4", "_prob_fixed.png")
        .replace(".avi", "_prob_fixed.png")
    )
    plt.savefig(plot_path)
    print(f"График сохранён в {plot_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["yolo", "ae", "lstm", "all"], required=True)
    parser.add_argument("--input", help="Путь к изображению, видео или папке")
    parser.add_argument("--yolo_model", default=Config.YOLO_PATH)
    parser.add_argument("--ae_model", default=Config.AE_PATH)
    parser.add_argument("--lstm_model", default=Config.COMPLETE_MODEL_PATH)
    args = parser.parse_args()

    if args.model == "yolo":
        if not args.input:
            print("Для YOLO укажите --input <изображение>")
            return
        test_yolo(args.yolo_model, args.input)

    elif args.model == "ae":
        if not args.input:
            print("Для Autoencoder укажите --input <изображение>")
            return
        test_autoencoder(args.ae_model, args.input)

    elif args.model == "lstm":
        if not args.input:
            print("Для полной модели укажите --input <видео или папка>")
            return
        test_lstm(args.lstm_model, args.input)

    elif args.model == "all":
        if not args.input:
            print("Для all укажите --input <изображение>")
            return
        test_yolo(args.yolo_model, args.input)
        test_autoencoder(args.ae_model, args.input)
        print("\nДля полной temporal-модели запускай отдельно:")
        print("python test_model_fixed.py --model lstm --input <папка_или_видео>")


if __name__ == "__main__":
    main()