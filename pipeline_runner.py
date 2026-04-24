import argparse
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from config import Config
from complete_model import CompleteDefectDetector
from train_autoencoder import Autoencoder


def natural_key(path: Path):
    s = path.name
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def list_frames(folder: str):
    folder = Path(folder)
    frames = (
        list(folder.glob("*.jpg")) +
        list(folder.glob("*.jpeg")) +
        list(folder.glob("*.png")) +
        list(folder.glob("*.bmp"))
    )
    return sorted(frames, key=natural_key)


def load_torch_checkpoint(path, device):
    return torch.load(path, map_location=device)


def read_image_rgb(path: Path, size: int):
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Не удалось прочитать {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    return img


def image_to_tensor(img_rgb: np.ndarray):
    return torch.from_numpy(img_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)


def cuda_sync_if_needed():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class PipelineRunner:
    def __init__(self, yolo_path, ae_path, temporal_path, device=None):
        self.device = device or Config.DEVICE

        self.yolo = YOLO(yolo_path)

        self.ae = Autoencoder(latent_dim=Config.AE_EMB_SIZE)
        ae_checkpoint = load_torch_checkpoint(ae_path, self.device)
        ae_state = (
            ae_checkpoint["model_state_dict"]
            if isinstance(ae_checkpoint, dict) and "model_state_dict" in ae_checkpoint
            else ae_checkpoint
        )
        self.ae.load_state_dict(ae_state, strict=True)
        self.ae.eval().to(self.device)

        self.temporal_model = CompleteDefectDetector()
        checkpoint = load_torch_checkpoint(temporal_path, self.device)
        state = (
            checkpoint["model_state_dict"]
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
            else checkpoint
        )
        missing, unexpected = self.temporal_model.load_state_dict(state, strict=False)
        print(f"Temporal checkpoint loaded: missing={len(missing)}, unexpected={len(unexpected)}")
        self.temporal_model.eval().to(self.device)

    @torch.no_grad()
    def run_yolo_on_frame(self, image_path: Path, save_path: Path):
        cuda_sync_if_needed()
        t0 = time.perf_counter()

        results = self.yolo(str(image_path), verbose=False)[0]

        cuda_sync_if_needed()
        yolo_time_ms = (time.perf_counter() - t0) * 1000.0

        img = cv2.imread(str(image_path))
        detections = []

        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                class_name = results.names.get(
                    cls,
                    Config.CLASS_NAMES[cls] if cls < len(Config.CLASS_NAMES) else str(cls)
                )
                confidence_percent = round(conf * 100.0, 2)

                detections.append({
                    "class_id": cls,
                    "class_name": class_name,
                    "confidence": conf,
                    "confidence_percent": confidence_percent,
                    "bbox_xyxy": [x1, y1, x2, y2],
                })

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label = f"{class_name} {confidence_percent:.1f}%"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 7
                thickness = 3

                (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

                text_x = max(x1 + 4, 0)
                text_y = max(y1 + text_h + 6, text_h + 2)

                bg_x1 = text_x - 2
                bg_y1 = text_y - text_h - 4
                bg_x2 = min(text_x + text_w + 2, img.shape[1] - 1)
                bg_y2 = min(text_y + baseline + 2, img.shape[0] - 1)

                cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 255, 0), -1)
                cv2.putText(
                    img,
                    label,
                    (text_x, text_y),
                    font,
                    font_scale,
                    (0, 0, 0),
                    thickness,
                    lineType=cv2.LINE_AA
                )

        cv2.imwrite(str(save_path), img)
        return detections, float(yolo_time_ms)

    @torch.no_grad()
    def run_ae_on_frame(self, image_path: Path):
        img_rgb = read_image_rgb(image_path, Config.IMG_SIZE_AE)
        img_tensor = image_to_tensor(img_rgb).unsqueeze(0).to(self.device)

        cuda_sync_if_needed()
        t0 = time.perf_counter()

        recon, latent = self.ae(img_tensor)

        cuda_sync_if_needed()
        ae_time_ms = (time.perf_counter() - t0) * 1000.0

        error = torch.mean((recon - img_tensor) ** 2).item()

        return {
            "ae_error": float(error),
            "ae_latent_mean": float(latent.mean().item()),
            "ae_latent_std": float(latent.std().item()),
            "ae_time_ms": float(ae_time_ms),
        }

    @torch.no_grad()
    def run_temporal_on_window(self, frame_paths):
        frames = []
        for p in frame_paths:
            img_rgb = read_image_rgb(p, Config.IMG_SIZE_YOLO)
            frames.append(image_to_tensor(img_rgb))

        seq = torch.stack(frames, dim=0).unsqueeze(0).to(self.device)

        cuda_sync_if_needed()
        t0 = time.perf_counter()

        output = self.temporal_model(seq)

        cuda_sync_if_needed()
        temporal_time_ms = (time.perf_counter() - t0) * 1000.0

        probs = torch.exp(output["log_probs"])[0].detach().cpu().numpy()

        normal_prob = float(probs[4])
        defect_prob = float(1.0 - normal_prob)

        if defect_prob > Config.DEFECT_THRESHOLD:
            pred_idx = int(np.argmax(probs[:4]))
        else:
            pred_idx = 4

        ae_errors = output["ae_errors"][0].detach().cpu().numpy().tolist()
        attention = output["attention_weights"][0].detach().cpu().numpy().tolist()

        return {
            "predicted_class_idx": pred_idx,
            "predicted_class_name": Config.CLASS_NAMES[pred_idx],
            "defect_probability": defect_prob,
            "class_probabilities": {
                name: float(p) for name, p in zip(Config.CLASS_NAMES, probs)
            },
            "ae_errors": [float(v) for v in ae_errors],
            "attention_weights": [float(v) for v in attention],
            "frames": [str(p) for p in frame_paths],
            "temporal_time_ms": float(temporal_time_ms),
        }

    def run_folder(self, input_folder: str, output_folder: str, window_size=16, step=1):
        input_folder = Path(input_folder)
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        yolo_vis_dir = output_folder / "yolo_frames"
        yolo_vis_dir.mkdir(exist_ok=True)

        frames = list_frames(input_folder)
        if len(frames) == 0:
            raise ValueError("В папке нет кадров.")

        print(f"Найдено кадров: {len(frames)}")

        # 1. По каждому кадру: YOLO + AE + время
        per_frame_report = []
        yolo_times = []
        ae_times = []
        frame_total_times = []

        for i, frame_path in enumerate(frames):
            yolo_save_path = yolo_vis_dir / frame_path.name

            detections, yolo_time_ms = self.run_yolo_on_frame(frame_path, yolo_save_path)
            ae_info = self.run_ae_on_frame(frame_path)

            best_detection = None
            if len(detections) > 0:
                best_detection = max(detections, key=lambda d: d["confidence"])

            frame_total_time_ms = yolo_time_ms + ae_info["ae_time_ms"]

            yolo_times.append(yolo_time_ms)
            ae_times.append(ae_info["ae_time_ms"])
            frame_total_times.append(frame_total_time_ms)

            print(
                f"Frame {i:03d} ({frame_path.name}): "
                f"YOLO={yolo_time_ms:.2f} ms | "
                f"AE={ae_info['ae_time_ms']:.2f} ms | "
                f"total={frame_total_time_ms:.2f} ms"
            )

            per_frame_report.append({
                "frame_index": i,
                "frame_name": frame_path.name,
                "frame_path": str(frame_path),
                "yolo_visualization": str(yolo_save_path),
                "num_detections": len(detections),
                "best_detection": best_detection,
                "detections": detections,
                "yolo_time_ms": float(yolo_time_ms),
                "frame_total_time_ms": float(frame_total_time_ms),
                **ae_info,
            })

        with open(output_folder / "per_frame_report.json", "w", encoding="utf-8") as f:
            json.dump(per_frame_report, f, ensure_ascii=False, indent=2)

        # 2. По окнам: temporal
        windows_report = []
        temporal_times = []

        if len(frames) >= window_size:
            for start in range(0, len(frames) - window_size + 1, step):
                window_frames = frames[start:start + window_size]

                result = self.run_temporal_on_window(window_frames)
                result["window_start"] = start
                result["window_end"] = start + window_size - 1
                result["window_frame_names"] = [p.name for p in window_frames]

                windows_report.append(result)
                temporal_times.append(result["temporal_time_ms"])

                print(
                    f"⏱ Window {start:03d}-{start + window_size - 1:03d}: "
                    f"temporal={result['temporal_time_ms']:.2f} ms | "
                    f"class={result['predicted_class_name']} | "
                    f"defect_prob={result['defect_probability']:.4f}"
                )
        else:
            print(f"Кадров меньше {window_size}, temporal-анализ не выполнен.")

        with open(output_folder / "windows_report.json", "w", encoding="utf-8") as f:
            json.dump(windows_report, f, ensure_ascii=False, indent=2)

        # 3. Summary
        summary = {
            "input_folder": str(input_folder),
            "num_frames": len(frames),
            "window_size": window_size,
            "step": step,
            "max_defect_probability": 0.0,
            "best_window": None,
            "overall_prediction": "normal",
            "overall_defect_class": None,
            "num_frames_with_yolo_detections": sum(1 for x in per_frame_report if x["num_detections"] > 0),
            "max_ae_error": max((x["ae_error"] for x in per_frame_report), default=0.0),
            "timing": {
                "mean_yolo_time_ms": float(np.mean(yolo_times)) if yolo_times else None,
                "mean_ae_time_ms": float(np.mean(ae_times)) if ae_times else None,
                "mean_frame_total_time_ms": float(np.mean(frame_total_times)) if frame_total_times else None,
                "max_frame_total_time_ms": float(np.max(frame_total_times)) if frame_total_times else None,
                "mean_temporal_time_ms": float(np.mean(temporal_times)) if temporal_times else None,
                "max_temporal_time_ms": float(np.max(temporal_times)) if temporal_times else None,
                "meets_500ms_requirement_per_frame": bool(np.mean(frame_total_times) <= 500.0) if frame_total_times else None,
            }
        }

        if windows_report:
            best_window = max(windows_report, key=lambda x: x["defect_probability"])
            summary["max_defect_probability"] = float(best_window["defect_probability"])
            summary["best_window"] = {
                "window_start": best_window["window_start"],
                "window_end": best_window["window_end"],
                "window_frame_names": best_window["window_frame_names"],
                "predicted_class_name": best_window["predicted_class_name"],
                "defect_probability": best_window["defect_probability"],
                "class_probabilities": best_window["class_probabilities"],
                "temporal_time_ms": best_window["temporal_time_ms"],
            }

            if best_window["defect_probability"] > Config.DEFECT_THRESHOLD:
                summary["overall_prediction"] = "defect"
                summary["overall_defect_class"] = best_window["predicted_class_name"]

        with open(output_folder / "summary_report.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"Результаты сохранены в: {output_folder}")
        print(f"   - YOLO кадры: {yolo_vis_dir}")
        print(f"   - per_frame_report.json")
        print(f"   - windows_report.json")
        print(f"   - summary_report.json")

        if frame_total_times:
            print(
                f"Среднее время на 1 кадр: "
                f"YOLO={np.mean(yolo_times):.2f} ms | "
                f"AE={np.mean(ae_times):.2f} ms | "
                f"TOTAL={np.mean(frame_total_times):.2f} ms"
            )

        if temporal_times:
            print(
                f"⏱ Среднее время temporal на окно: {np.mean(temporal_times):.2f} ms | "
                f"max: {np.max(temporal_times):.2f} ms"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Папка с кадрами")
    parser.add_argument("--output", default="pipeline_output", help="Папка для результатов")
    parser.add_argument("--window", type=int, default=16, help="Размер temporal окна")
    parser.add_argument("--step", type=int, default=1, help="Шаг скользящего окна")
    parser.add_argument("--yolo_model", default=Config.YOLO_PATH)
    parser.add_argument("--ae_model", default=Config.AE_PATH)
    parser.add_argument("--temporal_model", default=Config.COMPLETE_MODEL_PATH)
    args = parser.parse_args()

    runner = PipelineRunner(
        yolo_path=args.yolo_model,
        ae_path=args.ae_model,
        temporal_path=args.temporal_model,
        device=Config.DEVICE,
    )

    runner.run_folder(
        input_folder=args.input,
        output_folder=args.output,
        window_size=args.window,
        step=args.step,
    )


if __name__ == "__main__":
    main()