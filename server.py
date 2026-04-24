import json
import os
import shutil
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config import Config


BASE_DIR = Path(__file__).resolve().parent
WORKSPACE = Path(Config.WORKSPACE)
DATA_DIR = Path(Config.DATA_ROOT)
YOLO_DIR = DATA_DIR / "yolo_train"
YOLO_IMAGES = YOLO_DIR / "images"
YOLO_LABELS = YOLO_DIR / "labels"
NORMAL_DIR = DATA_DIR / "normal"
TEMPORAL_DIRS = {
    "connection": DATA_DIR / "defect_connection" / "images",
    "foreign object": DATA_DIR / "defect_foreign" / "images",
    "garbage": DATA_DIR / "defect_garbage" / "images",
    "point": DATA_DIR / "defect_point" / "images",
    "normal": DATA_DIR / "normal",
}
CLASS_TO_ID = {name: i for i, name in enumerate(Config.YOLO_CLASS_NAMES)}
JOBS_FILE = WORKSPACE / "jobs.json"


app = FastAPI(title="Harvester Trainer")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount("/workspace", StaticFiles(directory=str(WORKSPACE)), name="workspace")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def ensure_dirs():
    for p in [YOLO_IMAGES, YOLO_LABELS, NORMAL_DIR, *TEMPORAL_DIRS.values(), WORKSPACE / "logs", WORKSPACE / "runs"]:
        p.mkdir(parents=True, exist_ok=True)
    ensure_data_yaml()
    if not JOBS_FILE.exists():
        JOBS_FILE.write_text("{}", encoding="utf-8")


def ensure_data_yaml():
    yaml_path = YOLO_DIR / "data.yaml"
    yaml_path.write_text(
        "".join([
            f"path: {YOLO_DIR.resolve()}\n",
            "train: images\n",
            "val: images\n",
            f"names: {Config.YOLO_CLASS_NAMES}\n",
            f"nc: {len(Config.YOLO_CLASS_NAMES)}\n",
        ]),
        encoding="utf-8",
    )


def read_jobs() -> Dict[str, Dict]:
    if not JOBS_FILE.exists():
        return {}
    return json.loads(JOBS_FILE.read_text(encoding="utf-8"))


def write_jobs(jobs: Dict[str, Dict]):
    JOBS_FILE.write_text(json.dumps(jobs, ensure_ascii=False, indent=2), encoding="utf-8")


def list_images(folder: Path) -> List[Path]:
    files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        files.extend(folder.glob(ext))
    return sorted(files)


def read_yolo_labels(image_name: str):
    label_path = YOLO_LABELS / (Path(image_name).stem + ".txt")
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, xc, yc, w, h = parts
        boxes.append({
            "class_id": int(float(cls)),
            "x_center": float(xc),
            "y_center": float(yc),
            "width": float(w),
            "height": float(h),
        })
    return boxes


def write_yolo_labels(image_name: str, boxes: List[dict]):
    label_path = YOLO_LABELS / (Path(image_name).stem + ".txt")
    lines = []
    for b in boxes:
        lines.append(
            f"{int(b['class_id'])} {float(b['x_center']):.6f} {float(b['y_center']):.6f} {float(b['width']):.6f} {float(b['height']):.6f}"
        )
    label_path.write_text("\n".join(lines), encoding="utf-8")


def dataset_stats():
    yolo_images = list_images(YOLO_IMAGES)
    labeled = sum(1 for p in yolo_images if (YOLO_LABELS / (p.stem + ".txt")).exists())
    temporal_stats = {k: len(list_images(v if k != "normal" else NORMAL_DIR)) for k, v in TEMPORAL_DIRS.items()}
    return {
        "yolo_images": len(yolo_images),
        "yolo_labeled": labeled,
        "autoencoder_normal": len(list_images(NORMAL_DIR)),
        "temporal": temporal_stats,
        "weights": {
            "yolo": Path(Config.YOLO_PATH).exists(),
            "ae": Path(Config.AE_PATH).exists(),
            "complete": Path(Config.COMPLETE_MODEL_PATH).exists(),
        },
    }


def save_upload(file: UploadFile, dest: Path):
    suffix = Path(file.filename).suffix.lower() or ".jpg"
    safe_name = Path(file.filename).name.replace(" ", "_")
    target = dest / safe_name
    counter = 1
    while target.exists():
        target = dest / f"{Path(safe_name).stem}_{counter}{suffix}"
        counter += 1
    with target.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    return target


def start_job(stage: str, script_name: str):
    ensure_dirs()
    jobs = read_jobs()
    if any(j.get("status") == "running" for j in jobs.values()):
        raise HTTPException(status_code=409, detail="Another training job is already running")

    job_id = uuid.uuid4().hex[:10]
    log_path = WORKSPACE / "logs" / f"{stage}_{job_id}.log"
    env = os.environ.copy()
    env["WORKSPACE_DIR"] = str(WORKSPACE)
    cmd = [sys.executable, str(BASE_DIR / script_name)]
    with log_path.open("wb") as log_file:
        proc = subprocess.Popen(cmd, cwd=str(BASE_DIR), stdout=log_file, stderr=subprocess.STDOUT, env=env)

    jobs[job_id] = {
        "id": job_id,
        "stage": stage,
        "script": script_name,
        "status": "running",
        "pid": proc.pid,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "log_path": str(log_path),
    }
    write_jobs(jobs)
    return jobs[job_id]


def refresh_jobs():
    jobs = read_jobs()
    changed = False
    for job in jobs.values():
        if job.get("status") != "running":
            continue
        pid = job.get("pid")
        try:
            result = subprocess.run(["bash", "-lc", f"ps -p {pid} -o pid="], capture_output=True, text=True)
            alive = bool(result.stdout.strip())
        except Exception:
            alive = False
        if not alive:
            log_text = Path(job["log_path"]).read_text(encoding="utf-8", errors="ignore") if Path(job["log_path"]).exists() else ""
            job["status"] = "finished" if "Traceback" not in log_text else "error"
            changed = True
    if changed:
        write_jobs(jobs)
    return jobs


@app.on_event("startup")
def startup_event():
    ensure_dirs()


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    stats = dataset_stats()
    jobs = list(refresh_jobs().values())[::-1]
    return templates.TemplateResponse("index.html", {"request": request, "stats": stats, "jobs": jobs, "classes": Config.YOLO_CLASS_NAMES})


@app.get("/datasets", response_class=HTMLResponse)
def datasets_page(request: Request):
    stats = dataset_stats()
    yolo_images = [p.name for p in list_images(YOLO_IMAGES)]
    return templates.TemplateResponse("datasets.html", {"request": request, "stats": stats, "yolo_images": yolo_images, "temporal_dirs": TEMPORAL_DIRS})


@app.post("/upload")
async def upload_files(target: str = Form(...), files: List[UploadFile] = File(...), defect_class: str = Form("")):
    ensure_dirs()
    if target == "yolo":
        dest = YOLO_IMAGES
    elif target == "autoencoder":
        dest = NORMAL_DIR
    elif target == "temporal":
        if defect_class not in TEMPORAL_DIRS:
            raise HTTPException(status_code=400, detail="Invalid temporal class")
        dest = TEMPORAL_DIRS[defect_class]
    else:
        raise HTTPException(status_code=400, detail="Invalid upload target")

    saved = []
    for file in files:
        saved.append(save_upload(file, dest).name)
    ensure_data_yaml()
    return RedirectResponse(url="/datasets", status_code=303)


@app.get("/label", response_class=HTMLResponse)
def label_list(request: Request, image: str | None = None):
    images = list_images(YOLO_IMAGES)
    image_name = image or (images[0].name if images else None)
    boxes = read_yolo_labels(image_name) if image_name else []
    return templates.TemplateResponse("label.html", {
        "request": request,
        "images": [p.name for p in images],
        "image_name": image_name,
        "boxes": boxes,
        "classes": list(enumerate(Config.YOLO_CLASS_NAMES)),
    })


@app.get("/pimg/{image_name}")
def project_image(image_name: str):
    path = YOLO_IMAGES / image_name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return RedirectResponse(url=f"/workspace/data/yolo_train/images/{image_name}")


@app.post("/label/save")
async def save_label(payload: Request):
    data = await payload.json()
    image_name = data.get("img")
    boxes = data.get("boxes", [])
    if not image_name:
        return JSONResponse({"ok": False, "error": "Image name is required"}, status_code=400)
    write_yolo_labels(image_name, boxes)
    return {"ok": True}


@app.get("/train", response_class=HTMLResponse)
def train_page(request: Request):
    stats = dataset_stats()
    jobs = list(refresh_jobs().values())[::-1]
    return templates.TemplateResponse("train.html", {"request": request, "stats": stats, "jobs": jobs})


@app.post("/train/{stage}")
def launch_training(stage: str):
    mapping = {
        "yolo": "train_yolo.py",
        "autoencoder": "train_autoencoder.py",
        "complete": "train_complete.py",
    }
    if stage not in mapping:
        raise HTTPException(status_code=404, detail="Unknown training stage")
    job = start_job(stage, mapping[stage])
    return RedirectResponse(url="/train", status_code=303)


@app.get("/api/jobs")
def api_jobs():
    return list(refresh_jobs().values())[::-1]


@app.get("/api/log/{job_id}")
def api_log(job_id: str):
    jobs = refresh_jobs()
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    log_path = Path(job["log_path"])
    text = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.exists() else ""
    return {"job": job, "text": text[-40000:]}
