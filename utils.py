import os
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
import psutil

def ensure_dirs(*paths):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def get_video_paths(folder):
    exts = {".mp4", ".avi", ".webm"}
    paths = [
        str(p) for p in sorted(Path(folder).iterdir())
        if p.suffix.lower() in exts
    ]
    return paths

def get_video_meta(cap: cv2.VideoCapture):
    return {
        "width":     int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height":    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps":       cap.get(cv2.CAP_PROP_FPS) or 30.0,
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }

_PALETTE = [
    (255,  56,  56), (255, 157,  51), ( 77, 255, 224), ( 77,  83, 255),
    (255,  77, 195), ( 56, 255, 155), (255, 128,   0), (128, 128, 255),
    (255, 255,   0), (  0, 255, 255), (255,   0, 255), (128,   0, 255),
    (  0, 128, 255), (255,  64, 128), ( 64, 255,  64), (200, 200, 200),
    (255, 165,   0), (  0, 206, 209), (220,  20,  60), (127, 255,   0),
]


def id_color(obj_id):
    return _PALETTE[int(obj_id) % len(_PALETTE)]

def draw_box(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    obj_id: int,
    class_name: str,
    confidence: float,):
    color = id_color(obj_id)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"#{obj_id} {class_name} {confidence:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def draw_hud(
    frame: np.ndarray,
    frame_id: int,
    total_frames: int,
    fps: float,
    counts: Dict[str, int],):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 40), (20, 20, 20), -1)

    info = f"Frame {frame_id}/{total_frames}  |  FPS: {fps:.1f}"
    cv2.putText(frame, info, (8, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA)

    count_str = "  ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
    cv2.putText(frame, count_str, (w // 2, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 220, 100), 1, cv2.LINE_AA)
    return frame

def append_csv_rows(rows, csv_path):
    df  = pd.DataFrame(rows)
    hdr = not Path(csv_path).exists()
    df.to_csv(csv_path, mode="a", header=hdr, index=False)


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path):
    with open(path) as f:
        return json.load(f)

def get_model_size_mb(path):
    try:
        return os.path.getsize(path) / 1_048_576
    except OSError:
        return 0.0


def get_gpu_memory_mb():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1_048_576
    except Exception:
        pass
    return 0.0

def get_process_memory_mb():
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except Exception:
        return 0.0


