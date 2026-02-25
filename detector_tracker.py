import time
import os
import urllib.request
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np
from ultralytics import YOLO

YOLO_WEIGHTS_MAP = {
    "yolov8n": "yolov8n.pt",
    "yolov8s": "yolov8s.pt",
    "yolov8m": "yolov8m.pt",
}


def ensure_model(model_name = "yolov8n", model_dir = "models"):
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    filename  = YOLO_WEIGHTS_MAP.get(model_name, f"{model_name}.pt")
    local_path = os.path.join(model_dir, filename)

    if not os.path.exists(local_path):
        tmp = YOLO(filename)
        import shutil
        from ultralytics.utils import WEIGHTS_DIR
        cached = WEIGHTS_DIR / filename
        if cached.exists():
            shutil.copy(str(cached), local_path)
        else:
            alt = Path(filename)
            if alt.exists():
                shutil.copy(str(alt), local_path)
    return local_path

class TrackResult:
    def __init__(
        self,
        frame_id: int,
        obj_id: int,
        class_id: int,
        class_name: str,
        confidence: float,
        x1: int, y1: int, x2: int, y2: int,
        inference_time: float,
    ):
        self.frame_id      = frame_id
        self.obj_id        = obj_id
        self.class_id      = class_id
        self.class_name    = class_name
        self.confidence    = confidence
        self.x1, self.y1  = x1, y1
        self.x2, self.y2  = x2, y2
        self.inference_time = inference_time

    def to_dict(self):
        return {
            "frame_id":      self.frame_id,
            "object_id":     self.obj_id,
            "class_name":    self.class_name,
            "bbox":          f"[{self.x1},{self.y1},{self.x2},{self.y2}]",
            "confidence":    round(self.confidence, 4),
            "inference_time": round(self.inference_time * 1000, 3),  # ms
        }

class DetectorTracker:
    def __init__(
        self,
        model_name = "yolov8n",
        model_dir = "models",
        conf = 0.25,
        iou = 0.45,
        device = "",
        track_max_age= 30,
        img_size= 640,
    ):
        self.model_name = model_name
        self.model_dir = model_dir
        self.conf = conf
        self.iou_thresh = iou
        self.device = device
        self.track_max_age = track_max_age
        self.img_size = img_size

        self._model = None
        self._weights_path = ""


    def load(self):
        self._weights_path = ensure_model(self.model_name, self.model_dir)
        self._model = YOLO(self._weights_path)
        sample = np.zeros((640, 640, 3), dtype=np.uint8)
        self._model.track(sample, persist=True, verbose=False,
                          conf=self.conf, iou=self.iou_thresh,
                          device=self.device, imgsz=self.img_size)
        self._model.predictor = None

    @property
    def weights_path(self):
        return self._weights_path

    @property
    def class_names(self):
        if self._model:
            return self._model.names
        return {}

    def process_frame(self,frame,frame_id):
        assert self._model is not None, "Call load() first."

        t0 = time.perf_counter()
        results = self._model.track(
            frame,
            persist=True,
            verbose=False,
            conf=self.conf,
            iou=self.iou_thresh,
            device=self.device,
            imgsz=self.img_size,
            tracker="bytetrack.yaml",
        )
        inference_time = time.perf_counter() - t0

        track_results: List[TrackResult] = []

        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                if box.id is None:
                    continue
                obj_id    = int(box.id.item())
                cls_id    = int(box.cls.item())
                cls_name  = self._model.names[cls_id]
                conf_val  = float(box.conf.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                track_results.append(TrackResult(
                    frame_id=frame_id,
                    obj_id=obj_id,
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf_val,
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    inference_time=inference_time,
                ))

        return track_results, inference_time

    def reset_tracker(self):
        if self._model and self._model.predictor:
            self._model.predictor = None
