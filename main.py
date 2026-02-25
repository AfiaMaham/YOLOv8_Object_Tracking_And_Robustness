import os
import time
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
from detector_tracker import DetectorTracker
from utils import (
    ensure_dirs, get_video_paths, get_video_meta,
    draw_box, draw_hud,
    append_csv_rows, save_json)

INPUT_DIR          = "input_videos"
OUTPUT_VIDEO_DIR   = "outputs/annotated_videos"
OUTPUT_LOG_DIR     = "outputs/logs"
OUTPUT_REPORT_DIR  = "outputs/reports"
MODEL_DIR          = "models"

MODEL_NAME         = "yolov8n"
CONF_THRESHOLD     = 0.25
IOU_THRESHOLD      = 0.45
IMG_SIZE           = 640


def create_video_writer( output_path, fps, width, height):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))


def process_video(
    video_path: str,
    dt: DetectorTracker,
    out_video_dir: str,
    out_log_dir: str,
    out_report_dir: str,):

    stem      = Path(video_path).stem
    csv_path  = os.path.join(out_log_dir,   f"{stem}_log.csv")
    json_path = os.path.join(out_report_dir, f"{stem}_report.json")
    out_vid   = os.path.join(out_video_dir,  f"{stem}_annotated.mp4")

    if os.path.exists(csv_path):
        os.remove(csv_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[error] Cannot open: {video_path}")
        return {}

    meta   = get_video_meta(cap)
    writer = create_video_writer(
        out_vid, meta["fps"], meta["width"], meta["height"]
    )

    dt.reset_tracker()

    frame_id        = 0
    all_latencies   = []
    unique_ids      = set()
    class_counts    = defaultdict(set)
    csv_buffer      = []
    running_fps_sum = 0.0
    running_fps_cnt = 0

    total_frames = meta["total_frames"]
    print(f"\nProcessing: {Path(video_path).name}  "
          f"({total_frames} frames, {meta['fps']:.1f} FPS, "
          f"{meta['width']}×{meta['height']})")

    pipeline_start = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        track_results, inf_time = dt.process_frame(frame, frame_id)
        all_latencies.append(inf_time)

        frame_counts: dict = defaultdict(int)
        for tr in track_results:
            unique_ids.add(tr.obj_id)
            class_counts[tr.class_name].add(tr.obj_id)
            frame_counts[tr.class_name] += 1
            csv_buffer.append(tr.to_dict())

            draw_box(frame,
                     tr.x1, tr.y1, tr.x2, tr.y2,
                     tr.obj_id, tr.class_name, tr.confidence)

        elapsed = time.perf_counter() - pipeline_start
        fps_now = (frame_id + 1) / elapsed if elapsed > 0 else 0.0

        draw_hud(frame, frame_id + 1, total_frames, fps_now, dict(frame_counts))
        writer.write(frame)

        if len(csv_buffer) >= 200:
            append_csv_rows(csv_buffer, csv_path)
            csv_buffer.clear()

        frame_id += 1

        if frame_id % 100 == 0 or frame_id == total_frames:
            print(f"   frame {frame_id:>5}/{total_frames}  "
                  f"| FPS: {fps_now:.1f}  "
                  f"| latency: {inf_time*1000:.1f}ms  "
                  f"| tracks: {len(track_results)}")

    if csv_buffer:
        append_csv_rows(csv_buffer, csv_path)

    cap.release()
    writer.release()

    total_time = time.perf_counter() - pipeline_start
    avg_fps    = frame_id / total_time if total_time > 0 else 0.0
    avg_lat_ms = (sum(all_latencies) / len(all_latencies) * 1000
                  if all_latencies else 0.0)

    summary = {
        "video":              Path(video_path).name,
        "total_frames":       frame_id,
        "unique_objects":     len(unique_ids),
        "average_fps":        round(avg_fps,    2),
        "average_latency": round(avg_lat_ms, 3),
        "min_latency":     round(min(all_latencies) * 1000, 3) if all_latencies else 0,
        "max_latency":     round(max(all_latencies) * 1000, 3) if all_latencies else 0,
        "total_time":       round(total_time, 3),
        "model":              MODEL_NAME,
        "output_video":       out_vid,
        "log_csv":            csv_path,
    }

    save_json(summary, json_path)
    return summary

def main():
    ensure_dirs(INPUT_DIR, OUTPUT_VIDEO_DIR, OUTPUT_LOG_DIR,
                OUTPUT_REPORT_DIR, MODEL_DIR)

    video_paths = get_video_paths(INPUT_DIR)

    dt = DetectorTracker(
        model_name=MODEL_NAME,
        model_dir=MODEL_DIR,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        img_size=IMG_SIZE,
    )
    dt.load()
    all_summaries = []

    for vp in video_paths:
        summary = process_video(
            vp, dt,
            OUTPUT_VIDEO_DIR, OUTPUT_LOG_DIR, OUTPUT_REPORT_DIR,
        )
        if summary:
            all_summaries.append(summary)
            _print_summary(summary)

    if all_summaries:
        save_json(all_summaries,
                  os.path.join(OUTPUT_REPORT_DIR, "combined_report.json"))


def _print_summary(s: dict):
    print("\n" + "─" * 55)
    print(f"  Video           : {s['video']}")
    print(f"  Total frames    : {s['total_frames']}")
    print(f"  Unique objects  : {s['unique_objects']}")
    print(f"  Average FPS     : {s['average_fps']}")
    print(f"  Avg latency     : {s['average_latency']} ms")
    print(f"  Output video    : {s['output_video']}")
    print("─" * 55)


def _create_demo_video(path: str, n_frames: int = 120):
    import random
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w, h   = 854, 480
    writer = cv2.VideoWriter(path, fourcc, 30, (w, h))

    objects = [
        {"x": random.randint(50, 400), "y": random.randint(50, 300),
         "vx": random.choice([-4, -3, 3, 4]), "vy": random.choice([-3, -2, 2, 3]),
         "color": (random.randint(50,255), random.randint(50,255), random.randint(50,255)),
         "w": random.randint(60, 120), "h": random.randint(60, 120)}
        for _ in range(5)
    ]

    for _ in range(n_frames):
        frame = np.full((h, w, 3), 40, dtype=np.uint8)
        for obj in objects:
            obj["x"] = int(np.clip(obj["x"] + obj["vx"], 0, w - obj["w"]))
            obj["y"] = int(np.clip(obj["y"] + obj["vy"], 0, h - obj["h"]))
            if obj["x"] <= 0 or obj["x"] >= w - obj["w"]:
                obj["vx"] *= -1
            if obj["y"] <= 0 or obj["y"] >= h - obj["h"]:
                obj["vy"] *= -1
            cv2.rectangle(frame, (obj["x"], obj["y"]),
                          (obj["x"]+obj["w"], obj["y"]+obj["h"]),
                          obj["color"], -1)
        writer.write(frame)

    writer.release()

if __name__ == "__main__":
    main()
