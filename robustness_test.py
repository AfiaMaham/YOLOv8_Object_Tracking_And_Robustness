import os
import time
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from detector_tracker import DetectorTracker
from utils import (
    ensure_dirs, get_video_paths, get_video_meta,save_json)

INPUT_DIR         = "input_videos"
ROBUSTNESS_DIR    = "outputs/robustness"
MODEL_DIR         = "models"
MODEL_NAME        = "yolov8n"
CONF_THRESHOLD    = 0.25
IOU_THRESHOLD     = 0.45
IMG_SIZE          = 640
MAX_FRAMES        = 150


def corruption_gaussian_noise(frame, severity = 2):
    sigma_map = {1: 12, 2: 28, 3: 55}
    sigma = sigma_map.get(severity, 28)
    noise = np.random.normal(0, sigma, frame.shape).astype(np.float32)
    out   = np.clip(frame.astype(np.float32) + noise, 0, 255)
    return out.astype(np.uint8)


def corruption_motion_blur(frame, severity = 2):
    k_map = {1: 9, 2: 19, 3: 33}
    k = k_map.get(severity, 19)
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0 / k
    return cv2.filter2D(frame, -1, kernel)


def corruption_low_brightness(frame, severity = 2):
    alpha_map = {1: 0.55, 2: 0.30, 3: 0.12}
    alpha = alpha_map.get(severity, 0.30)
    return np.clip(frame.astype(np.float32) * alpha, 0, 255).astype(np.uint8)


def corruption_occlusion(frame, severity = 2, seed = 0):
    params = {
        1: {"n": 3,  "min_s": 30, "max_s": 60},
        2: {"n": 6,  "min_s": 50, "max_s": 100},
        3: {"n": 10, "min_s": 70, "max_s": 140},
    }
    p   = params.get(severity, params[2])
    rng = np.random.default_rng(seed)
    out = frame.copy()
    h, w = frame.shape[:2]
    for _ in range(p["n"]):
        pw = int(rng.integers(p["min_s"], p["max_s"]))
        ph = int(rng.integers(p["min_s"], p["max_s"]))
        px = int(rng.integers(0, max(1, w - pw)))
        py = int(rng.integers(0, max(1, h - ph)))
        out[py:py+ph, px:px+pw] = 0
    return out


CORRUPTION_FRAME_DROP_RATE = 0.20

CORRUPTIONS = {
    "clean":           None,
    "gaussian_noise":  corruption_gaussian_noise,
    "motion_blur":     corruption_motion_blur,
    "low_brightness":  corruption_low_brightness,
    "occlusion":       corruption_occlusion,
    "frame_dropping":  "frame_drop",
}

CORRUPTION_LABELS = {
    "clean":          "Clean",
    "gaussian_noise": "Gaussian Noise",
    "motion_blur":    "Motion Blur",
    "low_brightness": "Low Brightness",
    "occlusion":      "Occlusion Patches",
    "frame_dropping": "Frame Dropping",
}

def _run_pipeline(video_path, corruption_name, dt: DetectorTracker, max_frames: int = MAX_FRAMES,):
    dt.reset_tracker()
    corrupt_fn = CORRUPTIONS[corruption_name]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}

    meta         = get_video_meta(cap)
    total_frames = min(meta["total_frames"], max_frames)
    is_frame_drop = (corrupt_fn == "frame_drop")

    latencies:   List[float] = []
    unique_ids   = set()
    class_counts: Dict[str, set] = defaultdict(set)
    frames_processed = 0
    frames_skipped   = 0

    t_start = time.perf_counter()

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        if is_frame_drop:
            if random.random() < CORRUPTION_FRAME_DROP_RATE:
                frames_skipped += 1
                continue

        if corrupt_fn and corrupt_fn != "frame_drop":
            frame = corrupt_fn(frame, severity=2)

        track_results, inf_time = dt.process_frame(frame, frame_idx)
        latencies.append(inf_time)
        frames_processed += 1

        for tr in track_results:
            unique_ids.add(tr.obj_id)
            class_counts[tr.class_name].add(tr.obj_id)

    cap.release()

    total_time = time.perf_counter() - t_start
    avg_fps    = frames_processed / total_time if total_time > 0 else 0.0
    avg_lat    = sum(latencies) / len(latencies) * 1000 if latencies else 0.0
    total_dets = sum(len(v) for v in class_counts.values())

    return {
        "corruption":        corruption_name,
        "frames_attempted":  total_frames,
        "frames_processed":  frames_processed,
        "frames_skipped":    frames_skipped,
        "unique_objects":    len(unique_ids),
        "total_unique_per_class": {k: len(v) for k, v in class_counts.items()},
        "average_fps":       round(avg_fps,  2),
        "average_latency_ms": round(avg_lat, 3),
        "total_time_s":      round(total_time, 3),
    }



_COLORS = ["#4A90D9", "#E85D4A", "#F5A623", "#7ED321", "#9B59B6", "#1ABC9C"]


def _plot_comparison(results: List[Dict], save_path: str, video_name: str):
    labels      = [CORRUPTION_LABELS[r["corruption"]] for r in results]
    fps_vals    = [r["average_fps"]       for r in results]
    lat_vals    = [r["average_latency_ms"] for r in results]
    uid_vals    = [r["unique_objects"]    for r in results]
    drop_vals   = [r["frames_skipped"]    for r in results]

    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(
        f"Robustness Analysis — {video_name}",
        fontsize=15, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.38)

    def _bar(ax, vals, ylabel, title, highlight_idx=0):
        colors = [_COLORS[i % len(_COLORS)] for i in range(len(labels))]
        bars   = ax.bar(range(len(labels)), vals, color=colors, edgecolor="white",
                        alpha=0.88)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(vals)*0.01,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8)
        return bars


    ax1 = fig.add_subplot(gs[0, 0])
    _bar(ax1, fps_vals, "Frames Per Second", "Inference Speed (FPS)")
    if fps_vals:
        ax1.axhline(fps_vals[0], ls="--", color="grey", alpha=0.5, lw=1.2,
                    label=f"Clean: {fps_vals[0]:.1f} FPS")
        ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(gs[0, 1])
    _bar(ax2, lat_vals, "Latency (ms)", "Average Inference Latency")

    ax3 = fig.add_subplot(gs[0, 2])
    _bar(ax3, uid_vals, "Unique Object IDs", "Unique Objects Tracked")

    ax4 = fig.add_subplot(gs[1, 0])
    clean_fps = fps_vals[0] if fps_vals else 1.0
    drops = [max(0.0, (clean_fps - f) / clean_fps * 100) for f in fps_vals]
    colors_d = ["#4A90D9"] + ["#E85D4A"] * (len(drops) - 1)
    bars = ax4.bar(range(len(labels)), drops, color=colors_d, edgecolor="white", alpha=0.88)
    ax4.set_xticks(range(len(labels)))
    ax4.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax4.set_ylabel("FPS Drop (%)", fontsize=9)
    ax4.set_title("FPS Degradation vs Clean Baseline", fontsize=11, fontweight="bold")
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)
    ax4.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, drops):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{v:.1f}%", ha="center", va="bottom", fontsize=8)

    ax5 = fig.add_subplot(gs[1, 1])
    clean_uid = uid_vals[0] if uid_vals else 1
    uid_drops = [max(0.0, (clean_uid - u) / max(clean_uid, 1) * 100) for u in uid_vals]
    bars5 = ax5.bar(range(len(labels)), uid_drops, color=colors_d,
                    edgecolor="white", alpha=0.88)
    ax5.set_xticks(range(len(labels)))
    ax5.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax5.set_ylabel("Unique Object Drop (%)", fontsize=9)
    ax5.set_title("Tracking Quality Degradation", fontsize=11, fontweight="bold")
    ax5.spines["top"].set_visible(False)
    ax5.spines["right"].set_visible(False)
    ax5.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars5, uid_drops):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{v:.1f}%", ha="center", va="bottom", fontsize=8)

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    lines = ["SUMMARY TABLE\n"]
    lines.append(f"{'Corruption':<22} {'FPS':>7} {'Lat(ms)':>9} {'UIDs':>6}")
    lines.append("─" * 47)
    for r in results:
        lines.append(
            f"{CORRUPTION_LABELS[r['corruption']]:<22} "
            f"{r['average_fps']:>7.1f} "
            f"{r['average_latency_ms']:>9.2f} "
            f"{r['unique_objects']:>6}"
        )
    ax6.text(0.03, 0.97, "\n".join(lines),
             transform=ax6.transAxes, fontsize=8.5,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7))

    plt.savefig(save_path, bbox_inches="tight", dpi=130)
    plt.close(fig)


def main():
    ensure_dirs(INPUT_DIR, ROBUSTNESS_DIR, MODEL_DIR)

    video_paths = get_video_paths(INPUT_DIR)

    dt = DetectorTracker(
        model_name=MODEL_NAME,
        model_dir=MODEL_DIR,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        img_size=IMG_SIZE,
    )
    dt.load()

    for vp in video_paths:
        stem = Path(vp).stem
        print(f"\n{'='*60}")
        print(f"  Robustness test: {Path(vp).name}")
        print(f"{'='*60}")

        all_results = []

        for corruption_name in CORRUPTIONS:
            print(f"\n  [{corruption_name}] running…")
            metrics = _run_pipeline(vp, corruption_name, dt, max_frames=MAX_FRAMES)
            if metrics:
                all_results.append(metrics)
                print(f"     FPS: {metrics['average_fps']:.1f}  "
                      f"| Lat: {metrics['average_latency_ms']:.1f}ms  "
                      f"| UIDs: {metrics['unique_objects']}")

        if not all_results:
            continue

        json_path = os.path.join(ROBUSTNESS_DIR, f"{stem}_robustness_results.json")
        save_json({"video": Path(vp).name, "results": all_results}, json_path)

        plot_path = os.path.join(ROBUSTNESS_DIR, f"{stem}_robustness_comparison.png")
        _plot_comparison(all_results, plot_path, Path(vp).name)



if __name__ == "__main__":
    main()
