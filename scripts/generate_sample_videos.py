"""Builds the 3 demo site videos used by configs/sites.yaml.

A synthetic drawing of a person almost never triggers a real object
detector, so instead of drawing humanoid shapes from scratch we start from
a real, publicly available pedestrian video - OpenCV's own
samples/data/vtest.avi (BSD-licensed sample asset, shipped by the OpenCV
project specifically for this kind of testing) - and overlay a
detector-friendly "PPE" blob (hard-hat-colored ellipse on the head, hi-vis
rectangle on the torso) onto a controllable fraction of the real people
YOLO finds in it. That gives each site a different, reproducible violation
rate for a believable cross-site pattern, while every frame is still a real
photographed person, so YOLOv8's person detector reliably fires on it.

This is a simulated-compliance overlay for demo purposes, not a validated
real-world PPE detector - see services/vision_agent/detection.py.

Run: .venv/bin/python -m scripts.generate_sample_videos
"""

from __future__ import annotations

import random
import urllib.request
from pathlib import Path

import cv2
from ultralytics import YOLO

SOURCE_VIDEO_URL = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/vtest.avi"
DATA_DIR = Path("data/videos")
SOURCE_PATH = DATA_DIR / "_source_vtest.avi"

# site_id -> probability that a given detected person, in a given frame,
# gets drawn as PPE-compliant (i.e. NOT flagged as a violation downstream).
SITE_COMPLIANCE_RATES = {
    "site-a": 0.75,  # mostly compliant - the "healthy" site
    "site-b": 0.20,  # mostly violations - the site with a real problem
    "site-c": 0.50,  # mixed - webcam fallback video
}

HARD_HAT_COLOR_BGR = (0, 215, 255)  # bright yellow
VEST_COLOR_BGR = (0, 140, 255)  # hi-vis orange
COCO_PERSON_CLASS_ID = 0


def _ensure_source_video() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if SOURCE_PATH.exists():
        return
    print(f"Downloading {SOURCE_VIDEO_URL} -> {SOURCE_PATH}")
    urllib.request.urlretrieve(SOURCE_VIDEO_URL, SOURCE_PATH)


def _draw_ppe(frame, x1: int, y1: int, x2: int, y2: int) -> None:
    height = y2 - y1
    head_y2 = y1 + int(0.25 * height)
    torso_y1 = y1 + int(0.25 * height)
    torso_y2 = y1 + int(0.75 * height)

    center = ((x1 + x2) // 2, (y1 + head_y2) // 2)
    axes = (max((x2 - x1) // 2, 1), max((head_y2 - y1) // 2, 1))
    cv2.ellipse(frame, center, axes, 0, 0, 360, HARD_HAT_COLOR_BGR, thickness=-1)
    cv2.rectangle(frame, (x1, torso_y1), (x2, torso_y2), VEST_COLOR_BGR, thickness=-1)


def main() -> None:
    _ensure_source_video()

    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(str(SOURCE_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source video at {SOURCE_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")

    rngs = {site_id: random.Random(hash(site_id) & 0xFFFFFFFF) for site_id in SITE_COMPLIANCE_RATES}
    writers = {
        site_id: cv2.VideoWriter(str(DATA_DIR / f"{site_id}.mp4"), fourcc, fps, (width, height))
        for site_id in SITE_COMPLIANCE_RATES
    }

    frame_count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_count += 1

            results = model.predict(frame, classes=[COCO_PERSON_CLASS_ID], conf=0.4, verbose=False)
            boxes = [tuple(int(v) for v in b.xyxy[0].tolist()) for b in results[0].boxes] if results else []

            for site_id, writer in writers.items():
                site_frame = frame.copy()
                rng = rngs[site_id]
                compliance_rate = SITE_COMPLIANCE_RATES[site_id]
                for x1, y1, x2, y2 in boxes:
                    if rng.random() < compliance_rate:
                        _draw_ppe(site_frame, x1, y1, x2, y2)
                writer.write(site_frame)

            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames...")
    finally:
        cap.release()
        for writer in writers.values():
            writer.release()

    print(f"Done: {frame_count} frames -> {', '.join(f'{s}.mp4' for s in SITE_COMPLIANCE_RATES)}")


if __name__ == "__main__":
    main()
