"""PPE-violation detection: the thing a partner's real trained model would
eventually replace.

`Detector` is the interface everything upstream (the MCP tools, the capture
loop, the event store) depends on. `YoloPpeHeuristicDetector` is a genuinely
working stand-in: YOLOv8n (stock COCO weights) locates people, then OpenCV
color-heuristics over the head/torso regions decide whether a hard hat or
hi-vis vest is present. It is not as accurate as a model fine-tuned on real
PPE data, but it runs real inference end-to-end today. Swapping in a real
trained model later means writing one new Detector subclass — nothing else
in the system needs to change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO

from services.common.schemas import BoundingBox, ViolationType

_COCO_PERSON_CLASS_ID = 0

# HSV ranges (OpenCV: H 0-179, S/V 0-255) for colors typically worn as hard
# hats. Broad on purpose - false negatives (missing a real hat) are far more
# costly for a safety demo than false positives.
_HARD_HAT_HSV_RANGES: list[tuple[np.ndarray, np.ndarray]] = [
    (np.array([15, 80, 120]), np.array([35, 255, 255])),  # yellow/orange
    (np.array([0, 80, 120]), np.array([10, 255, 255])),  # red
    (np.array([170, 80, 120]), np.array([179, 255, 255])),  # red (wrap-around)
    (np.array([95, 60, 90]), np.array([130, 255, 255])),  # blue
    (np.array([0, 0, 180]), np.array([179, 40, 255])),  # white
]

# Hi-vis safety vest colors: fluorescent yellow-green / orange, higher
# saturation/value floor than the hard-hat ranges since vests are typically
# more uniformly bright than a helmet glinting in the light.
_VEST_HSV_RANGES: list[tuple[np.ndarray, np.ndarray]] = [
    (np.array([25, 100, 140]), np.array([45, 255, 255])),  # hi-vis yellow-green
    (np.array([5, 140, 140]), np.array([20, 255, 255])),  # hi-vis orange
]

_MIN_COLOR_RATIO = 0.08


@dataclass
class RawDetection:
    violation_type: ViolationType
    confidence: float
    bbox: BoundingBox


class Detector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[RawDetection]: ...


def _color_ratio(region: np.ndarray, hsv_ranges: list[tuple[np.ndarray, np.ndarray]]) -> float:
    if region.size == 0:
        return 0.0
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in hsv_ranges:
        mask |= cv2.inRange(hsv, lower, upper)
    return float((mask > 0).mean())


class YoloPpeHeuristicDetector(Detector):
    def __init__(self, model_path: str = "yolov8n.pt", person_conf: float = 0.4):
        self._model = YOLO(model_path)
        self._person_conf = person_conf

    def detect(self, frame: np.ndarray) -> list[RawDetection]:
        results = self._model.predict(
            frame, classes=[_COCO_PERSON_CLASS_ID], conf=self._person_conf, verbose=False
        )
        detections: list[RawDetection] = []
        if not results:
            return detections

        for box in results[0].boxes:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            conf = float(box.conf[0])
            bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
            person_h = y2 - y1

            head_region = frame[int(y1) : int(y1 + 0.25 * person_h), int(x1) : int(x2)]
            torso_region = frame[int(y1 + 0.25 * person_h) : int(y1 + 0.75 * person_h), int(x1) : int(x2)]

            if _color_ratio(head_region, _HARD_HAT_HSV_RANGES) < _MIN_COLOR_RATIO:
                detections.append(RawDetection(ViolationType.NO_HARD_HAT, conf, bbox))

            if _color_ratio(torso_region, _VEST_HSV_RANGES) < _MIN_COLOR_RATIO:
                detections.append(RawDetection(ViolationType.NO_SAFETY_VEST, conf, bbox))

        return detections


def build_default_detector() -> Detector:
    return YoloPpeHeuristicDetector()
