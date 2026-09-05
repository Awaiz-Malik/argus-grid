"""Unified frame source: loops a local video file, or reads a webcam with a
video-file fallback if the webcam can't be opened (e.g. no camera attached,
or running in a container without device passthrough).
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from services.common.schemas import VideoSource, WebcamSource

logger = logging.getLogger(__name__)


class FrameSource:
    def __init__(self, source: VideoSource | WebcamSource):
        self._source = source
        self._loop = isinstance(source, VideoSource) and source.loop
        self._cap = self._open()

    def _open(self) -> cv2.VideoCapture:
        if isinstance(self._source, WebcamSource):
            cap = cv2.VideoCapture(self._source.device_index)
            if cap.isOpened():
                return cap
            logger.warning(
                "Webcam device %s unavailable, falling back to %s",
                self._source.device_index,
                self._source.fallback_path,
            )
            if self._source.fallback_path:
                self._loop = True
                return cv2.VideoCapture(self._source.fallback_path)
            raise RuntimeError(f"Webcam device {self._source.device_index} unavailable and no fallback configured")

        cap = cv2.VideoCapture(self._source.path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source at {self._source.path}")
        return cap

    def read(self) -> np.ndarray | None:
        """Returns the next frame, or None if the source is exhausted and not looping."""
        ok, frame = self._cap.read()
        if ok:
            return frame
        if self._loop:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
            return frame if ok else None
        return None

    def close(self) -> None:
        self._cap.release()
