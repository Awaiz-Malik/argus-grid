import numpy as np

from services.vision_agent.detection import (
    _HARD_HAT_HSV_RANGES,
    _VEST_HSV_RANGES,
    _color_ratio,
)


def _solid_bgr(color_bgr: tuple[int, int, int], size: int = 40) -> np.ndarray:
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    frame[:, :] = color_bgr
    return frame


def test_color_ratio_detects_hard_hat_yellow():
    region = _solid_bgr((0, 215, 255))  # bright yellow, BGR
    assert _color_ratio(region, _HARD_HAT_HSV_RANGES) > 0.9


def test_color_ratio_rejects_skin_tone_for_hard_hat():
    region = _solid_bgr((120, 150, 200))  # a plain skin-tone-ish BGR patch
    assert _color_ratio(region, _HARD_HAT_HSV_RANGES) < 0.5


def test_color_ratio_detects_hivis_vest_orange():
    region = _solid_bgr((0, 140, 255))  # hi-vis orange, BGR
    assert _color_ratio(region, _VEST_HSV_RANGES) > 0.9


def test_color_ratio_empty_region_is_zero():
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    assert _color_ratio(empty, _HARD_HAT_HSV_RANGES) == 0.0
