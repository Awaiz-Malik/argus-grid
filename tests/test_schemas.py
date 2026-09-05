from datetime import datetime, timezone

from services.common.schemas import (
    BoundingBox,
    DetectionEvent,
    SiteConfig,
    ViolationType,
)


def test_site_config_discriminates_video_source():
    site = SiteConfig.model_validate(
        {
            "id": "site-a",
            "name": "Test Site",
            "url": "http://localhost:9001",
            "source": {"kind": "video", "path": "data/videos/site-a.mp4"},
        }
    )
    assert site.source.kind == "video"
    assert site.source.path == "data/videos/site-a.mp4"


def test_site_config_discriminates_webcam_source():
    site = SiteConfig.model_validate(
        {
            "id": "site-c",
            "name": "Test Site",
            "url": "http://localhost:9003",
            "source": {"kind": "webcam", "device_index": 0, "fallback_path": "data/videos/site-c.mp4"},
        }
    )
    assert site.source.kind == "webcam"
    assert site.source.device_index == 0


def test_detection_event_round_trips_through_json():
    event = DetectionEvent(
        site_id="site-a",
        timestamp=datetime.now(timezone.utc),
        violation_type=ViolationType.NO_HARD_HAT,
        confidence=0.87,
        bbox=BoundingBox(x1=1, y1=2, x2=3, y2=4),
    )
    restored = DetectionEvent.model_validate_json(event.model_dump_json())
    assert restored == event
