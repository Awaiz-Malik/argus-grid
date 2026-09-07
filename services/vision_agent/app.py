"""One site's Vision Agent: a background capture/detection loop feeding a
local event store, exposed to the outside world as:

  * an A2A Agent Card at /.well-known/agent-card.json (discovery only - this
    agent does not accept A2A tasks, it's not delegated to)
  * a stateless-HTTP MCP server at /mcp exposing detect_anomaly and
    get_event_history (the actual tool-calling interface the Orchestrator uses)

Run with: ARGUS_SITE_ID=site-a uvicorn services.vision_agent.app:app --port 9001
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
import time
from datetime import datetime

import numpy as np
from a2a.server.routes import add_a2a_routes_to_fastapi, create_agent_card_routes
from fastapi import FastAPI
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings

from services.common.a2a import build_agent_card
from services.common.schemas import DetectionEvent
from services.common.settings import get_settings
from services.common.site_registry import get_site
from services.vision_agent.capture import FrameSource
from services.vision_agent.detection import build_default_detector
from services.vision_agent.store import EventStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()
SITE_ID = os.environ["ARGUS_SITE_ID"]
site = get_site(settings.sites_config_path, SITE_ID)

os.makedirs(f"{settings.data_dir}/events", exist_ok=True)
os.makedirs(f"{settings.data_dir}/videos", exist_ok=True)

store = EventStore(f"{settings.data_dir}/events/{SITE_ID}.db")
detector = build_default_detector()

_latest_frame: np.ndarray | None = None
_latest_frame_lock = threading.Lock()
_capture_stop = threading.Event()


def _capture_loop() -> None:
    global _latest_frame
    frame_source = FrameSource(site.source)
    logger.info("Site %s: capture loop started (%s)", SITE_ID, site.source)
    try:
        while not _capture_stop.is_set():
            frame = frame_source.read()
            if frame is None:
                time.sleep(1.0)
                continue

            with _latest_frame_lock:
                _latest_frame = frame

            for raw in detector.detect(frame):
                store.add(
                    DetectionEvent(
                        site_id=SITE_ID,
                        violation_type=raw.violation_type,
                        confidence=raw.confidence,
                        bbox=raw.bbox,
                    )
                )
            # Paced well below video frame rate - this is a monitoring loop,
            # not a video pipeline; a real deployment would tune this per site.
            time.sleep(1.0)
    finally:
        frame_source.close()
        logger.info("Site %s: capture loop stopped", SITE_ID)


mcp = MCPServer(f"vision-agent-{SITE_ID}")


@mcp.tool()
def detect_anomaly() -> dict:
    """Run PPE-violation detection on this site's current frame and return any violations found right now."""
    with _latest_frame_lock:
        frame = None if _latest_frame is None else _latest_frame.copy()

    if frame is None:
        return {"site_id": SITE_ID, "violations": []}

    detections = detector.detect(frame)
    return {
        "site_id": SITE_ID,
        "violations": [
            {
                "violation_type": d.violation_type.value,
                "confidence": d.confidence,
                "bbox": d.bbox.model_dump(),
            }
            for d in detections
        ],
    }


@mcp.tool()
def get_event_history(since_iso: str | None = None, limit: int = 100) -> dict:
    """Return recent PPE-violation events recorded at this site.

    since_iso: optional ISO-8601 timestamp; only events at/after it are returned.
    limit: maximum number of events to return (most recent first).
    """
    since = datetime.fromisoformat(since_iso) if since_iso else None
    events = store.recent(since=since, limit=limit)
    return {"site_id": SITE_ID, "events": [e.model_dump(mode="json") for e in events]}


@contextlib.asynccontextmanager
async def lifespan(_: FastAPI):
    capture_thread = threading.Thread(target=_capture_loop, daemon=True)
    async with mcp.session_manager.run():
        capture_thread.start()
        yield
        _capture_stop.set()


app = FastAPI(title=f"Argus Grid Vision Agent - {site.name}", lifespan=lifespan)

agent_card = build_agent_card(
    name=f"Vision Agent - {site.name}",
    description=(
        "Watches one construction site for PPE (hard hat / safety vest) "
        "violations. Discovery only via this Agent Card - actual detections "
        "are pulled over MCP (detect_anomaly, get_event_history), not A2A tasks."
    ),
    url=site.url,
    skills=[
        (
            "detect_anomaly",
            "Detect Anomaly",
            "Run PPE-violation detection on the current frame and return any violations found.",
        ),
        (
            "get_event_history",
            "Get Event History",
            "Return recent PPE-violation events recorded at this site.",
        ),
    ],
)
add_a2a_routes_to_fastapi(app, agent_card_routes=create_agent_card_routes(agent_card))


@app.get("/health")
def health() -> dict:
    return {"site_id": SITE_ID, "status": "ok"}


# The MCP SDK's default DNS-rebinding protection only allows localhost
# Host/Origin headers - correct for a server exposed to untrusted browsers,
# but it silently rejects every request when the Orchestrator reaches this
# agent by its podman-compose service name (e.g. http://vision-agent-site-a:9001).
# All traffic here stays on our own internal compose network, so it's disabled.
app.mount(
    "/",
    mcp.streamable_http_app(
        stateless_http=True,
        json_response=True,
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    ),
)
