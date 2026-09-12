"""In-memory snapshot of the most recent orchestrator cycle, for the
dashboard to render without re-running discovery/MCP calls on every page
view. Updated once per cycle in app.py's run_cycle().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from services.common.schemas import DetectionEvent, SiteConfig

MAX_RECENT_EVENTS = 30


@dataclass
class SiteStatus:
    id: str
    name: str
    url: str
    reachable: bool


@dataclass
class LiveState:
    last_run_at: datetime | None = None
    sites: list[SiteStatus] = field(default_factory=list)
    recent_events: list[DetectionEvent] = field(default_factory=list)


_state = LiveState()


def get_state() -> LiveState:
    return _state


def update(
    *,
    configured_sites: list[SiteConfig],
    reachable_sites: list[SiteConfig],
    events: list[DetectionEvent],
) -> None:
    reachable_ids = {s.id for s in reachable_sites}
    _state.sites = [
        SiteStatus(id=s.id, name=s.name, url=s.url, reachable=s.id in reachable_ids) for s in configured_sites
    ]
    _state.recent_events = sorted(events, key=lambda e: e.timestamp, reverse=True)[:MAX_RECENT_EVENTS]
    _state.last_run_at = datetime.now().astimezone()
