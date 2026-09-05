"""Shared Pydantic models used across all Argus Grid services."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


class ViolationType(StrEnum):
    NO_HARD_HAT = "no_hard_hat"
    NO_SAFETY_VEST = "no_safety_vest"


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class DetectionEvent(BaseModel):
    """One PPE-violation observation from a single Vision Agent."""

    id: str = Field(default_factory=_new_id)
    site_id: str
    timestamp: datetime = Field(default_factory=_now)
    violation_type: ViolationType
    confidence: float
    bbox: BoundingBox
    snapshot_path: str | None = None


class VideoSource(BaseModel):
    kind: Literal["video"] = "video"
    path: str
    loop: bool = True


class WebcamSource(BaseModel):
    kind: Literal["webcam"] = "webcam"
    device_index: int = 0
    fallback_path: str | None = None


class SiteConfig(BaseModel):
    id: str
    name: str
    url: str
    source: Annotated[VideoSource | WebcamSource, Field(discriminator="kind")]


class AgentSkill(BaseModel):
    id: str
    name: str
    description: str


class AgentCard(BaseModel):
    """Minimal A2A Agent Card served at /.well-known/agent-card.json."""

    name: str
    description: str
    url: str
    version: str = "0.1.0"
    skills: list[AgentSkill] = Field(default_factory=list)
    capabilities: dict[str, bool] = Field(default_factory=dict)


class SeverityLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SeverityResult(BaseModel):
    level: SeverityLevel
    rationale: str
    recommended_action: str


class IncidentReport(BaseModel):
    title: str
    body_markdown: str
    generated_at: datetime = Field(default_factory=_now)


class IncidentStatus(StrEnum):
    OPEN = "open"
    TRIAGED = "triaged"
    REPORTED = "reported"


class Incident(BaseModel):
    id: str = Field(default_factory=_new_id)
    created_at: datetime = Field(default_factory=_now)
    site_ids: list[str]
    pattern_summary: str
    event_ids: list[str]
    status: IncidentStatus = IncidentStatus.OPEN
    severity: SeverityResult | None = None
    report: IncidentReport | None = None
