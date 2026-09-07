"""The Orchestrator's LangGraph: pull detections from every reachable site,
reason about whether they form a cross-site pattern worth escalating, and if
so delegate to the Triage and Reporting agents over real A2A tasks.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from services.common.a2a import call_agent
from services.common.llm import get_chat_model
from services.common.schemas import (
    DetectionEvent,
    Incident,
    IncidentReport,
    IncidentStatus,
    SeverityResult,
    SiteConfig,
)
from services.common.settings import Settings
from services.orchestrator import mcp_client
from services.orchestrator.discovery import discover_sites
from services.orchestrator.store import IncidentStore

logger = logging.getLogger(__name__)

LOOKBACK = timedelta(hours=24)
MAX_DELEGATED_EVENTS = 30


class PatternAnalysis(BaseModel):
    incident_worthy: bool = Field(
        description="Whether these events represent a genuine cross-site pattern worth escalating"
    )
    site_ids: list[str] = Field(default_factory=list, description="Sites involved in the flagged pattern")
    summary: str = Field(description="One or two sentence description of the pattern")


class OrchestratorState(TypedDict, total=False):
    sites: list[SiteConfig]
    events: list[DetectionEvent]
    analysis: PatternAnalysis | None
    severity: SeverityResult | None
    report: IncidentReport | None
    incident: Incident | None


def build_graph(settings: Settings, incident_store: IncidentStore, triage_url: str, reporting_url: str):
    llm = get_chat_model(settings, temperature=0.1)
    structured_llm = llm.with_structured_output(PatternAnalysis)

    async def discover_sites_node(state: OrchestratorState) -> OrchestratorState:
        sites = await discover_sites(settings)
        return {"sites": sites}

    async def collect_detections_node(state: OrchestratorState) -> OrchestratorState:
        since = datetime.now(timezone.utc) - LOOKBACK

        async def fetch(site: SiteConfig) -> list[DetectionEvent]:
            try:
                result = await mcp_client.call_tool(
                    site.url, "get_event_history", {"since_iso": since.isoformat(), "limit": 200}
                )
                return [DetectionEvent.model_validate(e) for e in result.get("events", [])]
            except Exception:
                logger.warning("Failed to pull events from %s", site.id, exc_info=True)
                return []

        results = await asyncio.gather(*(fetch(site) for site in state.get("sites", [])))
        events = [event for site_events in results for event in site_events]
        return {"events": events}

    async def analyze_cross_site_patterns_node(state: OrchestratorState) -> OrchestratorState:
        events = state.get("events", [])
        if not events:
            return {
                "analysis": PatternAnalysis(incident_worthy=False, site_ids=[], summary="No events this cycle.")
            }

        summary_lines = [
            f"- site={e.site_id} violation={e.violation_type.value} "
            f"confidence={e.confidence:.2f} at={e.timestamp.isoformat()}"
            for e in events
        ]
        prompt = (
            "You are the cross-site reasoning layer of a distributed PPE-safety "
            "inspection network. Below are PPE-violation events pulled from every "
            "site in the last 24 hours. A single site having some violations is "
            "normal background noise. Flag it as incident-worthy ONLY if you see "
            "a genuine cross-site pattern: the same violation type recurring "
            "across 2+ sites, or a clear spike/concentration at one site that "
            "looks like a systemic issue rather than one-off noise.\n\n" + "\n".join(summary_lines)
        )
        analysis = await structured_llm.ainvoke(prompt)
        return {"analysis": analysis}

    def route_after_analysis(state: OrchestratorState) -> str:
        analysis = state.get("analysis")
        return "delegate_triage" if analysis and analysis.incident_worthy else "end"

    def _delegated_events_payload(state: OrchestratorState, site_ids: list[str]) -> dict:
        events = [e for e in state.get("events", []) if e.site_id in site_ids]
        # A flagged pattern can involve hundreds of events; delegated agents only
        # need a representative sample to reason over, not every single one -
        # sending them all bloats the prompt and can make the LLM call slow
        # enough to trip the A2A client timeout.
        sample = events[:MAX_DELEGATED_EVENTS]
        return {"total_event_count": len(events), "events": [e.model_dump(mode="json") for e in sample]}

    async def delegate_triage_node(state: OrchestratorState) -> OrchestratorState:
        analysis = state["analysis"]
        payload = {
            "pattern_summary": analysis.summary,
            "site_ids": analysis.site_ids,
            **_delegated_events_payload(state, analysis.site_ids),
        }
        result = await call_agent(triage_url, payload)
        return {"severity": SeverityResult.model_validate(result)}

    async def delegate_reporting_node(state: OrchestratorState) -> OrchestratorState:
        analysis = state["analysis"]
        payload = {
            "pattern_summary": analysis.summary,
            "site_ids": analysis.site_ids,
            "severity": state["severity"].model_dump(mode="json"),
            **_delegated_events_payload(state, analysis.site_ids),
        }
        result = await call_agent(reporting_url, payload)
        return {"report": IncidentReport.model_validate(result)}

    def record_incident_node(state: OrchestratorState) -> OrchestratorState:
        analysis = state["analysis"]
        incident = Incident(
            site_ids=analysis.site_ids,
            pattern_summary=analysis.summary,
            event_ids=[e.id for e in state.get("events", []) if e.site_id in analysis.site_ids],
            status=IncidentStatus.REPORTED,
            severity=state["severity"],
            report=state["report"],
        )
        incident_store.add(incident)
        logger.info("Recorded incident %s (severity=%s)", incident.id, incident.severity.level)
        return {"incident": incident}

    graph = StateGraph(OrchestratorState)
    graph.add_node("discover_sites", discover_sites_node)
    graph.add_node("collect_detections", collect_detections_node)
    graph.add_node("analyze_cross_site_patterns", analyze_cross_site_patterns_node)
    graph.add_node("delegate_triage", delegate_triage_node)
    graph.add_node("delegate_reporting", delegate_reporting_node)
    graph.add_node("record_incident", record_incident_node)

    graph.add_edge(START, "discover_sites")
    graph.add_edge("discover_sites", "collect_detections")
    graph.add_edge("collect_detections", "analyze_cross_site_patterns")
    graph.add_conditional_edges(
        "analyze_cross_site_patterns",
        route_after_analysis,
        {"delegate_triage": "delegate_triage", "end": END},
    )
    graph.add_edge("delegate_triage", "delegate_reporting")
    graph.add_edge("delegate_reporting", "record_incident")
    graph.add_edge("record_incident", END)

    return graph.compile()
