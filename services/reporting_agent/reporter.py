"""LLM-backed incident report drafting."""

from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from services.common.schemas import IncidentReport


class _ReportDraft(BaseModel):
    """LLM output schema: content only. `generated_at` is set by our own code
    afterward - asking the model for it invites a hallucinated timestamp."""

    title: str
    body_markdown: str


async def draft_report(llm: ChatOpenAI, payload: dict[str, Any]) -> IncidentReport:
    structured_llm = llm.with_structured_output(_ReportDraft)

    events = payload.get("events", [])
    total_event_count = payload.get("total_event_count", len(events))
    severity = payload.get("severity", {})
    event_lines = [
        f"- site={e.get('site_id')} violation={e.get('violation_type')} "
        f"confidence={e.get('confidence', 0):.2f} at={e.get('timestamp')}"
        for e in events
    ]
    prompt = (
        "You are the Reporting Agent in a distributed construction-site PPE "
        "safety inspection network. Draft a concise incident report for a "
        "site safety manager, in markdown, based on the Orchestrator's "
        "cross-site pattern and the Triage Agent's severity classification.\n\n"
        f"Pattern summary: {payload.get('pattern_summary')}\n"
        f"Sites involved: {', '.join(payload.get('site_ids', []))}\n"
        f"Severity: {severity.get('level')} - {severity.get('rationale')}\n"
        f"Recommended action: {severity.get('recommended_action')}\n"
        f"Total matching events: {total_event_count} (showing a sample of {len(events)}):\n"
        + "\n".join(event_lines) + "\n\n"
        "Write a short title and a markdown body with sections for Summary, "
        "Evidence, and Recommended Action."
    )
    draft = await structured_llm.ainvoke(prompt)
    return IncidentReport(title=draft.title, body_markdown=draft.body_markdown)
