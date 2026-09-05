"""LLM-backed incident report drafting."""

from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI

from services.common.schemas import IncidentReport


async def draft_report(llm: ChatOpenAI, payload: dict[str, Any]) -> IncidentReport:
    structured_llm = llm.with_structured_output(IncidentReport)

    events = payload.get("events", [])
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
        f"Underlying events ({len(events)}):\n" + "\n".join(event_lines) + "\n\n"
        "Write a short title and a markdown body with sections for Summary, "
        "Evidence, and Recommended Action."
    )
    return await structured_llm.ainvoke(prompt)
