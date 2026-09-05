"""LLM-backed severity classification for a flagged cross-site pattern."""

from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI

from services.common.schemas import SeverityResult


async def classify_severity(llm: ChatOpenAI, payload: dict[str, Any]) -> SeverityResult:
    structured_llm = llm.with_structured_output(SeverityResult)

    events = payload.get("events", [])
    event_lines = [
        f"- site={e.get('site_id')} violation={e.get('violation_type')} confidence={e.get('confidence', 0):.2f}"
        for e in events
    ]
    prompt = (
        "You are the Triage Agent in a distributed construction-site PPE safety "
        "inspection network. The Orchestrator has flagged a cross-site pattern "
        "and delegated it to you for severity classification.\n\n"
        f"Pattern summary: {payload.get('pattern_summary')}\n"
        f"Sites involved: {', '.join(payload.get('site_ids', []))}\n"
        f"Events ({len(events)}):\n" + "\n".join(event_lines) + "\n\n"
        "Classify the severity (low/medium/high/critical), explain why in one "
        "or two sentences, and recommend one concrete next action for a site "
        "safety manager."
    )
    return await structured_llm.ainvoke(prompt)
