"""Reporting Agent: a separate A2A service the Orchestrator delegates
incident-report drafting to, after Triage has classified severity.

Run with: uvicorn services.reporting_agent.app:app --port 8091
"""

from __future__ import annotations

import logging
import os

from a2a.helpers import get_data_parts, new_data_part, new_task_from_user_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import TaskState
from fastapi import FastAPI

from services.common.a2a import attach_a2a_server, build_agent_card
from services.common.llm import configure_observability, get_chat_model
from services.common.settings import get_settings
from services.reporting_agent.reporter import draft_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()
configure_observability(settings)
llm = get_chat_model(settings, temperature=0.3)

SELF_URL = f"http://localhost:{settings.reporting_agent_port}"
if os.environ.get("ARGUS_SELF_URL"):
    SELF_URL = os.environ["ARGUS_SELF_URL"]


class ReportingExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.current_task
        if task is None:
            task = new_task_from_user_message(context.message)
            await event_queue.enqueue_event(task)

        task_updater = TaskUpdater(event_queue=event_queue, task_id=task.id, context_id=task.context_id)
        await task_updater.update_status(state=TaskState.TASK_STATE_WORKING)

        data_parts = get_data_parts(context.message.parts)
        payload = data_parts[0] if data_parts else {}

        try:
            report = await draft_report(llm, payload)
        except Exception:
            logger.exception("Report drafting failed")
            await task_updater.update_status(state=TaskState.TASK_STATE_FAILED)
            return

        await task_updater.add_artifact(parts=[new_data_part(report.model_dump(mode="json"))])
        await task_updater.update_status(state=TaskState.TASK_STATE_COMPLETED)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("Cancel is not supported.")


app = FastAPI(title="Argus Grid Reporting Agent")

agent_card = build_agent_card(
    name="Reporting Agent",
    description="Drafts a markdown incident report for a triaged cross-site PPE-violation pattern.",
    url=SELF_URL,
    skills=[
        (
            "draft_incident_report",
            "Draft Incident Report",
            "Given a flagged pattern, its events, and a severity classification, draft a markdown incident report.",
        )
    ],
)
attach_a2a_server(app, agent_card, ReportingExecutor())


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
