"""Triage Agent: a separate A2A service the Orchestrator delegates severity
classification to. Genuinely out-of-process - reached over JSON-RPC, not a
function call.

Run with: uvicorn services.triage_agent.app:app --port 8090
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
from services.triage_agent.classifier import classify_severity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()
configure_observability(settings)
llm = get_chat_model(settings, temperature=0.0)

SELF_URL = f"http://localhost:{settings.triage_agent_port}"
if os.environ.get("ARGUS_SELF_URL"):
    SELF_URL = os.environ["ARGUS_SELF_URL"]


class TriageExecutor(AgentExecutor):
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
            severity = await classify_severity(llm, payload)
        except Exception:
            logger.exception("Severity classification failed")
            await task_updater.update_status(state=TaskState.TASK_STATE_FAILED)
            return

        await task_updater.add_artifact(parts=[new_data_part(severity.model_dump(mode="json"))])
        await task_updater.update_status(state=TaskState.TASK_STATE_COMPLETED)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("Cancel is not supported.")


app = FastAPI(title="Argus Grid Triage Agent")

agent_card = build_agent_card(
    name="Triage Agent",
    description="Classifies the severity of a cross-site PPE-violation pattern flagged by the Orchestrator.",
    url=SELF_URL,
    skills=[
        (
            "classify_severity",
            "Classify Severity",
            "Given a flagged pattern and its underlying events, return a severity level, rationale, and recommended action.",
        )
    ],
)
attach_a2a_server(app, agent_card, TriageExecutor())


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
