"""The Orchestrator: runs the cross-site reasoning graph on a schedule (and
on demand via /trigger), and serves a small dashboard over what it found.

Run with: uvicorn services.orchestrator.app:app --port 8080
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from services.common.llm import configure_observability
from services.common.settings import get_settings
from services.orchestrator.graph import build_graph
from services.orchestrator.store import IncidentStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()
configure_observability(settings)

os.makedirs(f"{settings.data_dir}/events", exist_ok=True)

incident_store = IncidentStore(f"{settings.data_dir}/events/orchestrator_incidents.db")
graph = build_graph(settings, incident_store, settings.triage_agent_url, settings.reporting_agent_url)

templates = Jinja2Templates(directory="services/orchestrator/templates")


async def run_cycle() -> dict:
    logger.info("Running orchestrator cycle")
    final_state = await graph.ainvoke({})
    incident = final_state.get("incident")
    return {
        "sites_checked": [s.id for s in final_state.get("sites", [])],
        "events_collected": len(final_state.get("events", [])),
        "incident_created": incident.id if incident else None,
    }


async def _scheduler_loop() -> None:
    while True:
        try:
            await run_cycle()
        except Exception:
            logger.exception("Orchestrator cycle failed")
        await asyncio.sleep(settings.orchestrator_poll_interval_seconds)


@contextlib.asynccontextmanager
async def lifespan(_: FastAPI):
    task = asyncio.create_task(_scheduler_loop())
    yield
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


app = FastAPI(title="Argus Grid Orchestrator", lifespan=lifespan)


@app.post("/trigger")
async def trigger() -> dict:
    """Runs one orchestrator cycle right now, instead of waiting for the schedule."""
    return await run_cycle()


@app.get("/status")
def status() -> dict:
    return {"incidents": [i.model_dump(mode="json") for i in incident_store.recent(limit=20)]}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request) -> HTMLResponse:
    incidents = incident_store.recent(limit=20)
    return templates.TemplateResponse("dashboard.html", {"request": request, "incidents": incidents})
