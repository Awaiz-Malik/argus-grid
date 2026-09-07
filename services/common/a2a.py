"""Shared A2A plumbing: building an agent's server routes, and calling another
agent's server as a client. Payloads are structured JSON carried in A2A "data"
parts (a2a.helpers.new_data_message / get_data_parts) rather than free text,
since every agent here exchanges structured events/results, not chat.
"""

from __future__ import annotations

from typing import Any

import httpx
from a2a.client import A2ACardResolver, ClientConfig, create_client
from a2a.helpers import get_data_parts, new_data_message
from a2a.server.agent_execution import AgentExecutor
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import add_a2a_routes_to_fastapi, create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    Role,
    SendMessageRequest,
    Task,
    TaskState,
)
from fastapi import FastAPI


class A2ATaskError(RuntimeError):
    """Raised when a delegated A2A task ends in a non-completed terminal state."""


def build_agent_card(
    *,
    name: str,
    description: str,
    url: str,
    skills: list[tuple[str, str, str]],
    version: str = "0.1.0",
) -> AgentCard:
    """skills: list of (id, name, description) tuples."""
    return AgentCard(
        name=name,
        description=description,
        version=version,
        default_input_modes=["application/json"],
        default_output_modes=["application/json"],
        capabilities=AgentCapabilities(streaming=False, extended_agent_card=False),
        supported_interfaces=[
            AgentInterface(protocol_binding="JSONRPC", url=url, protocol_version="1.0")
        ],
        skills=[
            AgentSkill(id=sid, name=sname, description=sdesc, tags=["argus-grid"])
            for sid, sname, sdesc in skills
        ],
    )


def attach_a2a_server(app: FastAPI, agent_card: AgentCard, executor: AgentExecutor) -> None:
    """Wires an in-memory task store + request handler for `executor` and mounts
    the resulting Agent Card + JSON-RPC routes onto `app`."""
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
    )
    add_a2a_routes_to_fastapi(
        app,
        agent_card_routes=create_agent_card_routes(agent_card),
        jsonrpc_routes=create_jsonrpc_routes(request_handler, "/"),
    )


async def call_agent(base_url: str, payload: dict[str, Any], *, timeout: float = 60.0) -> dict[str, Any]:
    """Resolves base_url's agent card, sends `payload` as a single data message,
    waits for the task to complete, and returns the first data artifact as a dict.
    """
    async with httpx.AsyncClient(timeout=timeout) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base_url)
        agent_card = await resolver.get_agent_card()

        client = await create_client(
            agent=agent_card, client_config=ClientConfig(streaming=False, httpx_client=httpx_client)
        )
        try:
            message = new_data_message(payload, role=Role.ROLE_USER)
            request = SendMessageRequest(message=message)

            final_task: Task | None = None
            async for chunk in client.send_message(request):
                if chunk.HasField("task"):
                    final_task = chunk.task

            if final_task is None:
                raise A2ATaskError(f"Agent at {base_url} returned no task result")
            if final_task.status.state != TaskState.TASK_STATE_COMPLETED:
                raise A2ATaskError(
                    f"Agent at {base_url} task ended in state "
                    f"{TaskState.Name(final_task.status.state)}"
                )

            for artifact in final_task.artifacts:
                data_parts = get_data_parts(artifact.parts)
                if data_parts:
                    return data_parts[0]

            raise A2ATaskError(f"Agent at {base_url} completed task with no data artifact")
        finally:
            await client.close()
