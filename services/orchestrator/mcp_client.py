"""Calls a Vision Agent's MCP tools (detect_anomaly, get_event_history) over
stateless streamable HTTP - the Orchestrator's "pull detections" interface.
"""

from __future__ import annotations

import json
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


class MCPToolError(RuntimeError):
    pass


async def call_tool(base_url: str, tool_name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    mcp_url = f"{base_url.rstrip('/')}/mcp"
    async with streamable_http_client(mcp_url) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments or {})

            if result.is_error:
                message = next((c.text for c in result.content if hasattr(c, "text")), "unknown error")
                raise MCPToolError(f"{tool_name} at {base_url} failed: {message}")

            if result.structured_content:
                return result.structured_content

            for content in result.content:
                if hasattr(content, "text"):
                    return json.loads(content.text)

            return {}
