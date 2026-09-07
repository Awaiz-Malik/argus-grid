"""Integration test: runs a real Vision Agent process and calls its MCP tools
over real stateless streamable HTTP, the same way the Orchestrator does.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

VIDEO_PATH = Path("data/videos/site-a.mp4")
PORT = 9091

pytestmark = pytest.mark.skipif(
    not VIDEO_PATH.exists(), reason="run `python -m scripts.generate_sample_videos` first"
)


@pytest.fixture(scope="module")
def vision_agent_server():
    env = {**os.environ, "ARGUS_SITE_ID": "site-a"}
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "services.vision_agent.app:app", "--port", str(PORT)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(4)
    yield
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


async def test_list_tools_and_call_get_event_history(vision_agent_server):
    async with streamable_http_client(f"http://localhost:{PORT}/mcp") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            names = {tool.name for tool in tools.tools}
            assert {"detect_anomaly", "get_event_history"} <= names

            result = await session.call_tool("get_event_history", {"limit": 5})
            assert not result.is_error


async def test_detect_anomaly_runs(vision_agent_server):
    async with streamable_http_client(f"http://localhost:{PORT}/mcp") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("detect_anomaly", {})
            assert not result.is_error
