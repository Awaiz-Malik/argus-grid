"""Integration test: runs a real Triage Agent process and delegates a task to
it over real A2A (agent card resolution + JSON-RPC), the same way the
Orchestrator does. Uses a placeholder OPENAI_API_KEY so it's deterministic and
doesn't require a real key in CI - it verifies the full task lifecycle
(WORKING -> FAILED) and error surfacing, not the LLM's actual output.
"""

import os
import subprocess
import sys
import time

import pytest

from services.common.a2a import A2ATaskError, call_agent

PORT = 8099


@pytest.fixture(scope="module")
def triage_server():
    env = {
        **os.environ,
        "OPENAI_API_KEY": "sk-placeholder-for-plumbing-test",
        # Without this, the agent's own Agent Card advertises the *default*
        # triage_agent_port (8090) instead of the port it's actually bound to
        # here - the A2A client follows the card's declared URL for the real
        # message send, so a mismatch silently redirects requests elsewhere
        # (e.g. a real triage-agent instance already running on 8090).
        "ARGUS_SELF_URL": f"http://localhost:{PORT}",
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "services.triage_agent.app:app", "--port", str(PORT)],
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


async def test_task_lifecycle_reaches_failed_without_real_key(triage_server):
    payload = {
        "pattern_summary": "No safety vests observed across multiple sites this week",
        "site_ids": ["site-a", "site-b"],
        "events": [
            {"site_id": "site-a", "violation_type": "no_safety_vest", "confidence": 0.8},
        ],
    }
    with pytest.raises(A2ATaskError, match="TASK_STATE_FAILED"):
        await call_agent(f"http://localhost:{PORT}", payload)
