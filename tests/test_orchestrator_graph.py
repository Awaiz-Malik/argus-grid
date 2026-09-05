import os

os.environ.setdefault("OPENAI_API_KEY", "sk-placeholder-for-graph-construction")

from services.common.settings import get_settings  # noqa: E402
from services.orchestrator.graph import build_graph  # noqa: E402
from services.orchestrator.store import IncidentStore  # noqa: E402


def test_graph_compiles_with_expected_nodes(tmp_path):
    settings = get_settings()
    store = IncidentStore(str(tmp_path / "incidents.db"))
    graph = build_graph(settings, store, "http://localhost:8090", "http://localhost:8091")

    node_names = set(graph.get_graph().nodes.keys())
    assert {
        "discover_sites",
        "collect_detections",
        "analyze_cross_site_patterns",
        "delegate_triage",
        "delegate_reporting",
        "record_incident",
    } <= node_names
