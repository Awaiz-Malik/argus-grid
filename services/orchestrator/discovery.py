"""Site discovery: confirms each configured Vision Agent is actually up by
fetching its A2A Agent Card before the Orchestrator tries to pull detections
from it over MCP.
"""

from __future__ import annotations

import logging

import httpx
from a2a.client import A2ACardResolver

from services.common.schemas import SiteConfig
from services.common.settings import Settings
from services.common.site_registry import load_sites

logger = logging.getLogger(__name__)


async def discover_sites(settings: Settings) -> list[SiteConfig]:
    """Returns the subset of configs/sites.yaml's sites that are currently reachable."""
    sites = load_sites(settings.sites_config_path)
    reachable: list[SiteConfig] = []

    async with httpx.AsyncClient(timeout=10.0) as client:
        for site in sites:
            try:
                resolver = A2ACardResolver(httpx_client=client, base_url=site.url)
                card = await resolver.get_agent_card()
                logger.info("Discovered site %s: %s", site.id, card.name)
                reachable.append(site)
            except Exception:
                logger.warning("Site %s (%s) unreachable this cycle", site.id, site.url, exc_info=True)

    return reachable
