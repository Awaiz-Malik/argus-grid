"""Loads configs/sites.yaml into SiteConfig models. Shared by every Vision
Agent (to find its own config) and the Orchestrator (to discover all sites).
"""

from __future__ import annotations

import yaml

from services.common.schemas import SiteConfig


def load_sites(path: str) -> list[SiteConfig]:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return [SiteConfig.model_validate(entry) for entry in raw["sites"]]


def get_site(path: str, site_id: str) -> SiteConfig:
    for site in load_sites(path):
        if site.id == site_id:
            return site
    raise KeyError(f"Unknown site_id {site_id!r} in {path}")
