"""Central, non-secret configuration for every Argus Grid service.

Secrets (API keys) come from the environment / .env only — see .env.example.
Everything else defaults here and can still be overridden by env vars of the
same (upper-cased) name if a deployment ever needs to.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- secrets ---
    openai_api_key: str | None = None
    langchain_api_key: str | None = None

    # --- LLM ---
    llm_model: str = "gpt-4o-mini"
    langchain_tracing_v2: bool = True
    langchain_project: str = "argus-grid"

    # --- service ports (local/dev) ---
    vision_agent_site_a_port: int = 9001
    vision_agent_site_b_port: int = 9002
    vision_agent_site_c_port: int = 9003
    orchestrator_port: int = 8080
    triage_agent_port: int = 8090
    reporting_agent_port: int = 8091

    # --- service URLs the Orchestrator delegates to ---
    # Defaults assume local processes; podman-compose overrides these to the
    # service DNS names (e.g. http://triage-agent:8090) via env vars.
    triage_agent_url: str = "http://localhost:8090"
    reporting_agent_url: str = "http://localhost:8091"

    # --- orchestrator behavior ---
    orchestrator_poll_interval_seconds: int = 60

    # --- paths ---
    sites_config_path: str = "configs/sites.yaml"
    data_dir: str = "data"


@lru_cache
def get_settings() -> Settings:
    return Settings()
