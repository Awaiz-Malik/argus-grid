"""LLM + LangSmith wiring shared by the Orchestrator, Triage, and Reporting agents.

Every service calls configure_observability() once at startup and get_chat_model()
whenever it needs a model. LangSmith tracing activates automatically the moment
LANGCHAIN_API_KEY / LANGSMITH_API_KEY is set — no code changes needed.
"""

import os

from langchain_openai import ChatOpenAI

from services.common.settings import Settings


def configure_observability(settings: Settings) -> None:
    """Set the env vars LangChain/LangSmith read for tracing.

    Both LANGCHAIN_* (legacy) and LANGSMITH_* (current) names are set since
    different library versions/tools still look for either.
    """
    tracing_enabled = bool(settings.langchain_tracing_v2 and settings.langchain_api_key)
    for prefix in ("LANGCHAIN", "LANGSMITH"):
        os.environ[f"{prefix}_TRACING_V2"] = "true" if tracing_enabled else "false"
        os.environ[f"{prefix}_TRACING"] = "true" if tracing_enabled else "false"
        os.environ[f"{prefix}_PROJECT"] = settings.langchain_project
        if settings.langchain_api_key:
            os.environ[f"{prefix}_API_KEY"] = settings.langchain_api_key


def get_chat_model(settings: Settings, *, temperature: float = 0.2) -> ChatOpenAI:
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to .env before starting any agent "
            "that reasons with an LLM (Orchestrator, Triage, Reporting)."
        )
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        temperature=temperature,
    )
