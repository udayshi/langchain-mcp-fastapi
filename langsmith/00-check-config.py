"""Verify the environment required to upload LangSmith traces."""

from __future__ import annotations

import os

from langsmith import utils


def _environment_value(name: str) -> str | None:
    """Return a configured LangSmith or legacy LangChain environment value."""
    return os.getenv(f"LANGSMITH_{name}") or os.getenv(f"LANGCHAIN_{name}")


def main() -> None:
    """Fail clearly when tracing cannot be enabled by the current environment."""
    api_key = _environment_value("API_KEY")
    project = _environment_value("PROJECT") or "default"
    endpoint = _environment_value("ENDPOINT") or "https://api.smith.langchain.com"

    if not utils.tracing_is_enabled():
        raise RuntimeError(
            "Tracing is disabled. Set LANGSMITH_TRACING_V2=true before running the trace example."
        )
    if not api_key:
        raise RuntimeError("LANGSMITH_API_KEY is not set.")
    print(f"Tracing enabled: true\nProject: {project}\nEndpoint: {endpoint}\nAPI key: configured")


if __name__ == "__main__":
    main()
