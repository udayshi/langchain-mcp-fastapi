"""Check that the configured Ollama model returns text."""

from __future__ import annotations

import sys

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama


def has_text(content: object) -> bool:
    """Return whether a model response contains non-empty text."""
    return isinstance(content, str) and bool(content.strip())


def main() -> int:
    """Return zero only when local Ollama returns non-empty text."""
    try:
        response = ChatOllama(model="llama3.2:3b", temperature=0).invoke(
            [HumanMessage(content="Reply with: ready")]
        )
    except Exception as error:
        print(f"Ollama health check failed: {error}", file=sys.stderr)
        return 1
    if not has_text(response.content):
        print("Ollama health check failed: model returned no text", file=sys.stderr)
        return 1
    print("Ollama health check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
