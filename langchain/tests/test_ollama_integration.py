"""Opt-in integration test for the local Ollama model boundary."""

from __future__ import annotations

import os

import pytest
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_OLLAMA_INTEGRATION") != "1",
    reason="Set RUN_OLLAMA_INTEGRATION=1 to run local Ollama integration tests.",
)


def test_ollama_returns_non_empty_text() -> None:
    """Verify that the configured local Ollama model returns text."""
    response = ChatOllama(model="llama3.2:3b", temperature=0).invoke(
        [HumanMessage(content="Reply with the single word: ready")]
    )

    assert isinstance(response.content, str)
    assert response.content.strip()
