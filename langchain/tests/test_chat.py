"""Unit tests for the TDD chat domain logic."""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


class ChatDomain(Protocol):
    """Interface exposed by the chat domain module under test."""

    def answer_question(self, question: str, model: "FakeModel") -> str:
        """Return a text answer for one question."""


def _load_chat_domain() -> ChatDomain:
    """Load the local domain module without shadowing the installed LangChain package."""
    module_path = Path(__file__).resolve().parents[1] / "tdd_chat.py"
    specification = importlib.util.spec_from_file_location("tdd_chat_under_test", module_path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Could not load chat domain module: {module_path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return cast(ChatDomain, module)


class FakeModel:
    """Controllable local substitute for a LangChain model."""

    def __init__(self, response: AIMessage) -> None:
        self.response = response
        self.calls: list[list[BaseMessage]] = []

    def invoke(self, input: Sequence[BaseMessage]) -> AIMessage:
        """Record input and return the configured response."""
        self.calls.append(list(input))
        return self.response


def test_answer_question_returns_text() -> None:
    """A valid model text response is returned unchanged."""
    model = FakeModel(AIMessage(content="A tuple is immutable."))

    assert _load_chat_domain().answer_question("What is a tuple?", model) == "A tuple is immutable."
    assert model.calls == [[HumanMessage(content="What is a tuple?")]]


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_answer_question_rejects_blank_input(question: str) -> None:
    """Blank questions fail before the model boundary is invoked."""
    with pytest.raises(ValueError, match="question must not be blank"):
        _load_chat_domain().answer_question(question, FakeModel(AIMessage(content="unused")))


def test_answer_question_rejects_non_text_response() -> None:
    """Structured or multimodal content is rejected by the text-only boundary."""
    model = FakeModel(AIMessage(content=[{"type": "text", "text": "not accepted"}]))

    with pytest.raises(ValueError, match="model must return text"):
        _load_chat_domain().answer_question("What is a tuple?", model)
