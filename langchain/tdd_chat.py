"""Small, testable domain boundary for one chat-model request."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


class ChatModel(Protocol):
    """Minimal model interface required by the chat domain."""

    def invoke(self, input: Sequence[BaseMessage]) -> AIMessage:
        """Return a response for the supplied messages."""


def answer_question(question: str, model: ChatModel) -> str:
    """Return a text answer for one non-blank question."""
    if not question.strip():
        raise ValueError("question must not be blank")
    response = model.invoke([HumanMessage(content=question)])
    if not isinstance(response.content, str):
        raise ValueError("model must return text")
    return response.content
