"""Maintain a small in-memory conversation history."""

from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_ollama import ChatOllama


def text(message: AIMessage) -> str:
    """Return text content or reject non-text model output."""
    if not isinstance(message.content, str):
        raise ValueError("Expected the model to return text.")
    return message.content


def main() -> None:
    """Run two turns that share message history."""
    model = ChatOllama(model="llama3.2:3b", temperature=0)
    history: list[BaseMessage] = [HumanMessage(content="My name is Uday Shiwakoti.")]
    history.append(model.invoke(history))
    history.append(HumanMessage(content="What is my name? Reply with only the name."))
    print(text(model.invoke(history)))


if __name__ == "__main__":
    main()
