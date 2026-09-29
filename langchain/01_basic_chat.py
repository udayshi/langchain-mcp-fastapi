"""Send one question to local Ollama."""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama


def response_text(content: object) -> str:
    """Return text model content or reject unsupported responses."""
    if not isinstance(content, str):
        raise ValueError("Expected the model to return text.")
    return content


def main() -> None:
    """Ask one question and print text output."""
    model = ChatOllama(model="llama3.2:3b", temperature=0)
    response = model.invoke([HumanMessage(content="Explain a Python tuple in one sentence.")])
    print(response_text(response.content))


if __name__ == "__main__":
    main()
