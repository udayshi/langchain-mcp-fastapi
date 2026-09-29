"""Pass one Ollama model's draft to a second Ollama model for review."""

from __future__ import annotations

import os
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama


@dataclass(frozen=True)
class PipelineResult:
    """Text produced by the drafting and reviewing stages."""

    draft: str
    review: str


def invoke_text(model: ChatOllama, messages: list[SystemMessage | HumanMessage]) -> str:
    """Invoke a model and return its text response."""
    response = model.invoke(messages)
    if not isinstance(response.content, str):
        raise ValueError("Expected the model to return text.")
    return response.content


def run_pipeline(topic: str, writer_name: str, reviewer_name: str) -> PipelineResult:
    """Draft a short explanation, then provide that draft to a reviewer model."""
    writer = ChatOllama(model=writer_name, temperature=0)
    reviewer = ChatOllama(model=reviewer_name, temperature=0)
    draft = invoke_text(
        writer,
        [
            SystemMessage(content="You write accurate, concise technical explanations."),
            HumanMessage(content=f"Explain {topic} in three short bullet points."),
        ],
    )
    review = invoke_text(
        reviewer,
        [
            SystemMessage(
                content="You are a careful technical editor. Correct inaccuracies and return a clearer final answer."
            ),
            HumanMessage(content=f"Topic: {topic}\n\nDraft to review:\n---\n{draft}\n---"),
        ],
    )
    return PipelineResult(draft=draft, review=review)


def main() -> None:
    """Run the pipeline using configurable local Ollama models."""
    writer_name = os.getenv("WRITER_MODEL", "llama3.2:3b")
    reviewer_name = os.getenv("REVIEWER_MODEL", "gemma3:4b")
    result = run_pipeline("the benefits of Python type hints", writer_name, reviewer_name)

    print("DRAFT\n-----")
    print(result.draft)
    print("\nREVIEWED VERSION\n----------------")
    print(result.review)


if __name__ == "__main__":
    main()