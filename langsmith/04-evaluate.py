"""Evaluate a local Ollama answer function against a LangSmith dataset."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langsmith.evaluation import evaluate


DATASET_NAME = "tuple-answers"


def target(inputs: dict[str, str]) -> dict[str, str]:
    """Ask local Ollama for the candidate answer to one dataset input."""
    response = ChatOllama(model="llama3.2:3b", temperature=0).invoke(
        [
            SystemMessage(content="Answer in one sentence and include immutable when relevant."),
            HumanMessage(content=inputs["question"]),
        ]
    )
    if not isinstance(response.content, str) or not response.content.strip():
        raise ValueError("model must return non-empty text")
    return {"answer": response.content}


def contains_required_term(
    inputs: dict[str, str], outputs: dict[str, str], reference_outputs: dict[str, str]
) -> dict[str, float | str]:
    """Score whether the answer contains the required reference term."""
    del inputs
    required_term = reference_outputs["required_term"].casefold()
    score = float(required_term in outputs["answer"].casefold())
    return {"key": "required_term", "score": score}


def main() -> None:
    """Create an evaluation experiment from the named dataset."""
    evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[contains_required_term],
        experiment_prefix="tuple-baseline",
    )


if __name__ == "__main__":
    main()
