"""Return a validated Pydantic object from a local model."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field


class SupportAnswer(BaseModel):
    """Validated result for a support workflow."""

    answer: str = Field(description="Direct answer to the user")
    confidence: float = Field(ge=0, le=1, description="Model-estimated confidence")
    needs_human: bool = Field(description="Whether a human should review the request")


def main() -> None:
    """Request and print structured data."""
    model = ChatOllama(model="gemma3:4b", temperature=0)
    result = model.with_structured_output(SupportAnswer).invoke(
        [
            SystemMessage(content="Answer with a quick, self-contained response."),
            HumanMessage(content="Can I change my delivery address after dispatch?"),
        ]
    )
    if not isinstance(result, SupportAnswer):
        raise TypeError("Model did not return the requested SupportAnswer schema.")
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
