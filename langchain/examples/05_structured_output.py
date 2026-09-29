"""Return a validated Pydantic object from a local model."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field


class SupportAnswer(BaseModel):
    """Validated result for a support workflow."""

    answer: str = Field(description="Direct answer to the user")
    confidence: float = Field(ge=0, le=1, description="Model-estimated confidence")
    needs_human: bool = Field(description="Whether a human should review the request or not")


def main() -> None:
    """Request and print structured data."""
    model = ChatOllama(model="gemma3:4b", temperature=5)
    user_inputs=["I want to update my profile name.","I want admin rights to my account whom I need to contact."]
    for user_input in user_inputs:
        result = model.with_structured_output(SupportAnswer).invoke(
            [
                SystemMessage(
                    content=(
                        "Answer with a quick, self-contained response. Do not mention, cite, link to, "
                        "or recommend third-party sources."
                    )
                ),
                HumanMessage(content=user_input),
            ]
        )
        print(user_input)
        print(result.model_dump_json(indent=2))
        print()

if __name__ == "__main__":
    main()