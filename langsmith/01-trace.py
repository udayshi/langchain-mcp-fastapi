"""Create one simple LangSmith trace."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langsmith import Client, traceable


CLIENT = Client()


@traceable(name="ollama-greeting", run_type="chain", client=CLIENT)
def answer_greeting(name: str) -> str:
    """Ask local Ollama to create a short greeting."""
    cleaned_name = name.strip()
    if not cleaned_name:
        raise ValueError("name must not be blank")
    response = ChatOllama(model="llama3.2:3b", temperature=0).invoke(
        [
            SystemMessage(content="Return a warm greeting in one short sentence."),
            HumanMessage(content=f"Greet {cleaned_name}."),
        ]
    )
    if not isinstance(response.content, str) or not response.content.strip():
        raise ValueError("model must return non-empty text")
    return response.content


def main() -> None:
    """Run the traced function."""
    try:
        print(answer_greeting("Uday Shiwakoti"))
    finally:
        # Finish the SDK's background upload before this short script exits.
        CLIENT.flush()


if __name__ == "__main__":
    main()
