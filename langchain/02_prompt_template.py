"""Render a reusable chat prompt."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


def main() -> None:
    """Render a prompt with data and print its answer."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Explain {topic} for {audience} in at most two sentences."),
            ("human", "Question: {question}"),
        ]
    )
    messages = prompt.invoke(
        {
            "topic": "dependency injection",
            "audience": "Python beginners",
            "question": "Why is it useful in tests?",
        }
    )
    response = ChatOllama(model="llama3.2:3b", temperature=0).invoke(messages)
    if not isinstance(response.content, str):
        raise ValueError("Expected the model to return text.")
    print(response.content)


if __name__ == "__main__":
    main()
