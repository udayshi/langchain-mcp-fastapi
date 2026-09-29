"""Create and stream a simple LCEL chain."""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


def main() -> None:
    """Stream a one-sentence summary."""
    prompt = ChatPromptTemplate.from_template("Summarise this text in one sentence:\n\n{text}")
    model = ChatOllama(model="llama3.2:3b", temperature=0)
    chain = prompt | model | StrOutputParser()
    for chunk in chain.stream({"text": "LangChain standardises LLM application components."}):
        print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    main()