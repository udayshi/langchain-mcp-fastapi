"""Trace a local Ollama LangChain call with LangSmith."""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


def main() -> None:
    """Invoke a traced LangChain chain."""
    prompt = ChatPromptTemplate.from_template("Answer in one sentence: {question}")
    model = ChatOllama(model="llama3.2:3b", temperature=0)
    chain = prompt | model | StrOutputParser()
    print(chain.invoke({"question": "What is a Python tuple?"}))


if __name__ == "__main__":
    main()