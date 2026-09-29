"""Index a Markdown document locally and answer a grounded question."""

from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def format_documents(documents: list[Document]) -> str:
    """Format documents with sources for a prompt."""
    return "\n\n".join(f"Source: {item.metadata['source']}\n{item.page_content}" for item in documents)


def main() -> None:
    """Create a local index and ask it one question."""
    source_path = Path("rag_source/handbook.md")
    if not source_path.is_file():
        raise FileNotFoundError(f"Create the source file first: {source_path}")
    document = Document(source_path.read_text(encoding="utf-8"), metadata={"source": str(source_path)})
    chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120).split_documents([document])
    store = Chroma.from_documents(chunks, OllamaEmbeddings(model="nomic-embed-text"), persist_directory=".chroma")
    question = "How many days of annual leave are available?"
    context = format_documents(store.as_retriever(search_kwargs={"k": 4}).invoke(question))
    prompt = ChatPromptTemplate.from_template(
        "Answer only from this context. If absent, say so.\n\nContext:\n{context}\n\nQuestion: {question}"
    )
    answer = (prompt | ChatOllama(model="llama3.2:3b", temperature=0) | StrOutputParser()).invoke(
        {"context": context, "question": question}
    )
    print(answer)


if __name__ == "__main__":
    main()