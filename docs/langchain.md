# LangChain walkthrough: basic to advanced

Each practical section has a complete, standalone file that you can copy, save, and run. The examples use local
Ollama and `llama3.2:3b`; no example depends on a file from an earlier section. Once these examples are familiar,
continue with [the TDD companion walkthrough](lanchaintdd.md).

## 0. Setup

Run Ollama in one terminal:

```bash
ollama serve
```

In another terminal, create a project and install everything used below:

```bash
mkdir langchain-playground
cd langchain-playground
uv init --package langchain-playground
uv add langchain langchain-ollama langchain-chroma langchain-text-splitters pydantic
ollama pull llama3.2:3b
ollama pull nomic-embed-text
mkdir langchain
```

Run every example as `uv run python langchain/<file>.py`.

## 1. Basic: one chat message

Save as `langchain/01_basic_chat.py`:

```python
"""Send one question to local Ollama."""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama


def main() -> None:
    """Ask one question and print text output."""
    model = ChatOllama(model="llama3.2:3b", temperature=0)
    response = model.invoke([HumanMessage(content="Explain a Python tuple in one sentence.")])
    if not isinstance(response.content, str):
        raise ValueError("Expected the model to return text.")
    print(response.content)


if __name__ == "__main__":
    main()
```

```bash
uv run python langchain/01_basic_chat.py
```

`temperature=0` makes results more repeatable but does not guarantee identical output.

## 2. Prompts: reusable instructions and data

Save as `langchain/02_prompt_template.py`:

```python
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
```

```bash
uv run python langchain/02_prompt_template.py
```

Templates separate application data from instructions, but they are not a security boundary against prompt injection.

## 3. Chains: compose a pipeline

Save as `langchain/03_chain.py`:

```python
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
```

```bash
uv run python langchain/03_chain.py
```

The `|` operator composes compatible runnables. Use `batch` only for independent inputs.

## 3.1 Two-model pipeline: one model's output becomes another model's input

This pattern gives each model a clear responsibility. A writer produces a draft, then a reviewer receives that exact
draft in its prompt and returns an improved version. The default pairing is `llama3.2:3b` for drafting and `gemma3:4b`
for review. You can use `gpt-oss:20b` as either model if the machine has enough memory.

Download the models you plan to use:

```bash
ollama pull llama3.2:3b
ollama pull gemma3:4b
# Optional; this model is substantially larger:
ollama pull gpt-oss:20b
```

Save as `langchain/03b_two_model_pipeline.py`:

```python
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
```

Run the default pairing:

```bash
uv run python langchain/03b_two_model_pipeline.py
```

Use `gpt-oss:20b` as the reviewer instead:

```bash
REVIEWER_MODEL=gpt-oss:20b uv run python langchain/03b_two_model_pipeline.py
```

Model output is untrusted input for the next stage: use delimiters as above, validate any structured result, and do not
let a drafting model's text change the reviewer model's system-level instructions. This pipeline performs two separate
model calls, so it increases latency and local memory pressure. Ollama currently lists `llama3.2:3b`, `gemma3:4b`, and
`gpt-oss:20b`; the latter is listed as a 14 GB download, so use it only when the host can accommodate it.

## 4. Structured output: validated data

Save as `langchain/05_structured_output.py`. Use a model that supports structured output or tool calling.

```python
"""Return a validated Pydantic object from a local model."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field


class SupportAnswer(BaseModel):
    """Validated result for a support workflow."""

    answer: str = Field(description="Direct user-facing answer")
    confidence: float = Field(ge=0, le=1, description="Model-estimated confidence")
    needs_human: bool = Field(description="Whether a human should review the result")


def main() -> None:
    """Request and print structured data."""
    model = ChatOllama(model="llama3.2:3b", temperature=0)
    result = model.with_structured_output(SupportAnswer).invoke(
        [
            SystemMessage(
                content=(
                    "Answer with a quick, self-contained response. Do not mention, cite, link to, "
                    "or recommend third-party sources."
                )
            ),
            HumanMessage(content="Can I change my delivery address after dispatch?"),
        ]
    )
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
```

```bash
uv run python langchain/05_structured_output.py
```

Schema validation helps programs consume model output. Never use a model-generated confidence score alone for a
high-stakes decision.

### 4.1 Extract nested, constrained data

Use nested Pydantic models when the answer is passed to application logic. Field descriptions become part of the
schema given to the model, while Pydantic validates the returned object before your code uses it.

Save as `langchain/05b_purchase_extraction.py`:

```python
"""Extract a validated purchase decision from unstructured text."""

from __future__ import annotations

from typing import Literal

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field


class LineItem(BaseModel):
    """One requested inventory item."""

    sku: str = Field(min_length=1, max_length=20, description="Inventory SKU")
    quantity: int = Field(gt=0, le=100, description="Requested number of units")


class PurchaseDecision(BaseModel):
    """Validated purchase request extracted from an email."""

    customer_name: str = Field(min_length=1, description="Customer's name")
    items: list[LineItem] = Field(min_length=1, description="Requested inventory items")
    delivery: Literal["standard", "express"] = Field(description="Requested delivery speed")
    requires_review: bool = Field(description="True when a human must verify the request")


def main() -> None:
    """Extract a purchase request and print canonical JSON."""
    email = """Hi, I am Uday Shiwakoti. Please send two BOOK-001 copies by express delivery.
    Please let me know if you need anything else."""
    model = ChatOllama(model="llama3.2:3b", temperature=0)
    result = model.with_structured_output(PurchaseDecision).invoke(email)

    if not isinstance(result, PurchaseDecision):
        raise TypeError("Model did not return the requested PurchaseDecision schema.")
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
```

```bash
uv run python langchain/05b_purchase_extraction.py
```

`Literal`, length limits, and numeric bounds are enforced after the model responds. They protect your application
from malformed data, but cannot establish that a model's factual claim is true; review business-critical results.

### 4.2 Handle invalid model output explicitly

Some local models or model settings may not support structured output reliably. Use `include_raw=True` to retain the
unparsed response and surface a controlled error rather than silently continuing with incomplete data.

Save as `langchain/05c_structured_error_handling.py`:

```python
"""Expose structured-output parsing errors safely."""

from __future__ import annotations

from typing import Any

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field


class Meeting(BaseModel):
    """Structured meeting details."""

    title: str = Field(min_length=1, description="Short meeting title")
    attendees: list[str] = Field(min_length=1, description="Names of attendees")
    duration_minutes: int = Field(gt=0, le=480, description="Meeting duration in minutes")


def main() -> None:
    """Parse a meeting request and report any schema failure."""
    model = ChatOllama(model="llama3.2:3b", temperature=0)
    runnable = model.with_structured_output(Meeting, include_raw=True)
    result: dict[str, Any] = runnable.invoke(
        "Schedule a 30-minute architecture review with Uday Shiwakoti tomorrow."
    )
    parsed = result["parsed"]
    parsing_error = result["parsing_error"]

    if parsing_error is not None or not isinstance(parsed, Meeting):
        raise RuntimeError(f"Could not produce a valid meeting: {parsing_error}")

    print(parsed.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
```

```bash
uv run python langchain/05c_structured_error_handling.py
```

In an API, map this failure to a clear retryable error. Log only safe diagnostics: prompts and raw model responses can
contain sensitive data. Do not automatically retry an unbounded number of times.

When you are ready to test Pydantic schemas and model-boundary logic, use the offline examples in
[the TDD companion walkthrough](lanchain-walkthrough-tdd.md).

## 5. RAG: answer from documents

This repository includes `rag_source/handbook.md`. To create it in a new project:

```bash
mkdir -p rag_source
printf 'Employees receive 25 days of annual leave each year.\n' > rag_source/handbook.md
```

Save this complete index-and-query program as `langchain/06_rag.py`:

```python
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
    source_path = Path("docs/handbook.md")
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
```

```bash
uv run python langchain/06_rag.py
```

Evaluate RAG with known-answer questions, unanswerable questions, and stale or conflicting documents. Inspect retrieved
chunks before changing prompts.

## 6. Tools and agents: bounded actions

Save as `langchain/07_agent.py`. This example only reads an in-memory dictionary; never expose unrestricted shell,
database, filesystem, or network tools to a model.

```python
"""Use a bounded, read-only tool through a LangChain agent."""

from __future__ import annotations

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama


@tool
def product_stock(sku: str) -> str:
    """Return stock status for a validated product SKU."""
    cleaned_sku = sku.strip().upper()
    if not cleaned_sku or len(cleaned_sku) > 20:
        return "Invalid SKU."
    quantity = {"BOOK-001": 12, "BOOK-002": 0}.get(cleaned_sku)
    return "SKU not found." if quantity is None else f"{cleaned_sku}: {quantity} available"


def main() -> None:
    """Ask a tool-capable local model an inventory question."""
    agent = create_agent(
        model=ChatOllama(model="llama3.2:3b", temperature=0),
        tools=[product_stock],
        system_prompt="Use tools only when needed. Never invent stock levels.",
    )
    result = agent.invoke({"messages": [{"role": "user", "content": "Is BOOK-001 in stock?"}]})
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
```

```bash
uv run python langchain/07_agent.py
```

Use an agent only when a model must choose among multiple tools. Test tools as normal Python and impose limits on
permissions, calls, time, and cost.

## 7. Conversation state

Save as `langchain/08_conversation.py`:

```python
"""Maintain a small in-memory conversation history."""

from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_ollama import ChatOllama


def text(message: AIMessage) -> str:
    """Return text content or reject non-text model output."""
    if not isinstance(message.content, str):
        raise ValueError("Expected the model to return text.")
    return message.content


def main() -> None:
    """Run two turns that share message history."""
    model = ChatOllama(model="llama3.2:3b", temperature=0)
    history: list[BaseMessage] = [HumanMessage(content="My name is Uday Shiwakoti.")]
    history.append(model.invoke(history))
    history.append(HumanMessage(content="What is my name? Reply with only the name."))
    print(text(model.invoke(history)))


if __name__ == "__main__":
    main()
```

```bash
uv run python langchain/08_conversation.py
```

In production, isolate state per user, cap history size, and summarise or trim old turns. Do not log sensitive prompts
or replies.

## 8. Production health check

Save as `langchain/09_health_check.py`:

```python
"""Check that the configured Ollama model returns text."""

from __future__ import annotations

import sys

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama


def main() -> int:
    """Return zero only when local Ollama returns non-empty text."""
    try:
        response = ChatOllama(model="llama3.2:3b", temperature=0).invoke(
            [HumanMessage(content="Reply with: ready")]
        )
    except Exception as error:
        print(f"Ollama health check failed: {error}", file=sys.stderr)
        return 1
    if not isinstance(response.content, str) or not response.content.strip():
        print("Ollama health check failed: model returned no text", file=sys.stderr)
        return 1
    print("Ollama health check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

```bash
uv run python langchain/09_health_check.py
```

Before deployment, add input validation, timeouts and retries, safe errors, monitoring, versioned prompts, and an
offline regression suite. Require human approval for actions that send messages, alter records, or spend money.

## 9. Use the token-protected MCP server from LangChain

First follow [the MCP server guide](mcp.md) and start its server at `http://127.0.0.1:3890/mcp`. In this project,
install the MCP adapter and ensure `MCP_AUTH_TOKEN` has the same value used by the server:

```bash
uv add langchain-mcp-adapters
export MCP_AUTH_TOKEN="the-same-token-used-by-the-server"
```

Save this complete client as `langchain/10_mcp_client.py`:

```python
"""Use authenticated MCP tools from a LangChain agent."""

from __future__ import annotations

import asyncio
import os

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama


async def main() -> None:
    """Discover authenticated MCP tools and let a local model use them."""
    token = os.environ.get("MCP_AUTH_TOKEN", "")
    if len(token) < 32:
        raise RuntimeError("Set MCP_AUTH_TOKEN to the server's token before running this client.")

    client = MultiServerMCPClient(
        {
            "local_calculator": {
                "transport": "streamable_http",
                "url": "http://127.0.0.1:3890/mcp",
                "headers": {"Authorization": f"Bearer {token}"},
            }
        }
    )
    tools = await client.get_tools()
    agent = create_agent(
        model=ChatOllama(model="llama3.2:3b", temperature=0),
        tools=tools,
        system_prompt="Use the calculator tool for arithmetic. Do not invent calculations.",
    )
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "What is 21 multiplied by 2?"}]}
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
```

Run it while the MCP server is running:

```bash
uv run python langchain/10_mcp_client.py
```

The `Authorization` header is sent with each Streamable HTTP request. Keep the token outside code and configuration
files that may be committed. Use a tool-capable Ollama model; if the model does not make tool calls reliably, test
tool discovery independently before changing agent prompts.
