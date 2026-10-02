# LLM application learning lab

This repository is a hands-on path for building reliable local LLM applications with Python, `uv`, Ollama,
LangChain, LangGraph, LangSmith, and MCP. Each guide progresses from a small runnable example to the practices that
make an application testable, observable, and safe to operate.

## Why I created this project

I created this project to build practical, reproducible experience with local models, workflow orchestration,
testing, observability, evaluation, and secure tool access.

Contractual obligations prevent me from disclosing development URLs or proprietary workplace code.

## Why learn these topics?

Learning only how to call a model is not enough to build a dependable application. The guides are arranged so that
each one solves a different practical problem:

| Guide | Why it matters |
| --- | --- |
| [LangChain](docs/langchain.md) | A model call needs clear prompts, validated outputs, retrieval, and carefully bounded tools. |
| [LangChain TDD](docs/langchain-tdd.md) | Model calls are slow and variable; core application logic should still be fast and reliable to test. |
| [LangGraph](docs/langgraph.md) | Real workflows have state, branches, retries, tools, and human decisions that a linear chain cannot express clearly. |
| [LangSmith](docs/langsmith.md) | A final answer alone cannot explain a bad result or prove a change made an application better. |
| [MCP](docs/mcp.md) | Models need narrow, authenticated access to external capabilities rather than unrestricted system access. |

Together, the learning path moves from **one local model call** to a **tested, stateful, observable application with
secure tools**.

## Prerequisites

- Python 3.11 or later
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) for examples that call a local model
- A LangSmith account and API key only for the LangSmith guide

## Setup

Install the repository dependencies:

```bash
uv sync
```

Start Ollama in one terminal and download the baseline models in another:

```bash
ollama serve
```

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

`llama3.2:3b` is used for local chat and workflow examples. `nomic-embed-text` is used by the RAG examples.

## Suggested learning order

1. Start with [LangChain](docs/langchain.md): learn the model, prompt, chain, structured-output, RAG, and tool
   boundaries.
2. Continue with [LangChain TDD](docs/langchain-tdd.md): learn how to test those boundaries without depending on a
   running model.
3. Study [LangGraph](docs/langgraph.md): use explicit state and routing when the workflow becomes multi-step.
4. Follow [MCP](docs/mcp.md): expose a small, authenticated tool server.
5. Finish with [LangSmith](docs/langsmith.md): trace the application, capture feedback, and evaluate changes.

The MCP guide can also be completed independently after the basic LangChain examples.

## Run the examples

### LangChain

Each file in `langchain/` is independent and runnable:

```bash
uv run python langchain/01_basic_chat.py
uv run python langchain/03_chain.py
uv run python langchain/06_rag.py
uv run python langchain/09_health_check.py
```

The MCP-focused LangChain examples additionally require the token-protected MCP server and `MCP_AUTH_TOKEN`; follow
the [MCP guide](docs/mcp.md) before running `langchain/10_mcp_client.py` or later MCP examples.

### Tests and type checks

The unit tests are offline and do not require Ollama:

```bash
uv run python -m unittest discover -s tests
uv run mypy
```

The TDD companion tests live in `langchain/tests`. Run their deterministic tests with:

```bash
uv run pytest -q langchain/tests
```

The local-model integration test is skipped by default. Run it only after Ollama is running and the model is pulled:

```bash
RUN_OLLAMA_INTEGRATION=1 uv run pytest -q langchain/tests/test_ollama_integration.py
```

### LangGraph

The first two examples do not call a model; later examples require Ollama. Start with:

```bash
uv run python langgraph/01_basic_graph.py
uv run python langgraph/02_conditional_routing.py
uv run python langgraph/03_llm_node.py
```

### LangSmith

Set these values in the same terminal where you run the examples. The locked SDK uses
`LANGSMITH_TRACING_V2=true` to enable tracing:

```bash
export LANGSMITH_TRACING_V2=true
export LANGSMITH_API_KEY="your-langsmith-api-key"
export LANGSMITH_PROJECT="langsmith-playground"
uv run python langsmith/00-check-config.py
uv run python langsmith/01-trace.py
```

Find a trace ID and record feedback:

```bash
uv run python langsmith/06-list-runs.py
uv run python langsmith/07-feedback.py "<run-id>" --helpful
```

The [LangSmith guide](docs/langsmith.md) contains the complete numbered walkthrough and runnable source for tracing,
datasets, and evaluation.

### MCP server

Follow the [MCP guide](docs/mcp.md) for the complete security notes. In two clean terminals, use the same generated
token:

```bash
export MCP_AUTH_TOKEN="$(openssl rand -hex 32)"
uv run python mcp/server.py
```

```bash
export MCP_AUTH_TOKEN="the-same-token"
uv run python mcp/client.py
```
