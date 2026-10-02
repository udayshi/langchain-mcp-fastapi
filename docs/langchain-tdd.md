# LangChain walkthrough: test-driven development

This companion starts after you are comfortable with the basic-to-advanced examples in
[the main LangChain walkthrough](langchain.md). It teaches how to use TDD with LangChain while keeping
unit tests fast, deterministic, and independent of a running Ollama server.

In this repository, the runnable tests are in [`langchain/tests/`](../langchain/tests), and the small chat domain
module is [`langchain/tdd_chat.py`](../langchain/tdd_chat.py). The walkthrough's generic `src/` paths illustrate how
to structure a new standalone project; use the repository paths when running this checkout.

## 1. Setup

Create the project shown in the main walkthrough, then ensure its development tools are installed:

```bash
uv add langchain langchain-ollama pydantic
uv add --dev pytest mypy
mkdir -p src/langchain_playground tests
touch src/langchain_playground/__init__.py
```

Use strict type checking in `pyproject.toml`:

```toml
[tool.mypy]
strict = true
files = ["src", "tests"]
```

## 2. RED: write tests before the model boundary

Create `tests/test_chat.py` first:

```python
"""Unit tests for chat domain logic."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from langchain_playground.chat import answer_question


class FakeModel:
    """Controllable local substitute for a LangChain model."""

    def __init__(self, response: AIMessage) -> None:
        self.response = response
        self.calls: list[list[BaseMessage]] = []

    def invoke(self, input: Sequence[BaseMessage]) -> AIMessage:
        """Record input and return the configured response."""
        self.calls.append(list(input))
        return self.response


def test_answer_question_returns_text() -> None:
    model = FakeModel(AIMessage(content="A tuple is immutable."))

    assert answer_question("What is a tuple?", model) == "A tuple is immutable."
    assert model.calls == [[HumanMessage(content="What is a tuple?")]]


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_answer_question_rejects_blank_input(question: str) -> None:
    with pytest.raises(ValueError, match="question must not be blank"):
        answer_question(question, FakeModel(AIMessage(content="unused")))


def test_answer_question_rejects_non_text_response() -> None:
    model = FakeModel(AIMessage(content=[{"type": "text", "text": "not accepted"}]))

    with pytest.raises(ValueError, match="model must return text"):
        answer_question("What is a tuple?", model)
```

The test should fail at import time because the implementation is intentionally absent:

```bash
uv run pytest langchain/tests/test_chat.py
```

## 3. GREEN: implement the smallest behaviour

Create `src/langchain_playground/chat.py`:

```python
"""Domain logic for asking a chat model a question."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


class ChatModel(Protocol):
    """Minimal interface needed by the domain layer."""

    def invoke(self, input: Sequence[BaseMessage]) -> AIMessage:
        """Return a response for chat messages."""


def answer_question(question: str, model: ChatModel) -> str:
    """Return a text answer for a non-empty question."""
    if not question.strip():
        raise ValueError("question must not be blank")

    response = model.invoke([HumanMessage(content=question)])
    if not isinstance(response.content, str):
        raise ValueError("model must return text")
    return response.content
```

Run the tests and strict type checker:

```bash
uv run pytest langchain/tests/test_chat.py
uv run mypy
```

## 4. REFACTOR: test Pydantic schemas offline

Pydantic constraints are deterministic Python, so test them without a model. Save this as
`tests/test_purchase_schema.py`:

```python
"""Unit tests for the purchase schema."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, Field, ValidationError


class LineItem(BaseModel):
    """One requested inventory item."""

    sku: str = Field(min_length=1, max_length=20)
    quantity: int = Field(gt=0, le=100)


class PurchaseDecision(BaseModel):
    """Validated purchase request."""

    customer_name: str = Field(min_length=1)
    items: list[LineItem] = Field(min_length=1)


def test_purchase_schema_accepts_valid_data() -> None:
    decision = PurchaseDecision.model_validate(
        {"customer_name": "Uday Shiwakoti", "items": [{"sku": "BOOK-001", "quantity": 2}]}
    )

    assert decision.items[0].quantity == 2


@pytest.mark.parametrize(
    "data",
    [
        {"customer_name": "", "items": [{"sku": "BOOK-001", "quantity": 1}]},
        {"customer_name": "Uday Shiwakoti", "items": []},
        {"customer_name": "Uday Shiwakoti", "items": [{"sku": "BOOK-001", "quantity": 0}]},
    ],
)
def test_purchase_schema_rejects_invalid_data(data: object) -> None:
    with pytest.raises(ValidationError):
        PurchaseDecision.model_validate(data)
```

```bash
uv run pytest langchain/tests/test_purchase_schema.py
```

## 5. Keep live tests opt-in

Unit tests should use fakes. Test `ChatOllama`, RAG stores, and MCP servers in separate integration tests that run only
when explicitly enabled, for example:

```bash
RUN_OLLAMA_INTEGRATION=1 uv run pytest langchain/tests/test_ollama_integration.py
```

For each change, follow RED (a failing focused test), GREEN (the smallest passing implementation), then REFACTOR
(improve names and duplication while tests stay green). Do not assert an exact live-model sentence; assert stable
properties such as valid schemas, non-empty text, selected tools, and cited sources.
