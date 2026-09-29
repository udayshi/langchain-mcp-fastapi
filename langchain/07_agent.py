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
