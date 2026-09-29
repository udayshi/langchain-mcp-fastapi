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
    email = """Hi, I am Uday Shiwakoti. Please send two BOOK-001 copies by express delivery. """
    model = ChatOllama(model="llama3.2:3b", temperature=0)
    result = model.with_structured_output(PurchaseDecision).invoke(email)

    if not isinstance(result, PurchaseDecision):
        raise TypeError("Model did not return the requested PurchaseDecision schema.")
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()