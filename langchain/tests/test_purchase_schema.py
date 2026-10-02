"""Offline unit tests for the TDD purchase schema example."""

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
    """Valid customer and item data becomes a typed purchase decision."""
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
    """Invalid customer, item-list, and quantity values are rejected."""
    with pytest.raises(ValidationError):
        PurchaseDecision.model_validate(data)
