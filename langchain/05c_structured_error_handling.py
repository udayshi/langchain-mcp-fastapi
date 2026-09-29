"""Expose structured-output parsing errors safely."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field


class Meeting(BaseModel):
    """Structured meeting details."""

    title: str = Field(min_length=1, description="Short meeting title")
    attendees: list[str] = Field(min_length=1, description="Names of attendees")
    duration_minutes: int = Field(gt=0, le=480, description="Meeting duration in minutes")


def parse_meeting_result(result: Mapping[str, Any]) -> Meeting:
    """Return a valid meeting or raise a controlled parsing error."""
    parsed = result.get("parsed")
    parsing_error = result.get("parsing_error")
    if parsing_error is not None or not isinstance(parsed, Meeting):
        raise RuntimeError(f"Could not produce a valid meeting: {parsing_error}")
    return parsed


def main() -> None:
    """Parse a meeting request and report any schema failure."""
    model = ChatOllama(model="llama3.2:3b", temperature=0)
    result: Mapping[str, Any] = model.with_structured_output(Meeting, include_raw=True).invoke(
        "Schedule a 30-minute architecture review with Uday Shiwakoti tomorrow."
    )
    print(parse_meeting_result(result).model_dump_json(indent=2))


if __name__ == "__main__":
    main()
