"""Require a person to approve an action before completing it."""

from __future__ import annotations

# Literal limits approval status; TypedDict describes the graph state.
from typing import Literal, TypedDict

# Import checkpointing, graph markers, and pause/resume primitives.
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


# Carry the action awaiting a decision and its final approval status.
class ApprovalState(TypedDict):
    """State for a single approval request."""

    action: str
    status: Literal["pending", "approved", "rejected"]


def request_approval(state: ApprovalState) -> dict[str, Literal["approved", "rejected"]]:
    """Pause until a person approves or rejects the action."""
    # Stop graph execution and return this context to the external caller.
    approved = interrupt({"question": "Approve this action?", "action": state["action"]})
    # When resumed, treat only literal True as approval for a safe default.
    return {"status": "approved" if approved is True else "rejected"}


def main() -> None:
    """Pause a run and resume it with an approval decision."""
    # Create a graph whose shared data follows ApprovalState.
    builder = StateGraph(ApprovalState)
    # Register the pausing node and define its start-to-end path.
    builder.add_node("request_approval", request_approval)
    builder.add_edge(START, "request_approval")
    builder.add_edge("request_approval", END)
    # A checkpointer is required to resume an interrupted graph run.
    graph = builder.compile(checkpointer=InMemorySaver())
    # Keep this stable ID so the resume command addresses the paused run.
    config = {"configurable": {"thread_id": "approval-demo"}}

    # Run until interrupt() pauses the node and returns an interrupt payload.
    paused = graph.invoke({"action": "Send the monthly report", "status": "pending"}, config=config)
    # Display the requested action so a person can make an informed decision.
    print(paused["__interrupt__"])
    # Read a local human response; a production UI would provide this decision.
    user_input=input(f"Should I approve {paused['__interrupt__']}? (y/n) ?")
    # Start with rejection, so unknown input cannot approve the action.
    do_action:bool=False
    # Approve only explicit affirmative answers.
    if user_input.lower() in ["y","yes","ok"]:
        do_action=True

    # Resume the same paused graph run with the verified Boolean decision.
    complete = graph.invoke(Command(resume=do_action), config=config)
    # Print the final state written by request_approval after it resumes.
    print(complete["status"])


# Run this interactive demo only when this file is executed directly.
if __name__ == "__main__":
    main()
