"""Route a support request by its urgency."""

from __future__ import annotations

# Literal limits routing values; TypedDict describes the workflow state.
from typing import Literal, TypedDict

# Import the graph builder and its start and end markers.
from langgraph.graph import END, START, StateGraph


# Define the input, classification result, and routing result for one ticket.
class TicketState(TypedDict):
    """State used to triage one support ticket."""

    message: str
    priority: Literal["normal", "urgent"]
    queue: str


def classify(state: TicketState) -> dict[str, Literal["normal", "urgent"]]:
    """Classify an urgent ticket using an intentionally simple local rule."""
    # Apply a deterministic rule so routing is easy to inspect and test.
    priority: Literal["normal", "urgent"] = "urgent" if "outage" in state["message"].lower() else "normal"
    # Return the priority update for LangGraph to merge into state.
    return {"priority": priority}


def choose_queue(state: TicketState) -> Literal["urgent_queue", "normal_queue"]:
    """Select the queue node for the classified ticket."""
    # Return the registered node name that LangGraph should run next.
    return "urgent_queue" if state["priority"] == "urgent" else "normal_queue"


def assign_urgent(_: TicketState) -> dict[str, str]:
    """Assign an urgent ticket to the incident queue."""
    # This branch has a fixed state update, so it does not need to read state.
    return {"queue": "incident-response"}


def assign_normal(_: TicketState) -> dict[str, str]:
    """Assign a normal ticket to the standard queue."""
    # This branch also returns only the field it changes.
    return {"queue": "support"}


def main() -> None:
    """Build and run the routing workflow."""
    # Create a graph whose state must conform to TicketState.
    builder = StateGraph(TicketState)
    # Register the classifier and both possible destination nodes.
    builder.add_node("classify", classify)
    builder.add_node("urgent_queue", assign_urgent)
    builder.add_node("normal_queue", assign_normal)
    # Always classify first, then let choose_queue select one branch.
    builder.add_edge(START, "classify")
    builder.add_conditional_edges("classify", choose_queue)
    # Both queue nodes are terminal paths.
    builder.add_edge("urgent_queue", END)
    builder.add_edge("normal_queue", END)

    # Compile the declared nodes and edges before executing them.
    graph = builder.compile()
    # Invoke with complete initial state; classify will overwrite priority.
    result = graph.invoke({"message": "There is an outage in the payment service.", "priority": "normal", "queue": ""})
    # Print the queue written by the branch that ran.
    print(result["queue"])


# Run this demonstration only when the file is executed directly.
if __name__ == "__main__":
    main()
