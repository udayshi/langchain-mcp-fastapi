"""Run a minimal LangGraph workflow."""

from __future__ import annotations

# Import TypedDict so the graph state has named, type-checked fields.
from typing import TypedDict

# Import the graph builder and the built-in entry and exit markers.
from langgraph.graph import END, START, StateGraph


# Describe every value that can move through this workflow.
class GreetingState(TypedDict):
    """State carried through the greeting workflow."""

    name: str
    greeting: str


def create_greeting(state: GreetingState) -> dict[str, str]:
    """Create a greeting from a non-empty name."""
    # Read and normalise the input value without mutating shared graph state.
    name = state["name"].strip()
    # Stop immediately when the graph receives invalid input.
    if not name:
        raise ValueError("name must not be blank")
    # Return only the state field this node changes; LangGraph merges this update.
    return {"greeting": f"Hello, {name}!"}


def main() -> None:
    """Run the compiled workflow and print its final greeting."""
    # Create a builder that uses GreetingState as its shared state contract.
    builder = StateGraph(GreetingState)
    # Register the Python function under the node name used by graph edges.
    builder.add_node("create_greeting", create_greeting)
    # Define the flow: begin at START, run the node, then finish at END.
    builder.add_edge(START, "create_greeting")
    builder.add_edge("create_greeting", END)
    # Validate and compile the graph declaration into a runnable object.
    graph = builder.compile()
    # Supply initial state and execute the only path through the graph.
    result = graph.invoke({"name": "Uday Shiwakoti", "greeting": "Hi world"})

    # Read the node's merged update from the final state.
    print(result["greeting"])


# Run main only when this file is executed, not when it is imported.
if __name__ == "__main__":
    main()
