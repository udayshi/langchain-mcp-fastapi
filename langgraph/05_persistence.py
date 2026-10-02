"""Persist a short local conversation in memory."""

from __future__ import annotations

# Import the message type returned by the model node.
from langchain_core.messages import AnyMessage
# Use Ollama for the local conversational model.
from langchain_ollama import ChatOllama
# InMemorySaver checkpoints state by thread ID for this demonstration.
from langgraph.checkpoint.memory import InMemorySaver
# MessagesState preserves the growing conversation; START begins each run.
from langgraph.graph import START, MessagesState, StateGraph


def respond(state: MessagesState) -> dict[str, list[AnyMessage]]:
    """Return one model response for the accumulated conversation."""
    # Send all messages restored from the checkpoint plus the latest user message.
    response = ChatOllama(model="llama3.2:3b", temperature=0).invoke(state["messages"])
    # MessagesState appends this one new model response to the history.
    return {"messages": [response]}


def main() -> None:
    """Invoke one persistent conversation twice."""
    # Build a graph that uses the built-in conversation state schema.
    builder = StateGraph(MessagesState)
    # Register the response node and make it the first node in every invocation.
    builder.add_node("respond", respond)
    builder.add_edge(START, "respond")
    # Attach a checkpointer so state is saved after each graph invocation.
    graph = builder.compile(checkpointer=InMemorySaver())
    # Reuse this identifier to reload the same conversation on the next invocation.
    config = {"configurable": {"thread_id": "demo-conversation"}}

    # Store the first user message and the model's response under this thread ID.
    graph.invoke({"messages": [("user", "My name is Uday Shiwakoti.")]}, config=config)
    # Resume that thread; LangGraph restores the earlier messages before responding.
    result = graph.invoke({"messages": [("user", "What is my name?")]}, config=config)
    # Print the latest model response from the persisted conversation.
    print(result["messages"][-1].content)


# Run this demo only when this source file is executed directly.
if __name__ == "__main__":
    main()
