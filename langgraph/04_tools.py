"""Build a local calculator agent with an explicit tool loop."""

from __future__ import annotations

# Literal constrains the routing function to valid graph destinations.
from typing import Literal

# Import message types, the tool decorator, and the local model integration.
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
# MessagesState appends messages; ToolNode executes requested, registered tools.
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition


# Expose this narrow, typed function as the only tool the model may request.
@tool
def multiply(left: float, right: float) -> float:
    """Multiply two numbers."""
    # Keep tool logic deterministic and independently testable.
    return left * right


def call_model(state: MessagesState) -> dict[str, list[AnyMessage]]:
    """Ask a tool-capable local model for the next message."""
    # Bind only multiply so the model cannot select an unrestricted action.
    model = ChatOllama(model="llama3.2:3b", temperature=0).bind_tools([multiply])
    # Give the model its rule followed by all conversation history.
    response = model.invoke(
        [
            SystemMessage(content="Use the multiply tool for arithmetic. Do not calculate yourself."),
            *state["messages"],
        ]
    )
    # Append the AI response, including any tool call, to MessagesState.
    return {"messages": [response]}


def route_after_model(state: MessagesState) -> Literal["tools", "__end__"]:
    """Route tool calls for execution, otherwise complete the graph."""
    # Route to ToolNode when the latest AI message requests a tool; otherwise end.
    return tools_condition(state)


def main() -> None:
    """Run the graph for one arithmetic question."""
    # Build a graph with the built-in message-accumulating state schema.
    builder = StateGraph(MessagesState)
    # Register the model node and a ToolNode restricted to multiply.
    builder.add_node("model", call_model)
    builder.add_node("tools", ToolNode([multiply]))
    # Start with the model, then route based on whether it requested a tool.
    builder.add_edge(START, "model")
    builder.add_conditional_edges("model", route_after_model)
    # Return tool results to the model so it can write the final user-facing answer.
    builder.add_edge("tools", "model")

    # Run the loop from one user message and capture the accumulated messages.
    result = builder.compile().invoke({"messages": [HumanMessage(content="What is 21 multiplied by 2?")]})
    # The last message is the model's final answer after any tool execution.
    print(result["messages"][-1].content)


# Run the demonstration only when this source file is executed directly.
if __name__ == "__main__":
    main()
