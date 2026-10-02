"""Use an authenticated HTTP MCP tool from a LangGraph workflow."""

from __future__ import annotations

import asyncio
import os
from typing import Literal

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition


def load_token() -> str:
    """Load the bearer token required by the local MCP server."""
    # Read the secret only from the environment, never from source code.
    token = os.getenv("MCP_AUTH_TOKEN", "")
    # Reject missing or implausibly short tokens before attempting a connection.
    if len(token) < 32:
        raise RuntimeError("Set MCP_AUTH_TOKEN to the local MCP server token.")
    return token


async def main() -> None:
    """Ask a LangGraph tool loop to use the local MCP calculator."""
    # Configure the adapter to send the token with every request to the local endpoint.
    client = MultiServerMCPClient(
        {
            "local_calculator": {
                "transport": "http",
                "url": "http://127.0.0.1:3890/mcp",
                "headers": {"Authorization": f"Bearer {load_token()}"},
            }
        }
    )
    # Discover MCP tools and convert them into standard LangChain tools.
    tools = await client.get_tools()
    # Bind the discovered tools so the model can request only advertised MCP actions.
    model = ChatOllama(model="llama3.2:3b", temperature=0).bind_tools(tools)

    def call_model(state: MessagesState) -> dict[str, list[AnyMessage]]:
        """Ask the model whether it needs an MCP tool or can answer directly."""
        # Put the tool-use policy before the accumulated conversation history.
        response = model.invoke(
            [
                SystemMessage(content="Use the calculator tool for arithmetic. Do not calculate yourself."),
                *state["messages"],
            ]
        )
        # Append the model response so tools_condition can inspect its tool calls.
        return {"messages": [response]}

    def route_after_model(state: MessagesState) -> Literal["tools", "__end__"]:
        """Route model-requested MCP tools or complete the graph."""
        # Select ToolNode for a tool call; otherwise LangGraph ends the workflow.
        return tools_condition(state)

    # Build the same model → tools → model loop used for local LangChain tools.
    builder = StateGraph(MessagesState)
    builder.add_node("model", call_model)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "model")
    builder.add_conditional_edges("model", route_after_model)
    builder.add_edge("tools", "model")

    # Run the async graph and print the final model response after the MCP call.
    result = await builder.compile().ainvoke(
        {"messages": [HumanMessage(content="What is 21 multiplied by 2?")]}
    )
    print(result["messages"][-1].content)


# Run the asynchronous entry point only when this file is executed directly.
if __name__ == "__main__":
    asyncio.run(main())
