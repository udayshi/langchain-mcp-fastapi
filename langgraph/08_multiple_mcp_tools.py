"""Use tools from two authenticated HTTP MCP servers in one LangGraph workflow."""

from __future__ import annotations

import asyncio
import os
from typing import Literal

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition


def load_token(variable_name: str) -> str:
    """Load one sufficiently long MCP bearer token from the environment."""
    # Read each server secret independently so credentials never cross server boundaries.
    token = os.getenv(variable_name, "")
    # Fail before connecting when the required token was not configured.
    if len(token) < 32:
        raise RuntimeError(f"Set {variable_name} to the matching MCP server token.")
    return token


async def main() -> None:
    """Ask one LangGraph tool loop to use separate add and multiply MCP servers."""
    # Register each endpoint under a unique name and send its matching bearer token.
    client = MultiServerMCPClient(
        {
            "adder": {
                "transport": "http",
                "url": "http://127.0.0.1:3890/mcp",
                "headers": {"Authorization": f"Bearer {load_token('MCP_AUTH_TOKEN_1')}"},
            },
            "multiplier": {
                "transport": "http",
                "url": "http://127.0.0.1:3891/mcp",
                "headers": {"Authorization": f"Bearer {load_token('MCP_AUTH_TOKEN_2')}"},
            },
        },
        # Prefix tool names with their server names to prevent future name collisions.
        tool_name_prefix=True,
    )
    # Discover both servers' tools as one list of LangChain-compatible tools.
    tools = await client.get_tools()
    # Bind exactly those discovered tools to the local model.
    model = ChatOllama(model="llama3.2:3b", temperature=0).bind_tools(tools)

    def call_model(state: MessagesState) -> dict[str, list[AnyMessage]]:
        """Ask the model whether to call one of the discovered MCP tools."""
        # Require tool use for arithmetic rather than trusting model-calculated results.
        response = model.invoke(
            [
                SystemMessage(content="Use the available MCP tools for all arithmetic. Do not calculate yourself."),
                *state["messages"],
            ]
        )
        # Append the AI message so LangGraph can inspect any requested tool calls.
        return {"messages": [response]}

    def route_after_model(state: MessagesState) -> Literal["tools", "__end__"]:
        """Run requested MCP tools or finish when the model has answered."""
        # ToolNode runs only when the latest model message contains a tool request.
        return tools_condition(state)

    # Build the model → tools → model loop for both server tool sets.
    builder = StateGraph(MessagesState)
    builder.add_node("model", call_model)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "model")
    builder.add_conditional_edges("model", route_after_model)
    builder.add_edge("tools", "model")

    # Let the model select the correct server tools for both arithmetic operations.
    result = await builder.compile().ainvoke(
        {"messages": [HumanMessage(content="What are 2 plus 3 and 4 multiplied by 5?")]}
    )
    print(result["messages"][-1].content)


# Run the asynchronous graph only when this file is executed directly.
if __name__ == "__main__":
    asyncio.run(main())
