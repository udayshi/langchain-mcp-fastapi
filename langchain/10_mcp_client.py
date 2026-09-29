"""Use authenticated MCP tools from a LangChain agent."""

from __future__ import annotations

import asyncio
import os

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama


def mcp_token() -> str:
    """Read and validate the MCP bearer token from the environment."""
    token = os.environ.get("MCP_AUTH_TOKEN", "")
    if len(token) < 32:
        raise RuntimeError("Set MCP_AUTH_TOKEN to the server's token before running this client.")
    return token


async def main() -> None:
    """Discover authenticated MCP tools and let a local model use them."""
    client = MultiServerMCPClient(
        {
            "local_calculator": {
                "transport": "streamable_http",
                "url": "http://127.0.0.1:3890/mcp",
                "headers": {"Authorization": f"Bearer {mcp_token()}"},
            }
        }
    )
    tools = await client.get_tools()
    agent = create_agent(
        model=ChatOllama(model="llama3.2:3b", temperature=0),
        tools=tools,
        system_prompt="Use the calculator tool for arithmetic. Do not invent calculations.",
    )
    result = await agent.ainvoke({"messages": [{"role": "user", "content": "What is 21 multiplied by 2?"}]})
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
