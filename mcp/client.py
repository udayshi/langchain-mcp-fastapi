"""Manually verify authenticated access to the local MCP server."""

from __future__ import annotations

import asyncio
import os

from fastmcp import Client


def load_token() -> str:
    """Load the token used to authenticate to the local server."""
    token = os.getenv("MCP_AUTH_TOKEN", "")
    if len(token) < 32:
        raise RuntimeError("Set MCP_AUTH_TOKEN to the server token before running the client.")
    return token


async def main() -> None:
    """Call the add tool over authenticated Streamable HTTP."""
    async with Client("http://127.0.0.1:3890/mcp", auth=load_token()) as client:
        result = await client.call_tool("multiply", {"left": 2, "right": 3})
    print(result)


if __name__ == "__main__":
    asyncio.run(main())