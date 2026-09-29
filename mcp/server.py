"""Local token-protected MCP server."""

from __future__ import annotations

import os

from fastmcp import FastMCP
from fastmcp.server.auth.providers.jwt import StaticTokenVerifier


def load_token() -> str:
    """Load a sufficiently long local development token from the environment."""
    token = os.getenv("MCP_AUTH_TOKEN", "")
    if len(token) < 32:
        raise RuntimeError("Set MCP_AUTH_TOKEN to a random token of at least 32 characters.")
    return token


auth = StaticTokenVerifier(
    tokens={
        load_token(): {
            "client_id": "local-langchain-client",
            "scopes": ["tools:read"],
        }
    },
    required_scopes=["tools:read"],
)
mcp = FastMCP(name="Token-protected calculator", auth=auth)


@mcp.tool
def add(left: float, right: float) -> float:
    """Add two numbers and return their sum."""
    return left + right


@mcp.tool
def multiply(left: float, right: float) -> float:
    """Multiply two numbers and return their product."""
    return left * right


def main() -> None:
    """Serve authenticated Streamable HTTP MCP requests on the local loopback interface."""
    mcp.run(transport="streamable-http", host="127.0.0.1", port=3890)


if __name__ == "__main__":
    main()
