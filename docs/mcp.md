# Token-protected MCP server with Python and uv

This walkthrough creates an independent local Model Context Protocol (MCP) project with exactly two application files:

```text
mcp/
├── server.py
└── client.py
```

The server uses Streamable HTTP at `http://127.0.0.1:3890/mcp`. It accepts requests only from the local machine and
requires a Bearer token for every MCP request.

## 1. Create the project and install dependencies

Run these commands from the directory where you want the independent project to live:

```bash
mkdir mcp
cd mcp
uv init
uv add "fastmcp>=2.7,<3"
```

Do not run `server.py` until `uv add` completes successfully: it installs FastMCP and writes the project's dependency
configuration. If you reopen the project later, run `uv sync` before starting the server.

Before running either file, make sure this terminal is not using a virtual environment from another project. If your
shell prompt shows an active environment, run `deactivate`, then change to `mcp/`. This prevents uv's
`VIRTUAL_ENV does not match the project environment path` warning and ensures it uses `mcp/.venv`.

The static token verifier used below is suitable for local development only. A production service should validate
short-lived JWTs against a trusted issuer or OAuth/OIDC provider instead of storing a valid token in its process.

## 2. Create `server.py`

Save the following complete server as `mcp/server.py`:

```python
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
            "client_id": "local-client",
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
    """Serve authenticated Streamable HTTP on the loopback interface."""
    mcp.run(transport="streamable-http", host="127.0.0.1", port=3890)


if __name__ == "__main__":
    main()
```

FastMCP's Streamable HTTP server exposes the `/mcp` endpoint by default, making the full local URL
`http://127.0.0.1:3890/mcp`.

## 3. Generate a token and start the server

Generate a high-entropy token without writing it to a source file, then start the server:

```bash
# Run this in a clean terminal, or run `deactivate` first.
export MCP_AUTH_TOKEN="$(openssl rand -hex 32)"
uv run python server.py
```

Keep this terminal running. Do not print, commit, or paste the token into logs or issue trackers.

## 4. Create `client.py`

Open a second terminal in the same `mcp/` directory. Set `MCP_AUTH_TOKEN` to the same value, then save this manual
verification client as `mcp/client.py`:

```python
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
        result = await client.call_tool("add", {"left": 2, "right": 3})
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
```

## 5. Manually verify the server

With the server still running and the same token exported in the second terminal, run:

```bash
# Run this in a second clean terminal, or run `deactivate` first.
export MCP_AUTH_TOKEN="the-same-token-used-for-the-server"
uv run python client.py
```

The result should represent `5.0`. To verify authentication safely, run the command from a new shell without setting
`MCP_AUTH_TOKEN`; it should fail before revealing available tools.

## 6. Production authentication

Do not expose this static-token version outside local development. A production server should terminate TLS, validate
signed and expiring tokens, authorize each tool for the relevant user or tenant, store secrets in a secret manager,
and restrict CORS, hosts, request sizes, and rate limits.
