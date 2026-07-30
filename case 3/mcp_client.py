"""Talk to the robot's MCP server, the agent's hands.

Case 3 does NOT re-implement the robot interface. It consumes an MCP server as a
black box: the Case 1 server, or the pre-built one UR provides at the event. This
module launches that server as a subprocess over stdio, lists the tools it
exposes, and forwards the model's tool calls to it.

Two things the orchestrator needs:
    - ``openai_tools()``  -> the tool catalogue in OpenAI function-calling format,
      so it can be handed straight to :meth:`LLMClient.chat`.
    - ``call(name, args)`` -> run one tool, return its result as text.

Everything is async because the MCP SDK is. Use it as an async context manager so
the server process is started and cleaned up for you. Provided; you should not
need to change this.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class ToolBridge:
    """Adapter around an initialized MCP :class:`ClientSession`."""

    def __init__(self, session: ClientSession):
        self.session = session

    async def list_tools(self):
        """Raw MCP tool descriptors (name, description, inputSchema)."""
        resp = await self.session.list_tools()
        return resp.tools

    async def openai_tools(self) -> list[dict]:
        """Tool catalogue in OpenAI function-calling format.

        This is the 'menu' the model reads to decide which tool to call. Each
        tool's description and JSON-schema come straight from the server, so the
        quality of the server's docstrings (Case 1) drives the agent's choices.
        """
        tools = await self.list_tools()
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.inputSchema or {"type": "object", "properties": {}},
                },
            }
            for t in tools
        ]

    async def call(self, name: str, arguments: dict) -> str:
        """Run one tool and return its result as text.

        Errors are returned as an ``ERROR: ...`` string rather than raised, so the
        model sees the failure reason and can self-correct (that string is the
        feedback that closes the loop).
        """
        result = await self.session.call_tool(name, arguments)
        parts = [getattr(block, "text", str(block)) for block in result.content]
        text = "\n".join(p for p in parts if p)
        if getattr(result, "isError", False):
            return f"ERROR: {text}"
        return text


@asynccontextmanager
async def connect_tools(command: str, args: list[str]):
    """Launch an MCP server over stdio and yield a :class:`ToolBridge`.

    Args:
        command: Executable to launch (e.g. ``python``).
        args: Its arguments (e.g. ``["../case 1/server.py"]``).

    Usage::

        async with connect_tools("python", ["../case 1/server.py"]) as tools:
            schemas = await tools.openai_tools()
            out = await tools.call("get_robot_state", {})
    """
    params = StdioServerParameters(command=command, args=args)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield ToolBridge(session)
