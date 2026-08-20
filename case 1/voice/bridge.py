"""The two connections the voice assistant needs: the robot, and the model.

Kept in one file so ``voice/`` depends on nothing outside Case 1.

    connect_tools()  launch ``server.py`` over stdio and expose its tools
    LLMClient        one call to any OpenAI-compatible chat endpoint

The MCP half is a plain client of the Case 1 server: it launches it as a
subprocess, reads the tool catalogue, and forwards calls. The server has no
idea a microphone is involved, which is the point -- it stays usable from
Claude Code, Cursor or the Case 3 agent with nothing changed.

Configure the model with three environment variables (same names as Case 3, so
one ``.env`` serves both):

    AGENT_BASE_URL   e.g. https://integrate.api.nvidia.com/v1
    AGENT_API_KEY    your key
    AGENT_MODEL      the model id, e.g. z-ai/glm-5.2
"""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass


# --------------------------------------------------------------------------- #
# The robot: an MCP stdio client
# --------------------------------------------------------------------------- #
class ToolBridge:
    """Adapter around an initialized MCP session."""

    def __init__(self, session) -> None:
        self.session = session

    async def openai_tools(self) -> list[dict]:
        """The tool catalogue in OpenAI function-calling format.

        Descriptions and schemas come straight from ``server.py``'s docstrings,
        so tool-choice quality is driven by that file, not by this one.
        """
        resp = await self.session.list_tools()
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.inputSchema
                    or {"type": "object", "properties": {}},
                },
            }
            for t in resp.tools
        ]

    async def call(self, name: str, arguments: dict) -> str:
        """Run one tool, returning its result as text.

        Failures come back as ``ERROR: ...`` rather than raising, so the model
        reads the Diamond safety layer's plain-language reason and corrects
        itself instead of the conversation dying.
        """
        try:
            result = await self.session.call_tool(name, arguments)
        except Exception as exc:
            return f"ERROR: {type(exc).__name__}: {exc}"
        parts = [getattr(b, "text", str(b)) for b in result.content]
        text = "\n".join(p for p in parts if p)
        return f"ERROR: {text}" if getattr(result, "isError", False) else text


@asynccontextmanager
async def connect_tools(command: str, args: list[str],
                        log_path: str | None = None):
    """Launch the MCP server over stdio and yield a :class:`ToolBridge`.

    The session stays open for the whole conversation on purpose. The server
    holds live state -- the active motion pattern in ``PatternRunner``, the
    trajectory job registry, the freedrive flag -- so reconnecting between
    utterances would silently lose it, and "stir the pan" followed by "faster"
    would fail with "no pattern is running".

    Args:
        command: Executable that launches the server (usually the interpreter).
        args: Its arguments (the path to ``server.py``).
        log_path: Where to send the server's stderr. FastMCP prints a full
            rich traceback for every rejected tool call, which would scroll the
            spoken conversation off the screen, so it goes to a file by
            default. None keeps it on the terminal, which is what you want when
            debugging the server itself.
    """
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    # The MCP SDK does NOT inherit the environment. With ``env`` left unset it
    # falls back to ``get_default_environment()``, which forwards only HOME,
    # LOGNAME, PATH, SHELL, TERM and USER -- so every UR_* / VISION_* variable
    # was silently dropped on the floor and `UR_MODEL=ur5e python run_voice.py`
    # configured nothing at all. Pass the environment through explicitly, or
    # the server can only ever be configured through its argv.
    params = StdioServerParameters(command=command, args=args,
                                   env=dict(os.environ))
    errlog = open(log_path, "w", encoding="utf-8") if log_path else None
    try:
        stream = stdio_client(params) if errlog is None else \
            stdio_client(params, errlog=errlog)
        async with stream as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield ToolBridge(session)
    finally:
        if errlog is not None:
            errlog.close()


# --------------------------------------------------------------------------- #
# The model: an OpenAI-compatible chat client
# --------------------------------------------------------------------------- #
@dataclass
class ToolCall:
    """One tool the model wants to run."""

    id: str
    name: str
    arguments: dict


@dataclass
class Turn:
    """One assistant turn: optional text plus any tool calls."""

    text: str | None
    tool_calls: list[ToolCall]

    def as_message(self) -> dict:
        """Rebuild the assistant message to append to the conversation.

        Tool calls must be echoed verbatim so the follow-up ``tool`` messages
        line up with them by ``tool_call_id``.
        """
        msg: dict = {"role": "assistant", "content": self.text or ""}
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name,
                                 "arguments": json.dumps(c.arguments)},
                }
                for c in self.tool_calls
            ]
        return msg


class LLMClient:
    """Thin wrapper over an OpenAI-compatible chat-completions endpoint."""

    def __init__(self, model: str | None = None, base_url: str | None = None,
                 api_key: str | None = None, temperature: float = 0.2) -> None:
        from openai import OpenAI

        self.model = model or os.environ.get("AGENT_MODEL")
        if not self.model:
            raise ValueError(
                "No model set. Export AGENT_MODEL (and AGENT_BASE_URL / "
                "AGENT_API_KEY), or pass model=... See ../../llm-client/."
            )
        self.temperature = temperature
        self.client = OpenAI(
            base_url=base_url or os.environ.get("AGENT_BASE_URL"),
            api_key=api_key or os.environ.get("AGENT_API_KEY", "not-needed"),
        )

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> Turn:
        """Send the conversation plus the tool catalogue, return one turn."""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools or None,
            tool_choice="auto" if tools else None,
            temperature=self.temperature,
        )
        msg = resp.choices[0].message
        calls = []
        for tc in (msg.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append(ToolCall(id=tc.id, name=tc.function.name,
                                  arguments=args))
        return Turn(text=msg.content, tool_calls=calls)
