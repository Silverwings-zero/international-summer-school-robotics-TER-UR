"""Alternative brain: drive the conversation with your Claude Code subscription.

``conversation.VoiceAgent`` talks to an OpenAI-compatible endpoint and needs an
API key. This module is the drop-in alternative that uses the ``claude`` CLI you
are already logged into, so the tokens come out of your Claude subscription
instead of a separate API bill. Same two methods, so ``run_voice.py`` can swap
between them with a flag.

How it differs from the other backend, and why:

* **The agent loop is not ours.** ``claude-agent-sdk`` is Claude Code packaged
  as a library: it owns the reason/act loop, the context window and the MCP
  connection to ``server.py``. We supply a prompt and read the final text.
* **Settings are locked out on purpose.** ``strict_mcp_config`` and an empty
  ``setting_sources`` stop the SDK from merging ``~/.claude.json``. Without
  them it silently launches whatever ``ur-tools``/``case1`` server you have
  configured there *as well* as ours -- two servers, two robot connections.
* **Only the robot's tools are allowed.** Claude Code otherwise brings Bash,
  Write, Edit and the rest, which a kitchen assistant has no business holding.

Requires: ``pip install claude-agent-sdk`` and a logged-in ``claude`` CLI
(check with ``claude --version``; no ANTHROPIC_API_KEY is needed or used).
"""
from __future__ import annotations

import sys

from bridge import connect_tools
from conversation import STOP_TOOL, build_system_prompt

# The MCP server name the SDK is given. Tool names reaching the model are
# "mcp__{SERVER_NAME}__{tool}", which is what allowed_tools must list.
SERVER_NAME = "ur"

# Claude Code would otherwise offer its whole toolbox. Ours is a robot, not a
# workstation: nothing but the MCP tools is permitted.
BUILTIN_TOOLS_ALLOWED: list[str] = []


class ClaudeCodeAgent:
    """A voice conversation whose brain is the Claude Code subscription.

    Use as an async context manager so both the SDK session and the private
    stop channel are closed on the way out.

    Args:
        server_command: Executable that launches the MCP server.
        server_args: Its arguments (the path to ``server.py``).
        model: Model id, or None to use whatever the CLI is configured with.
        verbose: Print each tool call as it happens.
    """

    def __init__(self, server_command: str, server_args: list[str],
                 model: str | None = None, verbose: bool = True) -> None:
        self._command = server_command
        self._args = list(server_args)
        self._model = model
        self.verbose = verbose
        self._client = None
        self._stop_tools = None
        self._stop_cm = None
        self.tool_names: list[str] = []

    # --- lifecycle -------------------------------------------------------- #
    async def __aenter__(self) -> "ClaudeCodeAgent":
        from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

        # A private MCP session, used for two things: reading the tool
        # catalogue so allowed_tools can be built, and the hard stop below.
        # It is a SECOND server.py process, separate from the one the SDK
        # launches. That is deliberate -- see emergency_stop().
        self._stop_cm = connect_tools(self._command, self._args,
                                      log_path=None)
        self._stop_tools = await self._stop_cm.__aenter__()
        schemas = await self._stop_tools.openai_tools()
        self.tool_names = [s["function"]["name"] for s in schemas]

        options = ClaudeAgentOptions(
            mcp_servers={
                SERVER_NAME: {
                    "type": "stdio",
                    "command": self._command,
                    "args": self._args,
                }
            },
            strict_mcp_config=True,     # ignore ~/.claude.json MCP servers
            setting_sources=[],         # ignore user/project settings
            allowed_tools=BUILTIN_TOOLS_ALLOWED
            + [f"mcp__{SERVER_NAME}__{n}" for n in self.tool_names],
            system_prompt=build_system_prompt(self.tool_names),
            permission_mode="bypassPermissions",  # no UI here to approve calls
            model=self._model,
            max_turns=12,
        )
        self._client = ClaudeSDKClient(options=options)
        await self._client.connect()
        return self

    async def __aexit__(self, *exc) -> None:
        if self._client is not None:
            try:
                await self._client.disconnect()
            except Exception:
                pass
        if self._stop_cm is not None:
            try:
                await self._stop_cm.__aexit__(*exc)
            except Exception:
                pass

    # --- the loop --------------------------------------------------------- #
    async def ask(self, utterance: str) -> str:
        """Handle one spoken utterance and return the sentence to speak."""
        from claude_agent_sdk import (AssistantMessage, ResultMessage,
                                      TextBlock, ToolUseBlock)

        await self._client.query(utterance)
        texts: list[str] = []
        final: str | None = None

        async for message in self._client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, ToolUseBlock):
                        if self.verbose:
                            print(f"  -> {block.name}({block.input})")
                    elif isinstance(block, TextBlock) and block.text.strip():
                        texts.append(block.text.strip())
            elif isinstance(message, ResultMessage):
                if isinstance(message.result, str) and message.result.strip():
                    final = message.result.strip()
                if self.verbose and message.total_cost_usd:
                    print(f"     [{message.num_turns} turns, "
                          f"${message.total_cost_usd:.4f} equivalent]")

        # The ResultMessage carries the final answer; the text blocks are the
        # running commentary. Prefer the former, fall back to the last of the
        # latter so the operator is never met with silence.
        return final or (texts[-1] if texts else "Done.")

    # --- safety ----------------------------------------------------------- #
    async def emergency_stop(self) -> str:
        """Stop the arm immediately, with no model in the path.

        This goes down the private MCP session, not through Claude, because a
        round trip to the model is seconds and this cannot wait.

        It reaches a different ``server.py`` process than the one the SDK
        drives, so that process's ``PatternRunner`` has no job registered --
        but ``pause()`` falls through to ``_stop_stray()``, which stops the arm
        regardless of which process started the motion. The pattern loop runs
        on the robot controller, not in either Python process, which is exactly
        why stopping it from the outside works.
        """
        if self._stop_tools is None:
            return "ERROR: no stop channel; the agent is not started."
        return await self._stop_tools.call(STOP_TOOL, {})
