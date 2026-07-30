"""LLM client: one OpenAI-compatible chat call with tool calling.

The agent's "brain". Any OpenAI-compatible endpoint works, so you bring your own
model. Free options: GLM (z-ai/glm-5.2) via NVIDIA NIM (free API key), Z.AI
direct, Modal.com, or a local server. See ../llm-client for setup. Configure it
with three environment variables:

    AGENT_BASE_URL   e.g. https://integrate.api.nvidia.com/v1
    AGENT_API_KEY    your key
    AGENT_MODEL      the model id your endpoint serves (e.g. z-ai/glm-5.2)

``LLMClient.chat`` sends the running conversation plus the tool catalogue and
returns one :class:`Turn`: either free text, or one/more structured tool calls.
Parsing the model's raw output into that clean shape is done here so the
orchestrator (agent.py) stays readable. This file is provided and rarely needs
changing.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass


@dataclass
class ToolCall:
    """One tool the model wants to run: a name plus parsed JSON arguments."""

    id: str
    name: str
    arguments: dict


@dataclass
class Turn:
    """One assistant turn: optional text and any tool calls it emitted."""

    text: str | None
    tool_calls: list[ToolCall]

    def as_message(self) -> dict:
        """Rebuild the assistant message to append back to the conversation.

        The tool calls must be echoed verbatim so the follow-up ``tool`` messages
        (keyed by ``tool_call_id``) line up with them.
        """
        msg: dict = {"role": "assistant", "content": self.text or ""}
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": c.id,
                    "type": "function",
                    "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                }
                for c in self.tool_calls
            ]
        return msg


class LLMClient:
    """Thin wrapper over an OpenAI-compatible chat-completions endpoint."""

    def __init__(self, model: str | None = None, base_url: str | None = None,
                 api_key: str | None = None, temperature: float = 0.0):
        from openai import OpenAI

        self.model = model or os.environ.get("AGENT_MODEL")
        if not self.model:
            raise ValueError(
                "No model set. Pass model=... or set the AGENT_MODEL env var."
            )
        self.temperature = temperature
        self.client = OpenAI(
            base_url=base_url or os.environ.get("AGENT_BASE_URL"),
            api_key=api_key or os.environ.get("AGENT_API_KEY", "not-needed"),
        )

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> Turn:
        """Send the conversation (and tool catalogue) and return one turn.

        Args:
            messages: OpenAI-format message dicts (system, user, assistant, tool).
            tools: OpenAI-format tool schemas, or None to forbid tool use.

        Returns:
            A :class:`Turn` with the assistant's text and parsed tool calls.
        """
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
            calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
        return Turn(text=msg.content, tool_calls=calls)
