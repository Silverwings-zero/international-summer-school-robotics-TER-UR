"""The conversation: one persistent chat that can call the robot's tools.

Different in shape from the Case 3 agent, on purpose. That one is *task*
oriented -- one goal, a step budget, then it exits. A kitchen assistant is
*conversation* oriented: many short exchanges, memory carried between them, and
the robot left running between utterances.

One utterance becomes one :meth:`VoiceAgent.ask` call, which internally runs a
short reason/act loop (the model may need several tool calls to answer) and
returns the sentence to speak.
"""
from __future__ import annotations

import asyncio

from bridge import LLMClient, ToolBridge

# The tool that must always stop the arm. ``pause_motion`` is the right panic
# button rather than ``finish_motion``: it preserves the anchor and phase so the
# motion can be resumed, and PatternRunner.pause() falls through to
# _stop_stray() when no pattern is registered, so it stops the robot even if the
# arm is running a program left over from an earlier session.
STOP_TOOL = "pause_motion"

# How many tool calls the model may chain before it must answer. Enough for
# "check state, then move, then confirm"; low enough that a confused model
# cannot flail at the robot for a minute while someone waits to talk.
MAX_TOOL_STEPS = 6

# Conversation window. Tool results from this server are verbose (trajectory
# logs, joint dumps), so an unbounded history hits the context limit and slows
# every turn. Older exchanges are dropped, never the system prompt.
MAX_HISTORY_MESSAGES = 40


SYSTEM_PROMPT = """\
You are the voice assistant of a Universal Robots UR10e cobot working in a \
kitchen. You are speaking with a person through a microphone and a speaker, \
and you command the arm ONLY through the tools available to you.

Tools: {tool_names}

How to reply (your answer is READ ALOUD):
- One or two short sentences. Never lists, never markdown, never JSON, never \
tool names.
- Reply in the language the user speaks to you in.
- Say numbers the way a person would, rounded: "about twenty centimetres", \
not "0.1987 m".
- Say what you did or what you are about to do, not how you did it.

How to act:
- Check the robot's state before moving it if you do not know where it is.
- If the request is ambiguous, ASK before moving. In particular, if the user \
says "save this position" without specifying, ask whether they mean the \
Cartesian tool pose or the joint configuration.
- If a tool returns ERROR, read the explanation, fix the inputs and try again. \
Do not repeat the identical call. If you still cannot, explain the problem in \
one sentence.
- Stirring requires the utensil to be already in the centre of the container \
at working depth. If it is not, say so and ask for it to be positioned first.
- If the user says "stop", "halt" or "wait", call the stop tool immediately, \
before replying.
- Ask for confirmation before a large movement or one near people.

Safety: a validation layer rejects dangerous commands and tells you why. Trust \
that message and relay it to the user instead of forcing the move.
"""

# Spoken when the model is cut off by the tool-step budget, or returns nothing.
FALLBACK_REPLY = "Done."
FALLBACK_FAILED = "I could not complete that."
STOP_ACKNOWLEDGED = "Stopped."


def build_system_prompt(tool_names: list[str]) -> str:
    """Fill the spoken-assistant prompt with the live tool catalogue."""
    return SYSTEM_PROMPT.format(tool_names=", ".join(tool_names) or "(none)")


class VoiceAgent:
    """A running conversation with the robot behind it.

    Args:
        tools: The live MCP bridge. Must stay connected for the whole session.
        schemas: Tool catalogue in OpenAI format, from ``tools.openai_tools()``.
        llm: The chat client.
        verbose: Print every tool call and result. On during development, off
            during the demo so the transcript stays readable.
    """

    def __init__(self, tools: ToolBridge, schemas: list[dict],
                 llm: LLMClient, verbose: bool = True) -> None:
        self.tools = tools
        self.schemas = schemas
        self.llm = llm
        self.verbose = verbose
        self.tool_names = [s["function"]["name"] for s in schemas]
        self.messages: list[dict] = [
            {"role": "system", "content": build_system_prompt(self.tool_names)}
        ]

    # --- history ---------------------------------------------------------- #
    def _trim(self) -> None:
        """Drop the oldest exchanges, keeping the history structurally valid.

        An assistant message carrying ``tool_calls`` must keep the ``tool``
        messages that answer it, or the endpoint rejects the request. So the
        cut always lands on a ``user`` message, never in the middle of a
        tool round-trip.
        """
        if len(self.messages) <= MAX_HISTORY_MESSAGES:
            return
        system, rest = self.messages[0], self.messages[1:]
        cut = len(rest) - MAX_HISTORY_MESSAGES
        while cut < len(rest) and rest[cut].get("role") != "user":
            cut += 1
        self.messages = [system] + rest[cut:]

    # --- the loop --------------------------------------------------------- #
    async def ask(self, utterance: str) -> str:
        """Handle one spoken utterance and return the sentence to speak."""
        self.messages.append({"role": "user", "content": utterance})
        self._trim()

        for _step in range(MAX_TOOL_STEPS):
            # The OpenAI SDK is synchronous; keep the event loop free so the
            # MCP subprocess pipes are still serviced during the call.
            turn = await asyncio.to_thread(self.llm.chat, self.messages,
                                           self.schemas)
            self.messages.append(turn.as_message())

            if not turn.tool_calls:
                return (turn.text or "").strip() or FALLBACK_REPLY

            for call in turn.tool_calls:
                if self.verbose:
                    print(f"  -> {call.name}({call.arguments})")
                result = await self.tools.call(call.name, call.arguments)
                if self.verbose:
                    print(f"     {result[:300]}")
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result,
                })

        # Budget spent: make the model answer with tools switched off, so the
        # user always gets a spoken reply instead of silence.
        turn = await asyncio.to_thread(self.llm.chat, self.messages, None)
        self.messages.append(turn.as_message())
        return (turn.text or "").strip() or FALLBACK_FAILED

    # --- safety ----------------------------------------------------------- #
    async def emergency_stop(self) -> str:
        """Stop the arm immediately, bypassing the model.

        Called by the hard-stop key. The LLM is not in this path: a round trip
        to a cloud endpoint is seconds, and the whole point is that the arm
        stops now. The result is appended to the history so the model knows
        the robot was stopped under it.
        """
        if STOP_TOOL not in self.tool_names:
            return f"Stop tool '{STOP_TOOL}' not exposed by the server."
        result = await self.tools.call(STOP_TOOL, {})
        self.messages.append({
            "role": "user",
            "content": "[The operator pressed the emergency stop. The robot "
                       f"has been stopped. Result: {result}]",
        })
        return result
