"""The orchestrator: the reason -> act -> observe loop.

This is the heart of Case 3 and the file worth reading closely. It wires the
pieces together into one loop:

    reason   ask the LLM for the next step, given the task + latest observation
    act      run the tool call it emitted, via the MCP server (the robot)
    observe  read the new state, serialize it, and (optionally) evaluate it
    repeat   feed the observation back as context, until success or the budget

The loop is provided and works end to end (Bronze = run it and read the log).
The extension points for higher tiers are:
    - a task-specific Evaluator (Silver)          -> evaluator.py
    - smarter failure handling / re-planning (Gold) -> the observe/feedback block
    - a world model of named objects (Diamond)    -> extend the observation
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from evaluator import Evaluator, Verdict
from llm import LLMClient
from mcp_client import connect_tools
from prompts import build_system_prompt
from serializer import parse_tool_text, serialize_state

# Tool-name substrings we treat as "read the robot state" (for the observe step).
STATE_TOOL_HINTS = ("get_robot_state", "robot_state", "get_state")


@dataclass
class AgentResult:
    """Outcome of a run."""

    success: bool
    steps: int
    summary: str
    transcript: list[dict] = field(default_factory=list)


def _find_state_tool(schemas: list[dict]) -> str | None:
    """Pick the tool that reports robot state, for the observe step."""
    names = [s["function"]["name"] for s in schemas]
    for name in names:
        if any(h in name for h in STATE_TOOL_HINTS):
            return name
    return None


def _log(step: int, phase: str, text: str) -> None:
    """Print one line of the loop so the run is legible (Bronze reads this)."""
    print(f"[{step:02d} {phase:<7}] {text}")


async def _read_state(tools, state_tool: str | None) -> dict:
    """Call the state tool and parse its result into a dict (empty if none)."""
    if not state_tool:
        return {}
    try:
        return parse_tool_text(await tools.call(state_tool, {}))
    except Exception as exc:  # keep the loop alive; the model sees no state
        _log(0, "observe", f"state read failed: {exc}")
        return {}


async def run_agent(
    task: str,
    *,
    server_command: str,
    server_args: list[str],
    evaluator: Evaluator | None = None,
    max_steps: int = 12,
    llm: LLMClient | None = None,
) -> AgentResult:
    """Run the agent on ``task`` against an MCP server until done or out of steps.

    Args:
        task: The natural-language goal (e.g. "move the robot to its home pose").
        server_command: Executable launching the MCP server (e.g. "python").
        server_args: Arguments to it (e.g. ["../case 1/server.py"]).
        evaluator: Optional task-specific success check. Without one, the loop
            stops when the model declares it is done.
        max_steps: Hard cap on loop iterations (the step budget).
        llm: An :class:`LLMClient`; one is built from env vars if omitted.

    Returns:
        An :class:`AgentResult` with success, step count, and the transcript.
    """
    llm = llm or LLMClient()

    async with connect_tools(server_command, server_args) as tools:
        schemas = await tools.openai_tools()
        state_tool = _find_state_tool(schemas)
        tool_names = [s["function"]["name"] for s in schemas]

        messages: list[dict] = [
            {"role": "system", "content": build_system_prompt(tool_names)},
            {"role": "user", "content": task},
        ]
        _log(0, "task", task)

        for step in range(1, max_steps + 1):
            # --- REASON: ask the model for the next step -------------------- #
            turn = await asyncio.to_thread(llm.chat, messages, schemas)
            messages.append(turn.as_message())

            # No tool call => the model believes it is finished (or is stuck).
            if not turn.tool_calls:
                text = (turn.text or "").strip()
                _log(step, "reason", text or "(no action, stopping)")
                done = text.upper().startswith("DONE")
                return AgentResult(done, step, text, messages)

            _log(step, "reason", turn.text.strip() if turn.text else "(tool call)")

            # --- ACT: run each requested tool via the MCP server ------------ #
            last_result = "{}"
            for call in turn.tool_calls:
                _log(step, "act", f"{call.name}({call.arguments})")
                result = await tools.call(call.name, call.arguments)
                last_result = result
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": result}
                )
                _log(step, "act", f"-> {result}")

            # --- OBSERVE: read state, serialize, evaluate ------------------- #
            # Prefer a dedicated state tool. If the server exposes none (the Case 1
            # baseline does not -- its move tool reports the resulting state
            # itself), fall back to whatever the acting tool just returned.
            if state_tool:
                state = await _read_state(tools, state_tool)
            else:
                state = parse_tool_text(last_result)
            summary = serialize_state(state)
            _log(step, "observe", summary)

            feedback = f"Observation: {summary}"
            if evaluator is not None:
                verdict: Verdict = evaluator(state)
                if verdict.success:
                    _log(step, "observe", f"SUCCESS: {verdict.reason}")
                    return AgentResult(True, step, verdict.reason, messages)
                note = f"not done yet, {verdict.reason}"
                if verdict.hint:
                    note += f" (hint: {verdict.hint})"
                feedback += f"\nEVALUATION: {note}"
                _log(step, "observe", f"eval: {note}")

            # Feed the observation back as the next turn's context.
            messages.append({"role": "user", "content": feedback})

        _log(max_steps, "stop", "step budget exhausted")
        return AgentResult(False, max_steps, "step budget exhausted", messages)
