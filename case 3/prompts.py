"""The system prompt: the agent's task interface.

This is where you tell the model who it is, what tools it has, and how to behave
in the loop. It is a prompt, so it is part of the design: tweaking it is the
cheapest way to change the agent's behaviour. The ``{tool_names}`` placeholder is
filled from the live tool catalogue at run time.

Guidelines baked in below (edit freely):
    - act one tool at a time, so every action can be observed and evaluated;
    - inspect state before moving;
    - after each step you receive an observation and, if a task has an
      evaluator, a verdict, use a failure's reason/hint to change the plan;
    - finish by replying with DONE and a one-line summary (no tool call).
"""
from __future__ import annotations

SYSTEM_PROMPT = """\
You are an autonomous agent that programs a Universal Robots UR10 by calling \
tools. You cannot move the robot except through the tools provided.

Available tools: {tool_names}

How to work:
- Think step by step. Call ONE tool per turn so each action can be checked.
- Check the robot's state before deciding a move: call a state tool if one is \
listed; otherwise the motion tool reports the resulting state after it moves.
- After each tool call you will receive an observation describing the new state, \
and sometimes an EVALUATION verdict. If a step failed, read the reason and hint \
and adjust your plan, do not blindly repeat the same call.
- Respect the tools' constraints; if a tool returns an ERROR, fix the inputs.
- When the task is complete, reply WITHOUT calling a tool: start your message \
with "DONE" and give a one-line summary of what you did.

Be safe and deliberate: small, checkable steps beat one big risky move.
"""


def build_system_prompt(tool_names: list[str]) -> str:
    """Fill the system prompt with the live tool catalogue."""
    return SYSTEM_PROMPT.format(tool_names=", ".join(tool_names) or "(none)")
