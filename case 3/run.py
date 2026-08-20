"""Entry point: run the agent on a task against an MCP server.

Bronze example (move to home), against the Case 1 server:

    export AGENT_BASE_URL=... AGENT_API_KEY=... AGENT_MODEL=...
    python run.py "move the robot to its home pose" --goal-home

Point it at UR's pre-built server, or your own, with --server:

    python run.py "trace a 10 cm square" --server "python /path/to/server.py"

The default server is the Case 1 server next door. It exposes a motion tool
(move_robot_to_position) that reports the resulting state; the agent observes
from that. A server may instead expose a separate robot-state tool, which the
agent will prefer if present (see STATE_TOOL_HINTS in agent.py).
"""
from __future__ import annotations

import argparse
import asyncio
import os
import shlex
import sys

from agent import run_agent
from evaluator import ReachedPose

# Default MCP server: the Case 1 server in the sibling folder. Launch it with the
# same interpreter running this script (sys.executable) so it inherits this
# virtualenv and works on machines that have no bare `python` on PATH.
DEFAULT_SERVER = f'{shlex.quote(sys.executable)} {shlex.quote(os.path.join(os.path.dirname(__file__), "..", "case 1", "server.py"))}'

# UR10 home pose used by the --goal-home convenience evaluator (TCP xyz, metres).
# Forward kinematics of the PolyScope X UR10 sim at the kitchen home joints
# [0, -90, 90, -90, -90, 0] deg (ur_client.HOME_Q_RAD). A UR5e's shorter links
# put its home TCP elsewhere; likewise if a tool/TCP offset is configured --
# re-measure and update this.
HOME_TCP_XYZ = (-0.691, -0.174, 0.677)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("task", help="natural-language goal for the agent")
    ap.add_argument("--server", default=DEFAULT_SERVER,
                    help="command that launches the MCP server (quoted)")
    ap.add_argument("--max-steps", type=int, default=12, help="loop step budget")
    ap.add_argument("--goal-home", action="store_true",
                    help="use a ReachedPose evaluator targeting the home pose")
    args = ap.parse_args()

    cmd, *cmd_args = shlex.split(args.server)
    evaluator = ReachedPose(HOME_TCP_XYZ) if args.goal_home else None

    result = asyncio.run(run_agent(
        args.task,
        server_command=cmd,
        server_args=cmd_args,
        evaluator=evaluator,
        max_steps=args.max_steps,
    ))

    print("\n" + "=" * 60)
    print(f"{'SUCCESS' if result.success else 'STOPPED'} after {result.steps} steps")
    print(result.summary)


if __name__ == "__main__":
    main()
