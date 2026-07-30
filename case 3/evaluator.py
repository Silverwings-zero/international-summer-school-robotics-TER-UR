"""Evaluator: decide whether the task is done, and say why if not.

This is the piece that lets the agent SELF-CORRECT instead of blindly retrying.
After each step the orchestrator calls the evaluator with the current robot
state; it returns a :class:`Verdict` with success, a reason, and an optional
hint. On failure that reason/hint is fed back into the next reasoning step.

A generic distance-to-pose evaluator (:class:`ReachedPose`) is provided as a
worked example. WRITING TASK-SPECIFIC EVALUATORS IS THE SILVER TIER: e.g. "did
the TCP trace the requested shape", "is the part in the bin". An evaluator is
just a callable ``state -> Verdict``; see ``PathTraced`` for the stub to fill in.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from serializer import TCP_KEYS, _first, distance_m


@dataclass
class Verdict:
    """The evaluator's judgement of the current state.

    Attributes:
        success: True when the goal is met.
        reason: Short explanation of the judgement (shown to the agent).
        hint: Optional nudge toward fixing a failure (shown to the agent).
    """

    success: bool
    reason: str
    hint: str = ""


class Evaluator(Protocol):
    """Anything callable as ``state -> Verdict``."""

    def __call__(self, state: dict) -> Verdict: ...


@dataclass
class ReachedPose:
    """Success when the TCP is within ``tol_m`` of a target position.

    Worked example, good for goals like "move to home" or "go to this point".
    """

    target_xyz: tuple[float, float, float]
    tol_m: float = 0.01

    def __call__(self, state: dict) -> Verdict:
        tcp = _first(state, TCP_KEYS)
        if not isinstance(tcp, (list, tuple)) or len(tcp) < 3:
            return Verdict(False, "no TCP pose in the state",
                           "call the robot-state tool first")
        d = distance_m(tcp, self.target_xyz)
        if d <= self.tol_m:
            return Verdict(True, f"TCP within {d * 1000:.0f} mm of target")
        return Verdict(
            False,
            f"TCP is {d * 1000:.0f} mm from target",
            f"move toward {tuple(round(v, 3) for v in self.target_xyz)} m",
        )


@dataclass
class PathTraced:
    """STUB (Silver): success when the TCP path matches a requested shape.

    Fill this in for a tracing task. You will need the orchestrator to record the
    TCP pose after each move (accumulate a list of visited points), then compare
    that path to ``self.waypoints`` here, e.g. max deviation below a tolerance.
    """

    waypoints: list[tuple[float, float, float]]
    tol_m: float = 0.01

    def __call__(self, state: dict) -> Verdict:
        raise NotImplementedError(
            "Implement PathTraced for the Silver tier: compare the recorded TCP "
            "path against self.waypoints and return a Verdict."
        )
