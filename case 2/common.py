"""Split a recording into coordinated waypoint-to-waypoint segments.

One shared definition of "a move", used by the distillation and RL stages so they
all cut the recording the same way. A segment is the full recorded path from one
waypoint to the next: all joints moving together, dynamic length.

    from analysis import Recording
    from common import segments
    for seg in segments(Recording("sim_to_real.csv")):
        print(seg.joint, seg.i0, seg.i1, seg.i2, seg.dist)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# A joint counts as "moving" while its commanded speed is above this (rad/s).
MOVING_EPS = 0.05


@dataclass
class Segment:
    """One coordinated waypoint-to-waypoint move, as indices into a Recording.

    All joints move together over ``[i0, i1]`` (motion) and settle over
    ``[i1, i2]``. ``joint`` is the widest-travel joint, which represents the
    segment in the observation; ``start``/``dest``/``dist`` are that joint's angles
    (rad). ``vel``/``acc`` are the commanded movej numbers, or None if not logged.
    """

    joint: int
    i0: int
    i1: int
    i2: int
    start: float
    dest: float
    dist: float
    vel: float = None
    acc: float = None


def segments(rec, eps: float = MOVING_EPS) -> list[Segment]:
    """Split a recording into waypoint-to-waypoint segments, one per movej.

    A ``movej`` moves the robot from wherever it is to a waypoint, so one movej is
    one segment. The controller reports which movej line is running in
    ``script_control_line`` (scl): it holds that line's number through the move,
    reads 0 on the sleeps and overhead in between, and changes to the next movej's
    line when the next move starts. So a segment is the run of rows belonging to one
    movej line, from where that line first appears to where the next movej line
    appears. For example the scl sequence

        3 0 3 3 0 3 0 | 4 4 0 4 0 4 4 0 | 5 5

    is one segment on line 3, the next on line 4, then the start of line 5 (dropped,
    see below). Zeros belong to the current segment; only a change to a different
    nonzero line ends it.

    Segmentation runs within each source script separately (rows carry their script
    name, see Recording.script), so a segment never spans two pooled recordings.
    An interior movej is bounded by the next movej. The last movej of a script is
    bounded by the recording's end and kept if it came to rest (settled) before
    then; it is dropped only if the recording was stopped mid-move (still moving at
    the end), which leaves it partial. So N completed movejs give N segments.

    ``i0`` is where the movej begins, ``i2`` where the next movej begins (or the
    recording ends), and ``target_qd`` is used only to locate ``i1`` inside the
    segment: where the motion stops and the settle window (the ring) begins. It does
    not set the boundaries, so a blend (two movejs with no stop between) still splits
    at the scl change; that segment simply has no settle window.
    """
    moving = np.abs(rec.target_qd).max(axis=1) > eps    # any joint moving each row
    scl, script = rec.scl, rec.script
    n = len(scl)
    # Split the run into contiguous same-script blocks; segment each on its own.
    starts = [0] + [i for i in range(1, n) if script[i] != script[i - 1]] + [n]
    out = []
    for a, b in zip(starts, starts[1:]):
        # Onset of each distinct movej line in this block (0s are transparent).
        bounds, last = [], 0
        for i in range(a, b):
            if scl[i] != 0 and scl[i] != last:
                bounds.append(i)
                last = scl[i]
        for k, i0 in enumerate(bounds):
            i2 = bounds[k + 1] if k + 1 < len(bounds) else b   # next movej, or end
            mv = np.nonzero(moving[i0:i2])[0]        # motion stops when target_qd settles
            i1 = i0 + (int(mv[-1]) + 1 if len(mv) else 0)
            if k + 1 == len(bounds) and i1 >= i2:
                continue                            # last movej never settled: partial, drop
            if i1 - i0 < 3:
                continue                            # too few motion samples: not a real move
            j = int(np.argmax(np.abs(rec.target_q[i1] - rec.target_q[i0])))
            start, dest = float(rec.target_q[i0, j]), float(rec.target_q[i1, j])
            vel = float(rec.vel_cmd[i0]) if rec.vel_cmd is not None else None
            acc = float(rec.acc_cmd[i0]) if rec.acc_cmd is not None else None
            out.append(Segment(j, i0, i1, i2, start, dest, abs(dest - start), vel, acc))
    return out
