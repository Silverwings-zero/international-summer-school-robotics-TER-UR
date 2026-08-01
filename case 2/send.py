"""Send a URScript file to a UR controller (or URSim) and record the run.

Opens the RTDE stream, pushes the script to the secondary client interface
(port 30002) so the controller runs it, logs every sample to a CSV, and stops
when the program finishes.

    python send.py --robot-ip 127.0.0.1 --script scripts/shoulder_swing.script

--loop N repeats the motion N times then ends; without it the script runs once:

    python send.py --script scripts/shoulder_swing.script --loop 10 --out run.csv

    python send.py --script scripts/shoulder_swing.optimized.script --loop 10

--path streams a servoj path (a CSV of joint setpoints) instead of a script, at
a fixed time per row:

    python send.py --path scripts/shoulder_swing.path --dt 0.008 --out path_run.csv

Auto-stop: the wrapper flips a float register to 1 on the program's last line,
and recording stops when that register reads 1, so the program must end for it
to fire (run-once or --loop N both end). Ctrl-C stops and keeps the data so far.

Requires Remote Control mode on the robot/URSim; otherwise the controller
accepts the socket but does not run the script.

Pure Python standard library. Recording uses record.py's RTDE code.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import socket

from record import build_recipe, open_stream, parse_registers, record_stream
from utils import load_script

SCRIPT_PORT = 30002       # UR secondary client interface: accepts URScript programs
RUNTIME_PLAYING = 2       # RTDE runtime_state while a program is running
START_TIMEOUT_S = 5.0     # how long to wait for the program to start playing
DONE_REG = 3              # output float register we flip to 1 when the program ends
DONE_FIELD = f"output_double_register_{DONE_REG}"
SERVO_LOOKAHEAD = 0.1     # servoj lookahead_time (s): smooths the streamed path
SERVO_GAIN = 300          # servoj gain: how hard it tracks each setpoint


def _is_assignment(line: str) -> bool:
    """True for a plain variable assignment `name = ...` (not a call `foo(...)`)."""
    return re.match(r"\s*[A-Za-z_]\w*\s*=", line) is not None


def wrap_program(text: str, loop: int | None = None) -> str:
    """Wrap the script's statements in a `def prog(): ... end` program.

    The UR script interface runs a program, not loose top-level statements. Text
    already starting with `def` is returned unchanged (and does not get the loop
    or auto-stop below).

    Variable assignments are hoisted to initialize once, before any loop:
    URScript will not accept a variable first assigned inside a loop, and every
    variable must be initialized before use. The remaining (motion) statements go
    in the loop. ``loop`` repeats the motion that many times; ``None`` runs it
    once. URScript has no ``range``, so the repeat is a while-counter.

    A script on the secondary interface keeps ``runtime_state`` at PLAYING even
    after it returns, so "finished" cannot be read from it. Instead the program
    flips float register DONE_REG to 1 on its last line and the recorder stops
    on it.
    """
    if text.lstrip().startswith("def "):
        return text

    init, motion, seen_motion = [], [], False
    for ln in text.splitlines():
        s = ln.strip()
        if _is_assignment(s):
            init.append(s)                        # initialize once, before the loop
        elif s == "" or s.startswith("#"):
            (motion if seen_motion else init).append(s)
        else:
            seen_motion = True
            motion.append(s)                      # movej / sleep / register writes

    # Each block body is indented under its `def`/`while` header.
    IND = "  "
    out = ["def prog():", IND + f"write_output_float_register({DONE_REG}, 0)"]
    out += [(IND + l if l else "") for l in init]
    if loop is None:
        out += [(IND + l if l else "") for l in motion]
    else:
        out += [IND + "loop_count = 0", IND + f"while (loop_count < {loop}):"]
        out += [(2 * IND + l if l else "") for l in motion]
        out += [2 * IND + "loop_count = loop_count + 1", IND + "end"]
    out += [IND + f"write_output_float_register({DONE_REG}, 1)", "end"]
    return "\n".join(out) + "\n"


def wrap_path(rows, dt: float, loop: int | None = None) -> str:
    """Wrap a list of joint setpoints into a servoj-streaming program.

    Each row is six joint values, optionally followed by a per-row servoj time
    (a 7th column); rows without it use ``dt``. The program streams the setpoints
    with ``servoj``. Same done-flag auto-stop as ``wrap_program``; ``loop``
    repeats the whole path.
    """
    IND = "  "
    body = []
    for r in rows:
        q, dt_r = r[:6], (r[6] if len(r) > 6 else dt)
        body.append(f"servoj([{', '.join(f'{v:.6f}' for v in q)}], 0, 0, "
                    f"{dt_r:.6f}, {SERVO_LOOKAHEAD}, {SERVO_GAIN})")
    out = ["def prog():", IND + f"write_output_float_register({DONE_REG}, 0)"]
    if loop is None:
        out += [IND + b for b in body]
    else:
        out += [IND + "loop_count = 0", IND + f"while (loop_count < {loop}):"]
        out += [2 * IND + b for b in body]
        out += [2 * IND + "loop_count = loop_count + 1", IND + "end"]
    out += [IND + f"write_output_float_register({DONE_REG}, 1)", "end"]
    return "\n".join(out) + "\n"


def load_path(path: str) -> list:
    """Read a path CSV of joint setpoints (six columns q0..q5) into a list of rows."""
    rows = []
    with open(path, newline="") as f:
        for r in csv.reader(f):
            if not r:
                continue
            try:
                rows.append([float(x) for x in r])
            except ValueError:
                continue                   # skip a header line
    return rows


def send_program(host: str, program: str, port: int = SCRIPT_PORT):
    """Push a ready-made URScript program to the controller, which runs it."""
    if not program.endswith("\n"):
        program += "\n"                    # controller runs on the trailing newline
    with socket.create_connection((host, port), timeout=5) as s:
        s.sendall(program.encode())


def _done_check():
    """Build a per-sample stop test for the sent program's done-flag register.

    Returns a closure ``check(smp, t)`` for ``record.record_stream`` that returns
    a stop reason once the program ends:
      - program starts (runtime_state = PLAYING) and the wrapper resets the flag
        to 0 (armed = motion running under our program),
      - wrapper flips the flag to 1 on its last line -> finished,
      - runtime leaves PLAYING after it started -> aborted,
      - never starts within START_TIMEOUT_S -> not in Remote Control mode.
    """
    state = {"started": False, "armed": False}

    def check(smp, t):
        playing = smp["runtime_state"] == RUNTIME_PLAYING
        flag = smp[DONE_FIELD]
        if playing:
            state["started"] = True
        if state["started"] and flag < 0.5:        # our program reset it: motion running
            state["armed"] = True
        if state["armed"] and flag >= 0.5:         # prog flipped it on its last line
            return "program finished"
        if state["started"] and not playing:       # runtime dropped: script aborted
            return "program stopped (aborted? check URSim Log Messages)"
        if not state["started"] and t > START_TIMEOUT_S:
            return "program never started (is the robot in Remote Control mode?)"
        return None

    return check


def run_and_record(host, program, out, hz, recipe, port):
    """Send a ready-made program, record RTDE to `out`, stop when it finishes.

    Returns (n_samples, reason). Uses ``record.record_stream`` for the RTDE
    read/write loop, sending the program first and passing a done-flag stop
    test. Ctrl-C or a dropped stream also stop.
    """
    stream = open_stream(host, hz, recipe)
    send_program(host, program, port)      # stream open first, so we catch the start
    print(f"sent + recording {host}:{port} -> {out}  (Ctrl-C to stop)")
    try:
        return record_stream(stream, out, recipe, stop_check=_done_check())
    finally:
        stream.close()


def record_run(host, script, out, hz=125.0, registers=("1", "vel", "2", "acc"),
               loop=None, port=SCRIPT_PORT):
    """Run a URScript file on the robot once (or ``loop`` times) and record it.

    Loads the script, wraps it as a program, builds the RTDE recipe (the
    requested registers plus the done-flag), and records to ``out``.
    """
    regs = parse_registers(list(registers)) + [(DONE_REG, "_done")]
    recipe = build_recipe(regs)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    program = wrap_program(load_script(script), loop)
    return run_and_record(host, program, out, hz, recipe, port)


def record_path(host, path, out, dt=0.008, hz=125.0, loop=None, port=SCRIPT_PORT):
    """Stream a servoj path file on the robot once (or ``loop`` times) and record it.

    Like ``record_run`` but for a path CSV: wrap the setpoints as a servoj
    program at ``dt`` s per row. Only the done-flag register is logged (a servoj
    path carries no vel/acc registers).
    """
    recipe = build_recipe([(DONE_REG, "_done")])
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    program = wrap_path(load_path(path), dt, loop)
    return run_and_record(host, program, out, hz, recipe, port)


def main():
    ap = argparse.ArgumentParser(description="Send a URScript to a UR robot and record it.")
    ap.add_argument("--robot-ip", default="127.0.0.1",
                    help="UR controller IP (default 127.0.0.1, i.e. local URSim)")
    ap.add_argument("--script", default="scripts/shoulder_swing.script",
                    help="URScript file to run on the robot")
    ap.add_argument("--path", default=None,
                    help="servoj path CSV to stream instead of --script")
    ap.add_argument("--dt", type=float, default=0.008,
                    help="servoj time per row for --path (s, default 0.008)")
    ap.add_argument("--out", default="run.csv",
                    help="output CSV path (default: run.csv in the current folder)")
    ap.add_argument("--hz", type=float, default=125.0, help="sample rate (default 125)")
    ap.add_argument("--float-register", nargs="+", metavar="IDX NAME",
                    default=["1", "vel", "2", "acc"],
                    help="log output float registers, e.g. 1 vel 2 acc")
    ap.add_argument("--port", type=int, default=SCRIPT_PORT,
                    help="script interface port (30002 secondary, 30001 primary)")
    ap.add_argument("--loop", type=int, default=None,
                    help="repeat N times then stop (default: run once)")
    args = ap.parse_args()

    if args.path:                            # stream a servoj path
        n, stop = record_path(args.robot_ip, args.path, args.out, args.dt,
                              args.hz, args.loop, args.port)
    else:                                    # run a URScript
        n, stop = record_run(args.robot_ip, args.script, args.out, args.hz,
                             args.float_register, args.loop, args.port)
    print(f"\nwrote {n} samples to {args.out}" + (f"  ({stop})" if stop else ""))


if __name__ == "__main__":
    main()
