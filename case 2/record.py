"""Passively record a UR's RTDE state to a CSV.

Opens an RTDE stream to the controller and writes every sample to a CSV until
Ctrl-C. It does not move the robot; it logs whatever drives the arm (a PolyScope
program, a script sent separately).

    python record.py --robot-ip 10.54.5.147 --out run.csv

A running URScript can publish values with ``write_output_float_register(i,
value)``. Name those registers on the command line and each becomes a CSV
column. For a program doing ``write_output_float_register(1, vel)`` and
``(2, acc)``:

    python record.py --robot-ip 10.54.5.147 --float-register 1 vel 2 acc

Rows are flushed as they arrive, so Ctrl-C or a dropped connection keeps the
data recorded so far. The CSV is the schema ``analysis.Recording`` loads.

Channels per sample (joint vectors are 6 wide, base..wrist3):
    positions   target_q  / actual_q                       (rad)
    velocities  target_qd / actual_qd                      (rad/s)
    accel       target_qdd (commanded)  +  actual_TCP_acceleration
                RTDE has NO per-joint measured acceleration; only target_qdd.
    currents    target_current / actual_current            (A)
    torque      target_moment  / actual_current_as_torque  (Nm)
    control     joint_control_output
    TCP         target/actual pose + speed, actual force + acceleration
    state       robot_mode, runtime_state, script_control_line
    registers   whatever --float-register names

Pure Python standard library, one TCP socket to the RTDE interface (30004).
Field names and types are from the UR RTDE guide.
"""
from __future__ import annotations

import argparse
import csv
import os
import socket
import struct
import time

from utils import N_JOINTS, TIME_COL

RTDE_PORT = 30004

# RTDE protocol command bytes (see the UR RTDE guide).
_RTDE_REQUEST_PROTOCOL_VERSION = 86  # 'V'
_RTDE_SETUP_OUTPUTS = 79             # 'O'
_RTDE_START = 83                     # 'S'
_RTDE_DATA_PACKAGE = 85              # 'U'

# Fixed RTDE recipe: (field, kind), in wire order. Kinds and their byte sizes:
#   d = DOUBLE (8)   v6 = VECTOR6D, 6 doubles (48)   i32 = INT32 (4)   u32 = UINT32 (4)
RECIPE = [
    ("timestamp", "d"),
    ("target_q", "v6"), ("target_qd", "v6"), ("target_qdd", "v6"),
    ("target_current", "v6"), ("target_moment", "v6"),
    ("actual_q", "v6"), ("actual_qd", "v6"), ("actual_current", "v6"),
    ("actual_current_as_torque", "v6"), ("joint_control_output", "v6"),
    ("target_TCP_pose", "v6"), ("target_TCP_speed", "v6"),
    ("actual_TCP_pose", "v6"), ("actual_TCP_speed", "v6"),
    ("actual_TCP_force", "v6"), ("actual_TCP_acceleration", "v6"),
    ("robot_mode", "i32"), ("runtime_state", "u32"), ("script_control_line", "u32"),
]
_KIND_SIZE = {"d": 8, "v6": 48, "i32": 4, "u32": 4}


def build_recipe(registers):
    """Fixed recipe plus one DOUBLE per requested output float register.

    ``registers`` is a list of ``(index, column_name)``. Returns entries of
    ``(rtde_field, kind, csv_label)``; ``csv_label`` is None for the fixed fields.
    """
    recipe = [(name, kind, None) for name, kind in RECIPE]
    for idx, name in registers:
        recipe.append((f"output_double_register_{idx}", "d", name))
    return recipe


def csv_columns(recipe):
    """Flat header: t, then each field (v6 expands to label0..label5)."""
    cols = [TIME_COL]
    for name, kind, label in recipe:
        col = label or name
        cols += [f"{col}{i}" for i in range(N_JOINTS)] if kind == "v6" else [col]
    return cols


def _rtde_send(s, cmd, payload=b""):
    s.sendall(struct.pack(">HB", 3 + len(payload), cmd) + payload)


def _recv_exact(s, n):
    """Read exactly ``n`` bytes; raise if the controller closes the stream."""
    buf = b""
    while len(buf) < n:
        chunk = s.recv(n - len(buf))
        if not chunk:
            raise ConnectionError(
                "RTDE stream closed by the controller (it drops clients that "
                "cannot keep up, try a lower --hz)."
            )
        buf += chunk
    return buf


def _rtde_recv(s):
    size, cmd = struct.unpack(">HB", _recv_exact(s, 3))
    return cmd, _recv_exact(s, size - 3)


def open_stream(host, hz, recipe):
    """Connect, subscribe to ``recipe``, and start streaming."""
    fields = ",".join(name for name, _, _ in recipe)
    s = socket.create_connection((host, RTDE_PORT), timeout=5)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    _rtde_send(s, _RTDE_REQUEST_PROTOCOL_VERSION, struct.pack(">H", 2))
    _rtde_recv(s)  # version reply
    _rtde_send(s, _RTDE_SETUP_OUTPUTS, struct.pack(">d", hz) + fields.encode())
    _, reply = _rtde_recv(s)  # recipe id byte + variable types CSV
    types = reply[1:].decode(errors="replace")
    bad = [f for f, t in zip(fields.split(","), types.split(","))
           if t.strip() in ("NOT_FOUND", "IN_USE")]
    if bad:
        s.close()
        raise RuntimeError(f"RTDE rejected fields {bad} (reply: {types!r}).")
    _rtde_send(s, _RTDE_START)
    _rtde_recv(s)  # start reply
    return s


def read_sample(s, recipe):
    """Read the next data package into ``{field: value}``."""
    cmd, body = _rtde_recv(s)
    while cmd != _RTDE_DATA_PACKAGE:  # skip control replies
        cmd, body = _rtde_recv(s)
    off = 1  # first byte is the recipe id
    out = {}
    for name, kind, _ in recipe:
        chunk = body[off:off + _KIND_SIZE[kind]]
        if kind == "v6":
            out[name] = struct.unpack(">6d", chunk)
        elif kind == "d":
            out[name] = struct.unpack(">d", chunk)[0]
        elif kind == "i32":
            out[name] = struct.unpack(">i", chunk)[0]
        else:  # u32
            out[name] = struct.unpack(">I", chunk)[0]
        off += _KIND_SIZE[kind]
    return out


def sample_row(t, smp, recipe):
    """Flatten one sample into a CSV row matching :func:`csv_columns`."""
    row = [f"{t:.6f}"]
    for name, kind, _ in recipe:
        v = smp[name]
        if kind == "v6":
            row += [f"{x:.6f}" for x in v]
        elif kind == "d":
            row.append(f"{v:.6f}")
        else:
            row.append(str(v))
    return row


def parse_registers(tokens):
    """``['1','vel','2','acc']`` -> ``[(1,'vel'),(2,'acc')]`` (commas optional)."""
    toks = [t.strip(",") for t in (tokens or []) if t.strip(",")]
    if len(toks) % 2:
        raise SystemExit("--float-register needs index name pairs, e.g. 1 vel 2 acc")
    regs = []
    for i in range(0, len(toks), 2):
        idx = int(toks[i])
        if not 0 <= idx <= 47:
            raise SystemExit(f"float register index {idx} out of range 0..47")
        regs.append((idx, toks[i + 1]))
    return regs


def record_stream(stream, out, recipe, stop_check=None):
    """Read samples from an open RTDE ``stream`` and write them to CSV ``out``.

    The core record loop, shared by ``record.py`` (log until Ctrl-C) and
    ``send.py`` (log until the sent program finishes). ``stop_check(smp, t)`` is
    called per sample with the sample dict and elapsed seconds; return a reason
    string to stop, or ``None`` to keep going. When ``None`` is passed the loop
    only stops on Ctrl-C or a dropped stream.

    Returns ``(n_samples, reason)``.
    """
    n, stop = 0, None
    try:
        with open(out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(csv_columns(recipe))
            t0 = time.monotonic()
            last_io = t0
            while True:
                smp = read_sample(stream, recipe)
                now = time.monotonic()
                w.writerow(sample_row(now - t0, smp, recipe))
                n += 1
                if stop_check is not None:
                    stop = stop_check(smp, now - t0)
                    if stop:
                        break
                # Flush + progress a few times a second, NOT per sample: flushing
                # at the full rate starves the reader and the controller drops us.
                if now - last_io >= 0.25:
                    fh.flush()
                    print(f"\r  {n} samples  {now - t0:6.1f}s", end="", flush=True)
                    last_io = now
    except KeyboardInterrupt:
        stop = "Ctrl-C"
    except ConnectionError as exc:
        stop = str(exc)
    return n, stop


def main():
    ap = argparse.ArgumentParser(description="Passively log UR RTDE state to a CSV.")
    ap.add_argument("--robot-ip", default=os.environ.get("UR_HOST", "127.0.0.1"),
                    help="UR controller IP (default: $UR_HOST or 127.0.0.1)")
    ap.add_argument("--out", default="run.csv", help="output CSV path")
    ap.add_argument("--hz", type=float, default=125.0, help="sample rate (default 125)")
    ap.add_argument("--float-register", nargs="+", metavar="IDX NAME",
                    help="log output float registers, e.g. 1 vel 2 acc")
    args = ap.parse_args()

    recipe = build_recipe(parse_registers(args.float_register))

    out = args.out
    if os.path.isdir(out):  # a folder was given: write a default file inside it
        out = os.path.join(out, "run.csv")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    stream = open_stream(args.robot_ip, args.hz, recipe)
    print(f"recording {args.robot_ip} -> {out}  (Ctrl-C to stop)")
    try:
        n, stop = record_stream(stream, out, recipe)
    finally:
        stream.close()
    print(f"\nwrote {n} samples to {out}" + (f"  ({stop})" if stop else ""))


if __name__ == "__main__":
    main()
