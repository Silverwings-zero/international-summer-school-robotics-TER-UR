"""Record model-predicted vs actual UR signals over RTDE while the robot moves.

Subscribes its own RTDE recipe (does not touch ur_client's), streams at 125 Hz
into memory while commanding a short home -> bent -> home motion, then dumps
everything to signals.json next to this script.

Signals per sample:
  timestamp                    controller clock, seconds
  target_q / actual_q          commanded vs measured joint angles (rad)
  target_qd / actual_qd        commanded vs measured joint velocities (rad/s)
  target_current / actual_current  model-predicted vs measured current (A)
  target_moment                model-predicted joint torque (Nm)
  actual_TCP_force             external wrench estimate at the tool
"""
from __future__ import annotations

import json
import socket
import struct
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ur_client import URClient  # noqa: E402

HOST = "127.0.0.1"
RTDE_PORT = 30004
FIELDS = ("timestamp,target_q,actual_q,target_qd,actual_qd,"
          "target_current,actual_current,target_moment,actual_TCP_force")

HOME = [0.0, -1.5707963, 0.0, -1.5707963, 0.0, 0.0]
BENT = [0.3490659, -1.2217305, 0.7853982, -1.1344640, -0.5235988, 0.0]  # [20,-70,45,-65,-30,0] deg


def rtde_send(s, cmd, payload=b""):
    s.sendall(struct.pack(">HB", 3 + len(payload), cmd) + payload)


def rtde_recv(s):
    hdr = b""
    while len(hdr) < 3:
        c = s.recv(3 - len(hdr))
        if not c:
            raise ConnectionError("closed")
        hdr += c
    size, cmd = struct.unpack(">HB", hdr)
    body = b""
    while len(body) < size - 3:
        c = s.recv(size - 3 - len(body))
        if not c:
            raise ConnectionError("closed")
        body += c
    return cmd, body


def record(stop: threading.Event, out: list) -> None:
    with socket.create_connection((HOST, RTDE_PORT), timeout=5) as s:
        rtde_send(s, 86, struct.pack(">H", 2)); rtde_recv(s)
        rtde_send(s, 79, struct.pack(">d", 125.0) + FIELDS.encode())
        cmd, body = rtde_recv(s)
        types = body[1:].decode()
        assert "NOT_FOUND" not in types, f"recipe rejected: {types}"
        rtde_send(s, 83); rtde_recv(s)
        while not stop.is_set():
            cmd, body = rtde_recv(s)
            if cmd != 85:
                continue
            off = 1
            ts = struct.unpack(">d", body[off:off + 8])[0]; off += 8
            vecs = []
            for _ in range(8):
                vecs.append(list(struct.unpack(">6d", body[off:off + 48])))
                off += 48
            out.append([ts] + vecs)


def main() -> None:
    samples: list = []
    stop = threading.Event()
    t = threading.Thread(target=record, args=(stop, samples), daemon=True)
    t.start()

    r = URClient()
    time.sleep(1.0)                      # baseline at rest
    r.move_joint(HOME, 1.0, 1.4)
    r.move_joint(BENT, 1.0, 1.4)         # the interesting segment
    r.move_joint(HOME, 1.0, 1.4)
    time.sleep(1.0)                      # settle tail
    stop.set()
    t.join(timeout=3)

    t0 = samples[0][0]
    data = {
        "t_s": [round(s[0] - t0, 4) for s in samples],
        "target_q": [s[1] for s in samples],
        "actual_q": [s[2] for s in samples],
        "target_qd": [s[3] for s in samples],
        "actual_qd": [s[4] for s in samples],
        "target_current": [s[5] for s in samples],
        "actual_current": [s[6] for s in samples],
        "target_moment": [s[7] for s in samples],
        "tcp_force": [s[8] for s in samples],
    }
    out = Path(__file__).with_name("signals.json")
    out.write_text(json.dumps(data))

    n = len(samples)
    dur = data["t_s"][-1]
    gap_cur = max(abs(a - t) for s_a, s_t in zip(data["actual_current"], data["target_current"])
                  for a, t in zip(s_a, s_t))
    gap_q = max(abs(a - t) for s_a, s_t in zip(data["actual_q"], data["target_q"])
                for a, t in zip(s_a, s_t))
    peak_moment = max(abs(v) for s in data["target_moment"] for v in s)
    print(f"recorded {n} samples over {dur:.2f}s (~{n/dur:.0f} Hz) -> {out}")
    print(f"max |actual-target| current: {gap_cur:.6f} A")
    print(f"max |actual-target| position: {gap_q:.6f} rad")
    print(f"peak predicted joint torque: {peak_moment:.1f} Nm")


if __name__ == "__main__":
    main()
