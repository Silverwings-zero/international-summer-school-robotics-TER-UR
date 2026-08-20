"""Preflight for pointing the MCP servers at a real UR arm.

Run this BEFORE editing any config, from the repo venv:

    ../.venv/bin/python preflight_real_robot.py 192.168.1.10

It verifies, in order: the three controller ports are reachable, what the
dashboard says about the controller (PolyScope version, robot mode, safety
status, Remote Control mode), and that a full RTDE state snapshot decodes --
the exact read path the MCP tools use. Nothing here moves the robot.

Exit code 0 means every check passed and the arm is ready for the MCP
servers; 1 means at least one check failed (each failure prints what to do).
"""

from __future__ import annotations

import argparse
import math
import re
import socket
import sys

from ur_client import URClient

DASHBOARD_PORT = 29999
PRIMARY_PORT = 30001
RTDE_PORT = 30004

ROBOT_MODE_NAMES = {
    -1: "NO_CONTROLLER", 0: "DISCONNECTED", 1: "CONFIRM_SAFETY",
    2: "BOOTING", 3: "POWER_OFF", 4: "POWER_ON", 5: "IDLE",
    6: "BACKDRIVE", 7: "RUNNING", 8: "UPDATING_FIRMWARE",
}
SAFETY_STATUS_NAMES = {
    1: "NORMAL", 2: "REDUCED", 3: "PROTECTIVE_STOP", 4: "RECOVERY",
    5: "SAFEGUARD_STOP", 6: "SYSTEM_EMERGENCY_STOP",
    7: "ROBOT_EMERGENCY_STOP", 8: "VIOLATION", 9: "FAULT",
    10: "VALIDATE_JOINT_ID", 11: "UNDEFINED_SAFETY_MODE",
    12: "AUTOMATIC_MODE_SAFEGUARD_STOP", 13: "SYSTEM_THREE_POSITION_ENABLING_STOP",
}


class _DashboardSilent(Exception):
    """Dashboard answered with empty strings (PolyScope X simulator)."""


def check_port(host: str, port: int, label: str) -> bool:
    try:
        with socket.create_connection((host, port), timeout=3):
            print(f"  ok   {label} (port {port}) reachable")
            return True
    except OSError as exc:
        print(f"  FAIL {label} (port {port}) unreachable: {exc}")
        return False


class Dashboard:
    """Minimal client for the UR dashboard server (line protocol)."""

    def __init__(self, host: str):
        self._sock = socket.create_connection((host, DASHBOARD_PORT), timeout=3)
        self._file = self._sock.makefile("r", encoding="ascii", newline="\n")
        self.banner = self._file.readline().strip()

    def ask(self, command: str) -> str:
        self._sock.sendall((command + "\n").encode("ascii"))
        return self._file.readline().strip()

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("host", help="IP of the real robot, e.g. 192.168.1.10")
    args = ap.parse_args()
    host = args.host
    failures: list[str] = []

    print(f"Preflight for UR controller at {host}\n")

    print("1. Controller ports")
    ports_ok = all([
        check_port(host, DASHBOARD_PORT, "dashboard"),
        check_port(host, PRIMARY_PORT, "primary/URScript"),
        check_port(host, RTDE_PORT, "RTDE"),
    ])
    if not ports_ok:
        failures.append(
            "port(s) unreachable -- check the IP (pendant: hamburger menu > "
            "About), that this machine is on the same network/subnet, and "
            "that no firewall blocks the ports")

    print("\n2. Dashboard")
    remote_control = None
    if ports_ok:
        try:
            dash = Dashboard(host)
            print(f"  ok   {dash.banner}")
            version = dash.ask("PolyscopeVersion")
            if not version:
                # The PolyScope X simulator accepts the connection but
                # answers nothing; a real UR5e (PolyScope 5) always replies.
                print("  note dashboard connected but gave an empty reply.")
                print("       The PolyScope X SIMULATOR behaves this way "
                      "(expected there);")
                print("       a real UR5e answering empty means something "
                      "is wrong.")
                failures.append(
                    "dashboard gave empty replies -- if this host is the "
                    "simulator, ignore; on the real UR5e check PolyScope "
                    "is fully booted, and verify Remote Control mode "
                    "manually on the pendant")
                dash.close()
                raise _DashboardSilent
            print(f"  ok   PolyScope version: {version}")
            m = re.search(r"(\d+)\.(\d+)", version)
            ps = (int(m.group(1)), int(m.group(2))) if m else None
            print(f"  ok   {dash.ask('robotmode')}")
            # 'safetystatus' exists from PolyScope 5.4; older 5.x answer
            # "could not understand" -- fall back to 'safetymode' there.
            safety = (dash.ask("safetystatus") if ps is None or ps >= (5, 4)
                      else dash.ask("safetymode"))
            print(f"  ok   {safety}")
            if "could not understand" in safety.lower():
                print("  note controller too old for this query -- check "
                      "the safety state on the pendant")
            elif not safety.endswith("NORMAL"):
                failures.append(
                    f"safety is '{safety}' -- clear it on the pendant "
                    "before the MCP tools can move the arm")
            # 'is in remote control' exists from PolyScope 5.6; before 5.6
            # (and before the Local/Remote toggle existed at all in 5.0-5.4)
            # external URScript is accepted without it.
            if ps is not None and ps < (5, 6):
                print("  note PolyScope < 5.6: cannot query Remote Control "
                      "-- if the pendant has a Local/Remote toggle, verify "
                      "it manually")
            else:
                reply = dash.ask("is in remote control").strip().lower()
                if reply == "true":
                    print("  ok   remote control mode: true")
                elif reply == "false":
                    print("  FAIL remote control mode: false")
                    failures.append(
                        "Remote Control is OFF -- an e-Series arm ignores "
                        "external URScript in Local mode. On the pendant: "
                        "top-right hamburger/profile icon > Settings > "
                        "System > Remote Control > Enable, then switch the "
                        "mode toggle (top right) from Local to Remote")
                else:
                    print(f"  note remote control query answered "
                          f"'{reply}' -- verify the mode manually on the "
                          "pendant")
            dash.close()
        except _DashboardSilent:
            pass
        except OSError as exc:
            print(f"  FAIL dashboard conversation failed: {exc}")
            failures.append(f"dashboard conversation failed: {exc}")
    else:
        print("  skipped (ports unreachable)")

    print("\n3. RTDE state snapshot (the MCP read path)")
    if ports_ok:
        try:
            state = URClient(host=host).get_state()
            joints = ", ".join(f"{math.degrees(q):.1f}" for q in state.q_rad)
            tcp = ", ".join(f"{v:.3f}" for v in state.tcp_pose[:3])
            print(f"  ok   joints (deg): [{joints}]")
            print(f"  ok   TCP (m): [{tcp}]")
            print(f"  ok   robot mode: "
                  f"{ROBOT_MODE_NAMES.get(state.robot_mode, state.robot_mode)}"
                  f", safety: "
                  f"{SAFETY_STATUS_NAMES.get(state.safety_status, state.safety_status)}")
            if state.robot_mode != 7:
                failures.append(
                    "robot is not in RUNNING mode -- on the pendant power "
                    "the arm on and release the brakes")
        except Exception as exc:
            print(f"  FAIL RTDE snapshot failed: {exc}")
            failures.append(f"RTDE snapshot failed: {exc}")
    else:
        print("  skipped (ports unreachable)")

    print("\n" + "=" * 60)
    if failures:
        print(f"NOT READY -- {len(failures)} item(s) to fix:")
        for i, f in enumerate(failures, 1):
            print(f"  {i}. {f}")
        return 1

    print("READY. Before the first motion:")
    print("  - keep a hand on the e-stop; nobody inside the reach envelope")
    print("  - pendant speed slider at 25% or less for the first session")
    print("  - verify payload + TCP settings on the pendant match the tool")
    print("  - then: copy mcp.real-robot.json over .mcp.json (fill in the")
    print("    IP), restart Claude Code, and ask the robot to move")
    return 0


if __name__ == "__main__":
    sys.exit(main())
