"""Offline tests for the grasp-approach geometry (no robot, no camera).

The arithmetic that decides how far to descend is the part that can hurt
someone, and it is pure: given a taught camera-to-fingertip offset and a
depth reading, how far does the tool still travel, and how much of that is
blind? These exercise it with a stub robot so it can be checked without the
cell -- run the live smoke test (test_vision_tools.py) for the rest.

    VISION_MODE=sim python test_grasp_approach.py
"""
from __future__ import annotations

import math
import os

os.environ.setdefault("VISION_MODE", "sim")
os.environ.setdefault("UR_HOST", "127.0.0.1")

import vision_tools as vt  # noqa: E402


class StubRobot:
    """Tool pointing straight down; records the linear moves asked of it."""

    def __init__(self, z=0.5):
        self.pose = [0.0, -0.6, z, math.pi, 0.0, 0.0]  # +Z tool -> -Z base
        self.moves = []

    def get_tcp_pose(self):
        return list(self.pose)

    def move_linear(self, pose, speed, accel, **kw):
        self.moves.append((list(pose), speed, accel))
        self.pose = list(pose)


class StubPercept:
    def __init__(self, name, depth):
        self.name, self.conf, self.depth_m = name, 0.9, depth


def expect_error(fn, needle, label):
    try:
        fn()
    except (ValueError, RuntimeError) as exc:
        assert needle in str(exc), f"{label}: wrong message: {exc}"
        print(f"  ok   {label}")
        return
    raise AssertionError(f"{label}: expected an error, got none")


def main() -> None:
    cal_path = vt.GRASP_CAL_PATH
    backup = cal_path.read_text() if cal_path.exists() else None
    if cal_path.exists():
        cal_path.unlink()

    robot = StubRobot()
    vt.engine.robot = robot
    vt.engine.visible = lambda: [StubPercept("cup", 0.40)]
    vt.engine.shared.servo_on.clear()

    try:
        # --- approach refuses to guess before it is taught ---------------- #
        expect_error(lambda: vt.approach_to_grasp("cup"),
                     "no grasp calibration", "uncalibrated approach refused")

        # --- the two-step teach measures the offset by travel ------------- #
        vt.start_grasp_calibration("cup")
        expect_error(vt.finish_grasp_calibration, "has not moved",
                     "teach rejects a robot that did not descend")
        robot.pose[2] -= 0.25                       # descend 25 cm to the grip
        out = vt.finish_grasp_calibration()
        assert abs(out["camera_to_fingertip_m"] - 0.15) < 1e-6, out
        assert abs(out["travelled_m"] - 0.25) < 1e-6, out
        print(f"  ok   taught offset = {out['camera_to_fingertip_m']} m "
              f"(0.40 depth - 0.25 travel)")

        expect_error(vt.finish_grasp_calibration, "Nothing to finish",
                     "teach state cleared after finishing")

        # --- fully servoed approach: nothing blind ------------------------ #
        vt.engine.shared.servo_on.clear()
        calls = {}
        vt.track_object = lambda name, standoff_m, wait_s: (
            calls.update(standoff_m=standoff_m) or
            {"locked": True, "state": "LOCKED on target", "depth_m": standoff_m})
        robot.moves.clear()
        r = vt.approach_to_grasp("cup")
        assert abs(calls["standoff_m"] - 0.15) < 1e-9, calls
        assert r["blind_descent_m"] == 0.0, r
        assert not robot.moves, "should not move blind when fully servoed"
        print(f"  ok   servoed to {calls['standoff_m']} m, blind descent 0 mm")

        # --- close grasp: the unmeasurable last stretch goes open-loop ---- #
        vt.set_grasp_calibration(camera_to_fingertip_m=0.12)
        robot.moves.clear()
        z_before = robot.pose[2]
        r = vt.approach_to_grasp("cup", grasp_depth_m=0.05)
        assert abs(calls["standoff_m"] - vt.MIN_STANDOFF_M) < 1e-9, calls
        assert abs(r["blind_descent_m"] - 0.03) < 1e-9, r
        assert len(robot.moves) == 1, robot.moves
        dropped = z_before - robot.moves[0][0][2]
        assert abs(dropped - 0.03) < 1e-9, f"descended {dropped}, want 0.03"
        assert robot.moves[0][1] == vt.GRASP_SPEED_MS
        print(f"  ok   target 0.07 m -> servo {vt.MIN_STANDOFF_M} m + "
              f"{dropped * 1000:.0f} mm blind, descending along tool +Z")

        # --- refuses to travel further blind than the cap ----------------- #
        vt.set_grasp_calibration(camera_to_fingertip_m=0.05)
        expect_error(lambda: vt.approach_to_grasp("cup", grasp_depth_m=0.02),
                     "travel blind", "blind-descent cap enforced")

        # --- a failed lock must not become a blind lunge ------------------ #
        vt.set_grasp_calibration(camera_to_fingertip_m=0.12)
        vt.track_object = lambda name, standoff_m, wait_s: {
            "locked": False, "state": "servo on -- target lost, holding"}
        robot.moves.clear()
        expect_error(lambda: vt.approach_to_grasp("cup", grasp_depth_m=0.05),
                     "NO blind descent", "no descent without a lock")
        assert not robot.moves, "moved despite failing to lock!"

        # --- input validation --------------------------------------------- #
        expect_error(lambda: vt.set_grasp_calibration(1.5),
                     "0.02-0.60", "implausible offset rejected")
        expect_error(lambda: vt.approach_to_grasp("cup", grasp_depth_m=0.9),
                     "+/-0.05", "out-of-range grasp depth rejected")

        print("\nall grasp-approach geometry tests passed")
    finally:
        if backup is not None:
            cal_path.write_text(backup)
        elif cal_path.exists():
            cal_path.unlink()


if __name__ == "__main__":
    main()
