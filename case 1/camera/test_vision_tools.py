"""Smoke test for the vision-tools MCP server against the live simulator.

Runs in-process (same pattern as case 1's test_server.py): sim perception, a
virtual cup and phone, and the REAL simulated robot moving under URSim's
controller. Checks the language-level flow an LLM would drive:

    look -> place objects -> track_object("cup") -> LOCKED
         -> track_object("phone", standoff_m=0.18) -> LOCKED closer
         -> grasp_tracked_object(dry_run) -> stop_tracking

Prereqs: the simulator is up and the robot is RUNNING, and the environment
selects sim perception:

    VISION_MODE=sim python test_vision_tools.py
"""
from __future__ import annotations

import asyncio
import json
import math
import os

os.environ.setdefault("VISION_MODE", "sim")

from fastmcp import Client  # noqa: E402

from servo import VIEW_Q  # noqa: E402
from vision_tools import engine, mcp  # noqa: E402


def unpack(result):
    return json.loads(result.content[0].text)


async def main() -> None:
    tcp0 = engine.robot.get_tcp_pose()
    print("sim TCP at start:", [round(v, 3) for v in tcp0])

    async with Client(mcp) as client:
        tools = {t.name for t in await client.list_tools()}
        need = {"look", "track_object", "stop_tracking", "tracking_status",
                "grasp_tracked_object", "show_camera_view", "hide_camera_view"}
        assert need <= tools, f"missing tools: {need - tools}"
        print("tools:", ", ".join(sorted(tools)))

        # Start from a healthy configuration: the arm may be parked in a
        # singular pose from earlier experiments, and the servo refuses
        # movel steps from there (PolyScope X C204A1). In the merged server
        # the recovery is case 1's move_robot_to_position() with no
        # arguments; this file drives vision_tools alone, so it commands the
        # same pose straight through the engine's own robot client.
        engine.robot.move_joint(VIEW_Q, 0.8, 1.0)
        print("home/view pose ->", [round(v, 3) for v in engine.robot.get_tcp_pose()])
        tcp_view = engine.robot.get_tcp_pose()  # motion baseline for the end

        # An empty scene first: look() sees nothing, track_object refuses.
        out = unpack(await client.call_tool("look", {}))
        assert out["camera_mode"] == "sim" and out["objects"] == []
        print("look (empty scene) ok")
        try:
            await client.call_tool("track_object", {"object_name": "cup"})
            raise AssertionError("track_object accepted an empty scene")
        except Exception as exc:
            assert "No 'cup' in view" in str(exc), str(exc)
            print("track refuses empty scene:", str(exc).splitlines()[-1][:80])

        # Drop a cup ahead-right of the camera, a phone ahead-left. This is
        # a test fixture, not an MCP tool -- place_sim_object was removed from
        # the tool surface because the shipped config is always VISION_MODE=real.
        p_cup = engine.perception.place_ahead("cup", 0.50, 0.10, 0.06)
        print("placed cup at", [round(v, 4) for v in p_cup])
        p_phone = engine.perception.place_ahead("cell phone", 0.55, -0.12, 0.02)
        print("placed phone at", [round(v, 4) for v in p_phone])

        out = unpack(await client.call_tool("look", {}))
        names = {o["name"] for o in out["objects"]}
        assert names == {"cup", "cell phone"}, names
        print("look sees:", out["objects"])

        # "track the cup"
        st = unpack(await client.call_tool("track_object", {
            "object_name": "cup", "wait_s": 90}))
        print("track_object ->", st["state"])
        assert st["locked"], f"never locked: {st}"
        assert abs(st["depth_m"] - 0.35) < 0.025, st
        err = math.hypot(*st["offcenter_norm"])
        print(f"  cup centred: offcenter={err:.4f}, depth={st['depth_m']}m")

        # "close in on the phone" -- swaps target AND tightens the standoff.
        # (descend_on was removed: it was track_object with other defaults.)
        st = unpack(await client.call_tool("track_object", {
            "object_name": "phone", "standoff_m": 0.18, "wait_s": 90}))
        print("track_object (close) ->", st["state"])
        assert st["locked"], f"never locked: {st}"
        assert abs(st["depth_m"] - 0.18) < 0.025, st
        print(f"  phone at {st['depth_m']}m, offcenter="
              f"{[round(v,4) for v in st['offcenter_norm']]}")

        # The fixed pick routine: dry-run only, so the suite never closes a
        # gripper or lifts. It must refuse when nothing is locked, and its
        # plan must stay inside the blind-move cap.
        plan = unpack(await client.call_tool("grasp_tracked_object",
                                             {"dry_run": True}))
        assert plan["status"] == "planned", plan
        assert plan["correction_tool_m"] == [0.015, 0.0, 0.03], plan
        assert math.dist(plan["correction_tool_m"], [0, 0, 0]) <= 0.06, plan
        # At the home orientation tool +X is base +Y and tool +Z is base -Z,
        # so the correction must read as "sideways in +Y, straight down".
        base = plan["correction_base_m"]
        assert abs(base[0]) < 1e-3 and abs(base[1] - 0.015) < 1e-3, base
        assert abs(base[2] + 0.03) < 1e-3, base
        # A dry run must leave the servo loop alone.
        assert unpack(await client.call_tool("tracking_status", {}))["tracking"]
        print("grasp plan:", base, "->", plan["grasp_pose"])

        st = unpack(await client.call_tool("stop_tracking", {}))
        assert not st["tracking"]
        print("stop_tracking ok")

        # Bad inputs are rejected with readable reasons.
        for args, why in (
                ({"object_name": "cup", "standoff_m": 0.02}, "standoff_m"),
                ({"object_name": "unicorn"}, "in view")):
            try:
                await client.call_tool("track_object", args)
                raise AssertionError(f"accepted bad input {args}")
            except Exception as exc:
                assert why in str(exc), (why, str(exc))
        try:
            await client.call_tool("grasp_tracked_object", {})
            raise AssertionError("grasp_tracked_object ran without a lock")
        except Exception as exc:
            assert "Not locked" in str(exc), str(exc)
            print("grasp refuses without a lock ok")

        print("input validation ok")

    tcp1 = engine.robot.get_tcp_pose()
    moved = math.dist(tcp_view[:3], tcp1[:3])
    print(f"sim TCP at end:   {[round(v, 3) for v in tcp1]}")
    print(f"servoing carried the TCP {moved*100:.1f} cm from the view pose")
    assert moved > 0.05, "robot never moved?"
    print("\nALL VISION-TOOLS CHECKS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
