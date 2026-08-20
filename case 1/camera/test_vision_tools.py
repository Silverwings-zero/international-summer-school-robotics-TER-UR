"""Smoke test for the vision-tools MCP server against the live simulator.

Runs in-process (same pattern as case 1's test_server.py): sim perception, a
virtual cup and phone, and the REAL simulated robot moving under URSim's
controller. Checks the language-level flow an LLM would drive:

    look -> place objects -> track_object("cup") -> LOCKED
         -> descend_on("phone") -> LOCKED closer -> stop_tracking

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

from vision_tools import engine, mcp  # noqa: E402


def unpack(result):
    return json.loads(result.content[0].text)


async def main() -> None:
    tcp0 = engine.robot.get_tcp_pose()
    print("sim TCP at start:", [round(v, 3) for v in tcp0])

    async with Client(mcp) as client:
        tools = {t.name for t in await client.list_tools()}
        need = {"look", "track_object", "descend_on", "stop_tracking",
                "tracking_status", "calibrate_hand_eye", "place_sim_object",
                "go_view_pose", "show_camera_view", "hide_camera_view"}
        assert need <= tools, f"missing tools: {need - tools}"
        print("tools:", ", ".join(sorted(tools)))

        # Start from a healthy configuration: the arm may be parked in a
        # singular pose from earlier experiments, and the servo refuses
        # movel steps from there (PolyScope X C204A1). The recovery path an
        # LLM would take is exactly this tool.
        out = unpack(await client.call_tool("go_view_pose", {}))
        print("go_view_pose ->", out["state"])
        assert out["ok"], out
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

        # Drop a cup ahead-right of the camera, a phone ahead-left.
        out = unpack(await client.call_tool("place_sim_object", {
            "object_name": "cup", "forward_m": 0.50,
            "right_m": 0.10, "down_m": 0.06}))
        print("placed cup at", out["position_base_m"])
        out = unpack(await client.call_tool("place_sim_object", {
            "object_name": "cell phone", "forward_m": 0.55,
            "right_m": -0.12, "down_m": 0.02}))
        print("placed phone at", out["position_base_m"])

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

        # "descend on the phone" -- swaps target AND closes in to 0.18 m.
        st = unpack(await client.call_tool("descend_on", {
            "object_name": "phone", "standoff_m": 0.18, "wait_s": 90}))
        print("descend_on ->", st["state"])
        assert st["locked"], f"never locked: {st}"
        assert abs(st["depth_m"] - 0.18) < 0.025, st
        print(f"  phone at {st['depth_m']}m, offcenter="
              f"{[round(v,4) for v in st['offcenter_norm']]}")

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
        print("input validation ok")

    tcp1 = engine.robot.get_tcp_pose()
    moved = math.dist(tcp_view[:3], tcp1[:3])
    print(f"sim TCP at end:   {[round(v, 3) for v in tcp1]}")
    print(f"servoing carried the TCP {moved*100:.1f} cm from the view pose")
    assert moved > 0.05, "robot never moved?"
    print("\nALL VISION-TOOLS CHECKS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
