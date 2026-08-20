#!/bin/sh
# Launch the case 1 MCP server WITH the wrist camera (UR_VISION=1), so motion
# and perception arrive as a single "ur-tools" MCP.
#
# Everything it needs is inside case 1: the server next to this file, the
# camera stack in camera/. Nothing outside this directory is referenced, and
# no path is absolute, so the same file works on every clone and every machine.
#
# ROOT. librealsense cannot claim the D435 on macOS without it ("failed to set
# power state"), so there this is invoked via `sudo -n` and a NOPASSWD rule.
# On Linux the udev rules (99-realsense-libusb.rules, MODE 0666 + plugdev)
# hand the camera to the normal user, so run_voice.py calls it directly. This
# script is root-agnostic -- only the caller decides.
#
# sudo clears the environment, so every UR_* and VISION_* setting is exported
# HERE rather than in .mcp.json -- that is the whole reason this wrapper exists.
#
#   run_server.sh                 simulator robot + camera
#   run_server.sh 192.168.1.100   real UR5e + camera
#   run_server.sh --rs-probe      camera diagnostics only, no robot
#
# CASE1 is this file's own directory; REPO is only needed for the venv lookup.
CASE1="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO="$(CDPATH= cd -- "$CASE1/.." && pwd)"

# The interpreter to run the server with, most specific first:
#   VOICE_PYTHON  run_voice.py passes the exact interpreter it is running
#                 under, so the server always lands in the same environment as
#                 the client. Stripped by sudo, hence the fallbacks.
#   .venv         the repo virtualenv, when the clone has one.
#   python3       whatever is on PATH (this is the Anaconda case).
if [ -n "${VOICE_PYTHON:-}" ] && [ -x "${VOICE_PYTHON:-}" ]; then
    PYTHON="$VOICE_PYTHON"
elif [ -x "$REPO/.venv/bin/python" ]; then
    PYTHON="$REPO/.venv/bin/python"
else
    PYTHON="$(command -v python3)"
fi
if [ -z "$PYTHON" ]; then
    echo "run_server.sh: no python interpreter found (tried \$VOICE_PYTHON," >&2
    echo "  $REPO/.venv/bin/python, and python3 on PATH)." >&2
    exit 127
fi

# Do NOT hardware_reset the D435 on every open. The reset drops the camera off
# the bus and it re-enumerates at a NEW USB address, while librealsense's
# context still points at the old node -- so pipeline.start() answers "No
# device connected". camera.py reads that as "another process holds it" and
# breaks out of its fallback ladder, so startup fails outright instead of
# stepping down. With the reset off the D435 opens first try at 640x480@30
# with depth. To recover a genuinely wedged camera, reset it by hand:
#   ./run_server.sh --rs-probe --reset
export RS_HW_RESET=off

export VISION_MODE=real          # no simulated camera
export VISION_CAMERA=realsense   # fail loudly instead of falling back to webcam
export VISION_MODEL="$CASE1/camera/yolo26n.pt"  # absolute: cwd/HOME differ under sudo

# Observation pose for the standalone camera viewer's 'v' key, six joint
# angles in degrees. The merged server reaches this pose with case 1's
# move_robot_to_position() instead, which reads HOME_Q_RAD -- keep the two in
# step. Six joint angles in
# degrees. The code default is the kitchen home [0,-90,90,-90,-90,0] (upper
# arm vertical, forearm horizontal, tool straight down); once the base
# bearing that faces YOUR table is taught, set it here (sudo strips env, so
# it must live in this wrapper, not .mcp.json):
# export VISION_VIEW_Q_DEG="0,-90,90,-90,-90,0"

# Camera diagnostics still run the camera server standalone: those flags are
# vision_tools.py's own, and there is no robot to drive for them.
case "${1:-}" in
  --cam-test|--rs-probe|--reset)
    export UR_HOST="127.0.0.1"
    exec "$PYTHON" "$CASE1/camera/vision_tools.py" "$@"
    ;;
esac

# Robot the merged server drives. sudo strips env vars, so the target comes as
# the first ARGUMENT (see mcp.real-robot.json); with no argument the wrapper
# stays on the local simulator, so the sim and real configs can never
# split-brain. UR_MODEL is pinned to the host rather than configured
# separately, because a real UR5e driven with UR10e limits is the one
# combination CLAUDE.md rules out; an optional second argument overrides it.
case "${1:-}" in
  ""|-*)
    export UR_HOST="127.0.0.1"
    export UR_MODEL="${2:-ur10e}"     # the PolyScope X simulator is a UR10e
    ;;
  *)
    export UR_HOST="$1"
    export UR_MODEL="${2:-ur5e}"
    export UR_MAX_JOINT_SPEED="1.0"
    export UR_MAX_JOINT_ACCEL="2.0"
    export UR_MAX_TCP_SPEED="0.4"
    export UR_MAX_TCP_ACCEL="1.2"
    ;;
esac

export UR_VISION=1               # tells server.py to mount camera/ tools

exec "$PYTHON" "$CASE1/server.py"
