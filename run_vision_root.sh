#!/bin/sh
# Launch the MERGED MCP server (case 1 motion + case 4 vision) as root.
#
# librealsense cannot claim the RealSense D435 on macOS without root ("failed
# to set power state"), and Claude Code spawns MCP servers as the normal user,
# so .mcp.json invokes this via `sudo -n`. The NOPASSWD rule in
# /etc/sudoers.d/vision-tools names THIS path, which is why the merged server
# kept the old filename -- renaming the file would need a sudoers edit too.
#
# sudo clears the environment, so every UR_* and VISION_* setting is exported
# HERE rather than in .mcp.json -- that is the whole reason this wrapper exists.
#
#   run_vision_root.sh                 one server, simulator robot + camera
#   run_vision_root.sh 192.168.1.100   one server, real UR5e + camera
#   run_vision_root.sh --rs-probe      camera diagnostics only, no robot
REPO="/Users/MichaelQ/international-summer-school-robotics-TER-UR"

# Do NOT hardware_reset the D435 on every open. The reset drops the camera off
# the bus and it re-enumerates at a NEW USB address, while librealsense's
# context still points at the old node -- so pipeline.start() answers "No
# device connected". camera.py reads that as "another process holds it" and
# breaks out of its fallback ladder, so startup fails outright instead of
# stepping down. With the reset off the D435 opens first try at 640x480@30
# with depth. To recover a genuinely wedged camera, reset it by hand:
#   sudo -n ./run_vision_root.sh --rs-probe --reset
export RS_HW_RESET=off

export VISION_MODE=real          # no simulated camera
export VISION_CAMERA=realsense   # fail loudly instead of falling back to webcam
export VISION_MODEL="$REPO/case 4/yolo26n.pt"   # absolute: cwd/HOME differ under sudo

# Cell-specific HOME/observation pose for go_view_pose, six joint angles in
# degrees. The code default is the kitchen home [0,-90,90,-90,-90,0] (upper
# arm vertical, forearm horizontal, tool straight down); once the base
# bearing that faces YOUR table is taught, set it here (sudo strips env, so
# it must live in this wrapper, not .mcp.json):
# export VISION_VIEW_Q_DEG="0,-90,90,-90,-90,0"

# Camera diagnostics still run the case 4 server standalone: those flags are
# vision_tools.py's own, and there is no robot to drive for them.
case "${1:-}" in
  --cam-test|--rs-probe|--reset)
    export UR_HOST="127.0.0.1"
    exec "$REPO/.venv/bin/python" "$REPO/case 4/vision_tools.py" "$@"
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

export UR_VISION=1               # tells server.py to mount the case 4 tools

exec "$REPO/.venv/bin/python" "$REPO/case 1/server.py"
