#!/bin/sh
# Launch the case 4 vision MCP server as root.
#
# librealsense cannot claim the RealSense D435 on macOS without root ("failed
# to set power state"), and Claude Code spawns MCP servers as the normal user,
# so .mcp.json invokes this via `sudo -n` (see the NOPASSWD rule in
# /etc/sudoers.d/vision-tools).
#
# sudo clears the environment, so the VISION_* settings are exported HERE
# rather than in .mcp.json -- that is the whole reason this wrapper exists.
REPO="/Users/MichaelQ/international-summer-school-robotics-TER-UR"

export VISION_MODE=real          # no simulated camera
export VISION_CAMERA=realsense   # fail loudly instead of falling back to webcam
export VISION_MODEL="$REPO/case 4/yolo26n.pt"   # absolute: cwd/HOME differ under sudo

# Cell-specific HOME/observation pose for go_view_pose, six joint angles in
# degrees. The code default is the kitchen home [0,-90,90,-90,-90,0] (upper
# arm vertical, forearm horizontal, tool straight down); once the base
# bearing that faces YOUR table is taught, set it here (sudo strips env, so
# it must live in this wrapper, not .mcp.json):
# export VISION_VIEW_Q_DEG="0,-90,90,-90,-90,0"

# Robot the vision servo drives. sudo strips env vars, so the target comes
# as the first ARGUMENT (see mcp.real-robot.json); with no argument the
# wrapper stays on the local simulator, so the sim .mcp.json and the real
# config can never split-brain across the two MCP servers. Flag-style args
# (--cam-test, --rs-probe) are not hosts and fall through to the script.
case "${1:-}" in
  ""|-*) export UR_HOST="127.0.0.1" ;;
  *)     export UR_HOST="$1"; shift ;;
esac

exec "$REPO/.venv/bin/python" "$REPO/case 4/vision_tools.py" "$@"
