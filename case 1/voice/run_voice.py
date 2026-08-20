"""Entry point: talk to the robot out loud.

    python voice/run_voice.py                       # $UR_HOST, else simulator
    python voice/run_voice.py --robot 192.168.1.100 # the real UR5e + camera
    export UR_HOST=192.168.1.100                    # ...or set it once
    python voice/run_voice.py --no-vision           # robot only, cannot see
    python voice/run_voice.py --text                # typed in (no microphone)
    python voice/run_voice.py --tts none            # typed in, printed out
    python voice/run_voice.py --progress none       # no commentary
    python voice/run_voice.py --list-devices        # which microphone is which

WHICH ARM, AND WHETHER IT CAN SEE. Both follow from two flags, because the
pairing is a safety matter rather than a preference: a real address (from
``--robot IP`` or ``$UR_HOST``) means the real arm and therefore a ur5e safety
envelope, and loopback or nothing means the local simulator and its ur10e. Guarding a real UR5e with UR10e geometry would
approve targets 40 cm past its reach, so ``server.py`` refuses that pairing
outright. ``--vision`` (the default) launches the merged camera+robot server
through ``run_server.sh`` (under sudo on macOS, the only way librealsense can
claim the D435 on macOS -- and yields 59 tools; ``--no-vision`` launches
``server.py`` directly for 43 tools and no perception at all.

At the prompt: ENTER starts and stops a recording, ``!`` stops the robot
immediately, ``q`` quits. Ctrl-C also stops the robot before exiting.

A turn that calls several tools takes many seconds, so the assistant speaks
while it works -- its own "let me check where the arm is", a short line per
tool call, and a filler if a single call blocks for longer than --heartbeat
seconds. See ``progress.py``; ``--progress print`` keeps it on screen only.

Nothing here is required by ``server.py``. Kill this program and the MCP server
is still a normal MCP server for Claude Code, Cursor or the Case 3 agent.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import shlex
import sys

from bridge import LLMClient, connect_tools
from conversation import STOP_ACKNOWLEDGED, VoiceAgent
from progress import HEARTBEAT_SECONDS, make_narrator
from tts import DEFAULT_VOICE, make_speaker

_HERE = os.path.dirname(os.path.abspath(__file__))
# case 1 itself: this file's parent. Everything below is resolved from it, so
# the front-end never needs to know where the repo lives or what the sibling
# folders are called.
_CASE1 = os.path.abspath(os.path.join(_HERE, os.pardir))

# The robot-only server: 43 tools, no root needed, and NO camera. Launched with
# the interpreter running this script so it inherits the same virtualenv on
# machines with no bare `python` on PATH.
_ROBOT_ONLY_SERVER = os.path.join(_CASE1, "server.py")
# The merged server: those same 43 tools plus the 16 wrist-camera ones. It goes
# through the wrapper rather than being invoked directly because UR_VISION and
# every UR_*/VISION_* setting live INSIDE that file -- sudo strips the
# environment, so the robot has to arrive as an argument instead.
_MERGED_SERVER = os.path.join(_CASE1, "run_server.sh")

# Whether the camera needs root. This is a macOS problem, not a camera problem:
# librealsense cannot claim the D435 there without it ("failed to set power
# state"). On Linux the udev rules (99-realsense-libusb.rules, MODE 0666 +
# plugdev) hand the device to the normal user, so routing through `sudo -n`
# would add a passwordless-sudoers dependency that buys exactly nothing -- and
# on a box without that rule it turns every run into "Connection closed".
_CAMERA_NEEDS_ROOT = sys.platform == "darwin"


# Default target for a cell that always drives the same arm, so the IP is
# configuration rather than a flag typed on every run:
#     export UR_HOST=192.168.1.100
# --robot still wins over it, and it is deliberately NOT a literal in this file
# -- an address baked into the source is exactly the laptop-specific coupling
# the rest of this tree was cleaned of.
_ENV_ROBOT = os.environ.get("UR_HOST", "").strip()


def is_simulator(host: str) -> bool:
    """Loopback (or nothing) means the local simulator, not a real arm."""
    return not host or host == "localhost" or host.startswith("127.")


def resolve_target(args) -> tuple[str, str]:
    """(host, model) this run will drive, mirroring run_server.sh's rule.

    A real address means the real arm, which is a UR5e in this cell; loopback
    or nothing means the local simulator and its UR10e. Deriving both from one
    value is what stops the host and the model from disagreeing: a real UR5e
    guarded by UR10e geometry gets a safety layer that approves targets 40 cm
    beyond its reach and 7.5 kg past its payload, which is why server.py
    refuses that combination outright.

    The host comes from --robot, else $UR_HOST, else the simulator. To force
    the simulator while UR_HOST is exported, pass --robot 127.0.0.1.
    """
    host = args.robot or _ENV_ROBOT
    if is_simulator(host):
        return "127.0.0.1", (args.ur_model or "ur10e")
    return host, (args.ur_model or "ur5e")


def build_server_command(args, host: str) -> str:
    """The MCP server command line, unless --server overrode it outright.

    Takes the RESOLVED host rather than reading ``args.robot``: with $UR_HOST
    exported those two differ, and passing the flag would leave the server on
    the simulator while the client believed it was driving the real arm.
    """
    if args.server:
        return args.server
    if not args.vision:
        return " ".join(shlex.quote(p) for p in
                        (sys.executable, _ROBOT_ONLY_SERVER))
    # The wrapper takes the robot as its first argument and an optional model
    # override as its second; no arguments at all means the local simulator.
    parts = ["/usr/bin/sudo", "-n", _MERGED_SERVER] if _CAMERA_NEEDS_ROOT \
        else [_MERGED_SERVER]
    real = not is_simulator(host)
    if real or args.ur_model:
        parts.append(host if real else "")   # empty -> the simulator branch
        if args.ur_model:
            parts.append(args.ur_model)
    return " ".join(shlex.quote(p) for p in parts)


# Present only when camera/'s tools actually mounted, so it answers "can this
# session see?" without counting tools or parsing the server's log.
_VISION_SENTINEL = "look"


def banner(host: str, model: str, vision: bool,
           tool_names: list[str] | None = None) -> str:
    """The header, naming the arm and the tools this session actually got.

    The old banner said "UR10e" unconditionally, which was exactly wrong on the
    day it mattered -- the defaults reached the real UR5e. Print what was
    resolved instead, so the operator sees the target before speaking to it.

    ``vision`` is only what was ASKED for. ``_mount_vision`` is deliberately
    non-fatal -- a missing OpenCV costs the perception tools, not the robot --
    so a session can start with --vision and no eyes, and saying "camera +
    robot" there is a lie the operator only discovers by asking the robot to
    look at something. Report the catalogue that actually arrived.
    """
    where = "simulator" if is_simulator(host) else f"REAL ROBOT at {host}"
    if tool_names is None:
        eyes = "camera + robot" if vision else "robot only -- NO camera"
    elif _VISION_SENTINEL in tool_names:
        eyes = "camera + robot"
    elif vision:
        eyes = "robot only -- CAMERA FAILED to load (see the server log)"
    else:
        eyes = "robot only -- NO camera"
    return ("=========================================================\n"
            f" {model} -- voice assistant -- {where}\n"
            f" {eyes}\n"
            " [ENTER] speak    [!] STOP the robot    [q] quit\n"
            "=========================================================")

# FastMCP logs a full traceback for every rejected tool call. Useful, but not
# on top of a spoken conversation, so it lands here instead.
DEFAULT_SERVER_LOG = os.path.join(_HERE, "server.log")

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default=None,
                    help="override the MCP server command outright (quoted); "
                         "by default it is derived from --vision and --robot")
    ap.add_argument("--robot", default=None, metavar="IP",
                    help="drive the REAL robot at this address (e.g. "
                         "192.168.1.100); defaults to $UR_HOST, and falls "
                         "back to the local simulator. Pass 127.0.0.1 to "
                         "force the simulator when UR_HOST is exported")
    ap.add_argument("--ur-model", default=None,
                    choices=["ur10e", "ur5e", "ur5"],
                    help="which ARM the safety layer guards (not the LLM: "
                         "that is --model); defaults to ur5e with --robot "
                         "and ur10e without it")
    ap.add_argument("--vision", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="launch the merged camera+robot server via "
                         "run_server.sh (needs a passwordless sudo rule on "
                         "rule); --no-vision falls back to the 43-tool "
                         "robot-only server, which cannot see")
    ap.add_argument("--backend", choices=["claude", "openai"], default="claude",
                    help="'claude' uses your Claude Code subscription via the "
                         "claude CLI (no API key); 'openai' uses an "
                         "OpenAI-compatible endpoint via AGENT_* variables")
    ap.add_argument("--model", default=None,
                    help="model id; default is whatever the backend is set to")
    ap.add_argument("--text", action="store_true",
                    help="type instead of speaking (no microphone needed)")
    ap.add_argument("--tts", choices=["edge", "none"], default="edge",
                    help="'edge' for neural voice, 'none' to print replies")
    ap.add_argument("--voice", default=DEFAULT_VOICE,
                    help="Edge voice id (edge-tts --list-voices)")
    ap.add_argument("--rate", default="+0%",
                    help="speech speed, e.g. '+15%%' for a faster demo")
    ap.add_argument("--stt-model", default="base",
                    help="faster-whisper size: tiny, base, small, medium")
    ap.add_argument("--language", default="en",
                    help="spoken language code, or 'auto' to detect")
    ap.add_argument("--input-device", default=None,
                    help="microphone index or name substring (see "
                         "--list-devices); defaults to $VOICE_INPUT_DEVICE")
    ap.add_argument("--list-devices", action="store_true",
                    help="print the audio devices and exit")
    ap.add_argument("--quiet", action="store_true",
                    help="hide tool calls (cleaner for the demo)")
    ap.add_argument("--progress", choices=["speak", "print", "none"],
                    default="speak",
                    help="feedback while the model works: 'speak' says what it "
                         "is doing out loud, 'print' shows it on screen only, "
                         "'none' waits in silence for the answer")
    ap.add_argument("--heartbeat", type=float, default=HEARTBEAT_SECONDS,
                    help="seconds of silence during a turn before a filler is "
                         "spoken; 0 disables it")
    ap.add_argument("--server-log", default=DEFAULT_SERVER_LOG,
                    help="file for the server's stderr ('-' to keep it on screen)")
    return ap.parse_args()


def build_transcriber(args: argparse.Namespace):
    """Load Whisper up front, so the first utterance is not slow."""
    from stt import Transcriber

    print(f"loading Whisper '{args.stt_model}'...", flush=True)
    return Transcriber(
        model_size=args.stt_model,
        language=None if args.language == "auto" else args.language,
    )


def get_input_device(raw: str | None) -> int | str | None:
    """Accept either an index ('3') or a name substring ('USB')."""
    if raw is None:
        return None
    return int(raw) if raw.isdigit() else raw


def build_mic(args: argparse.Namespace):
    """Resolve the microphone once, before the conversation starts.

    Probing sample rates opens and closes streams and makes ALSA write to
    stderr, so doing it per utterance would be both slow and noisy.
    """
    from stt import MicConfig

    mic = MicConfig.resolve(get_input_device(args.input_device))
    print(mic.describe())
    return mic


async def main() -> int:
    args = parse_args()

    if args.list_devices:
        from stt import list_input_devices
        print(list_input_devices())
        return 0

    speaker = make_speaker(kind=args.tts, voice=args.voice, rate=args.rate)
    # From here on the narrator owns the speaker: it runs the one thread that
    # is allowed to talk, so progress lines and answers cannot overlap.
    narrator = make_narrator(speaker, mode=args.progress,
                             heartbeat=args.heartbeat)
    try:
        mic = None if args.text else build_mic(args)
    except RuntimeError as exc:
        print(f"\n{exc}\n\nUntil the microphone works, use --text.",
              file=sys.stderr)
        close_voice(narrator, speaker)
        return 3
    transcriber = None if args.text else build_transcriber(args)

    host, model = resolve_target(args)
    # The robot-only server is configured through the environment -- which now
    # actually reaches it, since bridge.py stopped letting the MCP SDK scrub
    # every UR_* variable. The merged server ignores all of this and takes its
    # configuration from the wrapper instead, because sudo strips it anyway.
    os.environ["UR_HOST"], os.environ["UR_MODEL"] = host, model
    # Tell the wrapper which interpreter to use, so the server lands in the
    # same environment as this client instead of guessing at a .venv that a
    # given clone may not have. sudo strips it; the wrapper falls back then.
    os.environ["VOICE_PYTHON"] = sys.executable
    command, *server_args = shlex.split(build_server_command(args, host))
    log_path = None if args.server_log == "-" else args.server_log
    if not args.quiet:
        print(f"server: {command} {' '.join(server_args)}".rstrip())

    if args.backend == "claude":
        try:
            # Probe the SDK itself: claude_backend imports it lazily, so
            # importing that module alone would not catch a missing install
            # and the failure would surface later as a traceback.
            import claude_agent_sdk  # noqa:

            from claude_backend import ClaudeCodeAgent
        except ImportError:
            print("\nThe claude backend needs the Agent SDK:\n"
                  "  pip install claude-agent-sdk\n"
                  "and a logged-in CLI (check: claude --version).\n"
                  "Or run with --backend openai.", file=sys.stderr)
            close_voice(narrator, speaker)
            return 2
        # ``started`` separates a server that never came up from an error
        # later in the conversation: only the former gets the diagnosis, the
        # latter still deserves its real traceback.
        started = False
        try:
            async with ClaudeCodeAgent(command, server_args, model=args.model,
                                       verbose=not args.quiet,
                                       narrator=narrator,
                                       log_path=log_path) as agent:
                started = True
                print(f"connected: {len(agent.tool_names)} tools available "
                      f"(brain: Claude Code subscription)")
                if log_path:
                    print(f"server log: {log_path}")
                print(banner(host, model, args.vision,
                             agent.tool_names))
                await converse(args, agent, narrator, transcriber, mic)
        except Exception:
            if started:
                raise
            report_server_failure(log_path, command, server_args)
            close_voice(narrator, speaker)
            return 4
        close_voice(narrator, speaker)
        print("closed. The MCP server remains usable on its own.")
        return 0

    try:
        llm = LLMClient()
    except ValueError as exc:
        # Missing configuration is the single most common first-run failure;
        # a traceback here teaches nobody anything.
        print(f"\n{exc}\n\n  export AGENT_BASE_URL=...\n"
              "  export AGENT_API_KEY=...\n  export AGENT_MODEL=...\n\n"
              "Free endpoints are listed in ../../llm-client/.\n"
              "Or use the default --backend claude.", file=sys.stderr)
        close_voice(narrator, speaker)
        return 2

    started = False
    try:
        async with connect_tools(command, server_args,
                                 log_path=log_path) as tools:
            schemas = await tools.openai_tools()
            started = True
            agent = VoiceAgent(tools, schemas, llm, verbose=not args.quiet,
                               narrator=narrator)
            print(f"connected: {len(schemas)} tools available "
                  f"(brain: {llm.model})")
            if log_path:
                print(f"server log: {log_path}")
            print(banner(host, model, args.vision,
                         [s['function']['name'] for s in schemas]))
            await converse(args, agent, narrator, transcriber, mic)
    except Exception:
        if started:
            raise
        report_server_failure(log_path, command, server_args)
        close_voice(narrator, speaker)
        return 4

    close_voice(narrator, speaker)
    print("closed. The MCP server remains usable on its own.")
    return 0


# Matched against the server's stderr to turn its own words into the next
# thing to type. Ordered: the first hit wins, so put the specific ones first.
_STARTUP_HINTS = (
    ("a password is required",
     "The launcher went through `sudo -n`, but this machine has no\n"
     "passwordless rule for it. On Linux the camera does not need root:\n"
     "the udev rules give the D435 to your user, so this should not be a\n"
     "sudo call at all. Re-run, or use --no-vision to skip the camera."),
    ("Cannot reach the robot",
     "The arm is not answering. For the simulator:\n"
     "    cd \"simulation environment\" && docker compose up -d\n"
     "then power on and release the brakes at http://localhost.\n"
     "For the real arm, check --robot IP and the network."),
    ("ModuleNotFoundError",
     "The server started, but under an interpreter without its\n"
     "dependencies. Check which python the launcher picked (printed as\n"
     "'server:' above) against the one running this client."),
    ("No such file or directory",
     "A path inside the launcher does not exist on this machine."),
)


def report_server_failure(log_path: str | None, command: str,
                          server_args: list[str]) -> None:
    """Say why the MCP server would not start, instead of dumping a traceback.

    Every startup failure arrives identically -- ``McpError: Connection
    closed``, wrapped in an anyio ExceptionGroup that is sixty lines of
    asyncio plumbing around one line of cause. The server's own stderr already
    explains itself (``server.py`` even names the docker command that fixes
    it), so print that instead and let the operator act on it.
    """
    print("\nThe MCP server did not start, so there are no tools to talk to.",
          file=sys.stderr)
    print(f"  launched: {command} {' '.join(server_args)}".rstrip(),
          file=sys.stderr)

    log = ""
    if log_path and os.path.exists(log_path):
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            log = fh.read().strip()
    if not log:
        print("\nIts stderr was not captured. Re-run with --server-log - to "
              "see it.", file=sys.stderr)
        return

    tail = log.splitlines()[-15:]
    print("\n--- the server said " + "-" * 38, file=sys.stderr)
    for line in tail:
        print(f"  {line}", file=sys.stderr)
    print("-" * 58, file=sys.stderr)

    for needle, hint in _STARTUP_HINTS:
        if needle in log:
            print(f"\n{hint}", file=sys.stderr)
            break


def close_voice(narrator, speaker) -> None:
    """Stop the speaking thread before the mixer it uses goes away."""
    narrator.close()
    speaker.close()


async def converse(args, agent, narrator, transcriber, mic) -> None:
    """The input loop, shared by both backends.

    Both agents expose the same two methods -- ``ask`` and ``emergency_stop``
    -- so nothing below cares which brain is behind them. The narrator is the
    only thing that speaks: the agents feed it progress during the turn, and
    the answer goes through the same thread so the two never overlap.
    """
    while True:
        try:
            utterance = await asyncio.to_thread(
                read_utterance, args, transcriber, mic
            )
        except KeyboardInterrupt:
            print("\ninterrupted -- stopping the robot")
            narrator.cancel()
            await agent.emergency_stop()
            return

        if utterance is None:            # quit
            return
        if utterance == "!":             # hard stop, no LLM in the path
            result = await agent.emergency_stop()
            print(f"     {result[:200]}")
            # finish(), not a bare say(): it discards any progress line still
            # queued, so the robot does not narrate a move it just abandoned.
            await asyncio.to_thread(narrator.finish, STOP_ACKNOWLEDGED)
            continue
        if not utterance:                # silence or misfire
            print("  (did not catch that, try again)")
            continue

        print(f"you> {utterance}")
        try:
            reply = await agent.ask(utterance)
        except KeyboardInterrupt:
            print("\ninterrupted -- stopping the robot")
            narrator.cancel()
            await agent.emergency_stop()
            return
        except Exception as exc:
            reply = f"Error: {exc}"
        # Synthesis and playback block for seconds; keep them off the event
        # loop so the MCP subprocess pipes stay serviced meanwhile. finish()
        # returns once the line has actually been played, which is what stops
        # the microphone from opening while the speakers are still talking.
        await asyncio.to_thread(narrator.finish, reply)


def read_utterance(args: argparse.Namespace, transcriber, mic) -> str | None:
    """One turn of input. Returns None to quit, '!' to stop, '' for silence.

    Runs on a worker thread (both ``input()`` and Whisper block), which keeps
    the event loop free to service the MCP subprocess pipes.
    """
    from stt import record_until_enter, wait_for_key

    if args.text:
        line = wait_for_key("\nyou> ")
        if line in ("q", "quit", "exit"):
            return None
        return line

    line = wait_for_key("\n[ENTER] speak  |  [!] stop  |  [q] quit > ")
    if line in ("q", "quit", "exit"):
        return None
    if line == "!":
        return "!"

    audio = record_until_enter(mic)
    print("  transcribing...", flush=True)
    return transcriber.transcribe(audio)


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        sys.exit(130)
