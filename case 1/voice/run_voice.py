"""Entry point: talk to the robot out loud.

    python voice/run_voice.py                       # simulator + camera
    python voice/run_voice.py --robot 192.168.1.100 # the real UR5e + camera
    python voice/run_voice.py --no-vision           # robot only, cannot see
    python voice/run_voice.py --text                # typed in (no microphone)
    python voice/run_voice.py --tts none            # typed in, printed out
    python voice/run_voice.py --progress none       # no commentary
    python voice/run_voice.py --list-devices        # which microphone is which

WHICH ARM, AND WHETHER IT CAN SEE. Both follow from two flags, because the
pairing is a safety matter rather than a preference: ``--robot IP`` means the
real arm and therefore a ur5e safety envelope, and no ``--robot`` means the
local simulator and its ur10e. Guarding a real UR5e with UR10e geometry would
approve targets 40 cm past its reach, so ``server.py`` refuses that pairing
outright. ``--vision`` (the default) launches the merged camera+robot server
through ``run_vision_root.sh`` under sudo -- the only way librealsense can
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
_REPO = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))

# The robot-only server: 43 tools, no root needed, and NO camera. Launched with
# the interpreter running this script so it inherits the same virtualenv on
# machines with no bare `python` on PATH.
_ROBOT_ONLY_SERVER = os.path.abspath(
    os.path.join(_HERE, os.pardir, "server.py"))
# The merged server: those same 43 tools plus the 16 wrist-camera ones. It has
# to go through the wrapper rather than being invoked directly, because
# librealsense cannot claim the D435 on macOS without root -- and sudo strips
# the environment, which is why UR_VISION and every UR_*/VISION_* setting live
# INSIDE the wrapper and the robot arrives as its argument.
_MERGED_SERVER = os.path.join(_REPO, "run_vision_root.sh")


def resolve_target(args) -> tuple[str, str]:
    """(host, model) this run will drive, mirroring run_vision_root.sh's rule.

    No ``--robot`` means the local simulator and its UR10e; an IP means the
    real arm, which is a UR5e in this cell. Deriving both from one flag is what
    stops the host and the model from disagreeing: a real UR5e guarded by
    UR10e geometry gets a safety layer that approves targets 40 cm beyond its
    reach and 7.5 kg past its payload, which is why server.py now refuses that
    combination outright.
    """
    if args.robot:
        return args.robot, (args.ur_model or "ur5e")
    return "127.0.0.1", (args.ur_model or "ur10e")


def build_server_command(args) -> str:
    """The MCP server command line, unless --server overrode it outright."""
    if args.server:
        return args.server
    if not args.vision:
        return " ".join(shlex.quote(p) for p in
                        (sys.executable, _ROBOT_ONLY_SERVER))
    # The wrapper takes the robot as its first argument and an optional model
    # override as its second; no arguments at all means the local simulator.
    parts = ["/usr/bin/sudo", "-n", _MERGED_SERVER]
    if args.robot or args.ur_model:
        parts.append(args.robot or "")   # empty -> the simulator branch
        if args.ur_model:
            parts.append(args.ur_model)
    return " ".join(shlex.quote(p) for p in parts)


def banner(host: str, model: str, vision: bool) -> str:
    """The header, naming the arm and the tools this session actually got.

    The old banner said "UR10e" unconditionally, which was exactly wrong on the
    day it mattered -- the defaults reached the real UR5e. Print what was
    resolved instead, so the operator sees the target before speaking to it.
    """
    where = ("simulator" if host == "localhost" or host.startswith("127.")
             else f"REAL ROBOT at {host}")
    eyes = "camera + robot" if vision else "robot only -- NO camera"
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
                         "192.168.1.100); omit it for the local simulator")
    ap.add_argument("--ur-model", default=None,
                    choices=["ur10e", "ur5e", "ur5"],
                    help="which ARM the safety layer guards (not the LLM: "
                         "that is --model); defaults to ur5e with --robot "
                         "and ur10e without it")
    ap.add_argument("--vision", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="launch the merged camera+robot server via "
                         "run_vision_root.sh (needs the passwordless sudo "
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
    command, *server_args = shlex.split(build_server_command(args))
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
        async with ClaudeCodeAgent(command, server_args, model=args.model,
                                   verbose=not args.quiet,
                                   narrator=narrator) as agent:
            print(f"connected: {len(agent.tool_names)} tools available "
                  f"(brain: Claude Code subscription)")
            print(banner(host, model, args.vision))
            await converse(args, agent, narrator, transcriber, mic)
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

    async with connect_tools(command, server_args, log_path=log_path) as tools:
        schemas = await tools.openai_tools()
        agent = VoiceAgent(tools, schemas, llm, verbose=not args.quiet,
                           narrator=narrator)
        print(f"connected: {len(schemas)} tools available "
              f"(brain: {llm.model})")
        if log_path:
            print(f"server log: {log_path}")
        print(banner(host, model, args.vision))
        await converse(args, agent, narrator, transcriber, mic)

    close_voice(narrator, speaker)
    print("closed. The MCP server remains usable on its own.")
    return 0


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
