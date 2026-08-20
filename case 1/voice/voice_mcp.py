"""MCP server exposing the microphone and the speaker as tools.

This is the answer to "I drive the robot from Claude Code -- how do I add voice
there?". You cannot inject speech into Claude Code's own loop: Claude Code owns
the conversation. So instead of wrapping it, voice is offered to it as two more
tools, next to the robot's:

    listen()      record the operator and return what they said
    speak(text)   say something out loud
    voice_check() report which microphone and voice are in use

Add this server alongside ``ur-tools`` (see voice/README.md), then tell Claude
Code to "start voice mode". It will call ``listen``, act on the robot with the
``ur-tools`` tools, ``speak`` the result, and ``listen`` again.

Two constraints shape this file, and breaking either one breaks the protocol:

* **stdout is the transport.** Every byte written to stdout must be MCP JSON.
  ``pygame`` prints a banner on import and ``tts.say`` prints the spoken line,
  so all of it is redirected to stderr. A stray ``print`` here corrupts the
  session in a way that looks like a hung client.
* **stdin is the transport too.** There is no keyboard to gate on, so
  ``listen`` cannot use push-to-talk. It records until the speaker stops,
  using the adaptive voice-activity detector in ``stt.py``.
"""
from __future__ import annotations

import contextlib
import os
import sys
import threading

# pygame greets stdout on import unless this is set before it loads.
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastmcp import FastMCP

import stt
from tts import DEFAULT_VOICE, make_speaker

mcp = FastMCP("voice-io")

# Whisper needs a few seconds and ~75 MB to load. Doing it at import would
# delay the client's startup handshake, so it is built on first use and then
# reused; the lock stops two concurrent calls loading it twice.
_LOCK = threading.Lock()
_STATE: dict = {"mic": None, "transcriber": None, "speaker": None}

VOICE = os.environ.get("VOICE_TTS_VOICE", DEFAULT_VOICE)
LANGUAGE = os.environ.get("VOICE_LANGUAGE", "en")
MODEL_SIZE = os.environ.get("VOICE_STT_MODEL", "base")


@contextlib.contextmanager
def _stdout_to_stderr():
    """Shield the transport while a third-party library initialises.

    ``contextlib.redirect_stdout`` swaps ``sys.stdout`` for the WHOLE process,
    so this must wrap the shortest possible span. Holding it across a 15 second
    recording would send any MCP response written in that window to stderr, and
    the client would simply wait forever. It is therefore used only around
    one-time library setup (pygame opening a device, Whisper loading), never
    around the recording or playback itself -- our own code writes to stderr
    directly instead.
    """
    with contextlib.redirect_stdout(sys.stderr):
        yield


def _mic() -> stt.MicConfig:
    if _STATE["mic"] is None:
        with _stdout_to_stderr():
            _STATE["mic"] = stt.MicConfig.resolve()
    return _STATE["mic"]


def _transcriber() -> stt.Transcriber:
    if _STATE["transcriber"] is None:
        with _stdout_to_stderr():
            _STATE["transcriber"] = stt.Transcriber(
                model_size=MODEL_SIZE,
                language=None if LANGUAGE == "auto" else LANGUAGE,
            )
    return _STATE["transcriber"]


def _speaker():
    if _STATE["speaker"] is None:
        with _stdout_to_stderr():
            _STATE["speaker"] = make_speaker(kind="edge", voice=VOICE)
    return _STATE["speaker"]


@mcp.tool
def listen(max_seconds: float = 15.0) -> dict:
    """Listen to the operator and return what they said.

    Recording starts immediately and stops on its own once the person stops
    talking, so tell them you are listening (with ``speak``) before calling
    this, or they will miss the start of the window.

    Call this again after every ``speak`` to keep the conversation going, until
    the operator says they are finished.

    Args:
        max_seconds: Hard cap on one recording, 1 to 60 seconds. Raise it only
            if the operator is expected to say something long.

    Returns:
        A dict with ``heard`` (the transcribed text, empty if nobody spoke) and
        ``status``: "ok" when there is speech, "silence" when the window
        expired with nothing said -- in that case say something and listen
        again rather than calling repeatedly in a tight loop.

    Raises:
        ValueError: max_seconds out of range.
        RuntimeError: No usable microphone; the message says how to pick one.
    """
    if not 1.0 <= max_seconds <= 60.0:
        raise ValueError(
            f"max_seconds must be between 1 and 60, got {max_seconds}."
        )
    with _LOCK:
        mic = _mic()
        audio = stt.record_until_silence(mic, max_seconds=float(max_seconds))
        if audio.size == 0:
            return {"heard": "", "status": "silence",
                    "note": "Nobody spoke within the window."}
        text = _transcriber().transcribe(audio)
    if not text:
        return {"heard": "", "status": "silence",
                "note": "Only background noise was captured."}
    return {"heard": text, "status": "ok",
            "seconds_recorded": round(audio.size / stt.SAMPLE_RATE, 1)}


@mcp.tool
def speak(text: str) -> dict:
    """Say something out loud to the operator.

    The text is read aloud, so write it to be heard: one or two short
    sentences, no markdown, no lists, no JSON, no tool names, and numbers
    rounded the way a person would say them ("about twenty centimetres").

    Blocks until the speaker has finished, so a following ``listen`` does not
    record the robot talking to itself.

    Args:
        text: What to say, 1 to 600 characters.

    Returns:
        A dict with ``spoken`` (the text as it was actually pronounced, after
        markdown is stripped) and the ``voice`` used.

    Raises:
        ValueError: Empty text, or longer than 600 characters.
    """
    if not text or not text.strip():
        raise ValueError("Nothing to say: text is empty.")
    if len(text) > 600:
        raise ValueError(
            f"text is {len(text)} characters; keep spoken replies under 600. "
            "Say the short version instead."
        )
    with _LOCK:
        speaker = _speaker()
        speaker.say(text)
        from tts import clean_for_speech
        result = {"spoken": clean_for_speech(text), "voice": VOICE,
                  "engine": speaker.name}
        if speaker.name == "none":
            # Nothing was audible. Say so, or the model reports success while
            # the operator hears silence and concludes the robot is dead.
            result["status"] = "not_audible"
            result["note"] = ("No audio output device is available, so this "
                              "was NOT spoken aloud. Tell the operator to "
                              "check their speakers.")
        return result


@mcp.tool
def voice_check() -> dict:
    """Report the microphone, model and voice this server will use.

    Call it once before starting voice mode to confirm the audio is set up, or
    when ``listen`` keeps returning silence.
    """
    with _LOCK:
        try:
            mic = _mic()
            microphone = mic.describe()
        except RuntimeError as exc:
            microphone = f"UNAVAILABLE: {exc}"
        return {
            "microphone": microphone,
            "input_device_env": os.environ.get(stt.DEVICE_ENV_VAR, "(unset)"),
            "whisper_model": MODEL_SIZE,
            "language": LANGUAGE,
            "voice": VOICE,
            "note": "Set VOICE_INPUT_DEVICE, VOICE_TTS_VOICE, VOICE_LANGUAGE "
                    "or VOICE_STT_MODEL in the MCP server config to change "
                    "these.",
        }


def _selftest() -> int:
    """Exercise the microphone and the speaker without any MCP client.

    Running an stdio server by hand looks like a hang -- it prints a banner and
    then waits silently for JSON-RPC that never comes. This gives the obvious
    command something useful to do, and proves the audio half works before the
    server is ever wired into a client.
    """
    print("voice-io self-test (no MCP client involved)\n", file=sys.stderr)
    try:
        mic = _mic()
        print(f"  microphone : {mic.describe()}", file=sys.stderr)
    except RuntimeError as exc:
        print(f"  microphone : UNAVAILABLE\n{exc}", file=sys.stderr)
        return 1

    speaker = _speaker()
    print(f"  speaker    : {speaker.name}"
          f"{' (NO AUDIO OUTPUT)' if speaker.name == 'none' else ''}",
          file=sys.stderr)
    speaker.say("Voice check. Say something after the beep.")

    print("\n  listening for 5 seconds, speak now...", file=sys.stderr)
    audio = stt.record_until_silence(_mic(), max_seconds=5.0)
    if audio.size == 0:
        print("  heard      : (nothing)", file=sys.stderr)
        print("\nThe microphone captured no speech. Set VOICE_INPUT_DEVICE "
              "to the right device and try again.", file=sys.stderr)
        return 1
    text = _transcriber().transcribe(audio)
    print(f"  heard      : {text!r}", file=sys.stderr)
    if not text:
        print("\nSound was captured but no words were recognised. Usually the "
              "wrong microphone:\nset VOICE_INPUT_DEVICE and run this again.",
              file=sys.stderr)
        return 1
    speaker.say(f"I heard: {text}")
    print("\nAudio works. Now add this server to your .mcp.json -- see "
          "voice/README.md section 4b.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())

    # Started by hand in a terminal? Say what is about to happen, because a
    # silent stdio server is indistinguishable from a hung one.
    if sys.stdin.isatty():
        print(
            "\nThis is an MCP server. It is now waiting for a client to talk "
            "JSON-RPC on stdin,\nso nothing more will happen here -- that is "
            "normal, not a hang.\n\n"
            "  To test the audio directly :  python voice_mcp.py --selftest\n"
            "  To use it from Claude Code :  add it to .mcp.json, see "
            "voice/README.md section 4b\n\n"
            "  Ctrl-C to quit.\n",
            file=sys.stderr,
        )
    mcp.run()
