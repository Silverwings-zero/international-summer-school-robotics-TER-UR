"""Text to speech: Microsoft Edge neural voices, played through pygame.

``edge-tts`` reaches Microsoft's public endpoint, so the voices are neural and
free but **require an internet connection**. That is a real risk on conference
wifi, so every failure here degrades to printing the line instead of raising:
losing the voice must never stop the robot conversation.

Playback goes through ``pygame.mixer``, which decodes MP3 in-process. No
ffmpeg, no external player, no temp file -- the MP3 bytes go straight from the
socket into a BytesIO and out of the speakers.
"""
from __future__ import annotations

import asyncio
import io
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

# pygame greets stdout on import. Harmless in a terminal, fatal in an MCP
# server where stdout is the protocol, so it is silenced for every caller.
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

# English neural voices worth trying. Aria is clear and neutral, which carries
# best over a noisy room; Guy is the calmer male alternative. Any locale works
# -- run ``edge-tts --list-voices`` for the full list.
VOICE_EN_FEMALE = "en-US-AriaNeural"
VOICE_EN_MALE = "en-US-GuyNeural"
VOICE_EN_GB = "en-GB-SoniaNeural"

DEFAULT_VOICE = VOICE_EN_FEMALE

# Spoken replies should be short. Anything longer is almost certainly the model
# ignoring its instructions, and reading it aloud would stall the demo.
MAX_SPOKEN_CHARS = 600


def _run_coroutine(coro):
    """Run a coroutine to completion from either sync or async context.

    ``asyncio.run`` raises if a loop is already running in this thread, which
    is exactly the case in ``run_voice.py`` -- ``say()`` is reached from inside
    the async input loop. Handing the coroutine to a private thread gives it a
    thread with no running loop, so the same call works everywhere.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)          # no loop here: run it directly
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def clean_for_speech(text: str) -> str:
    """Strip anything that sounds wrong when read aloud.

    Models leak markdown, code fences and raw JSON into chat replies. A TTS
    engine will happily pronounce every asterisk and brace, so they are removed
    here rather than hoped away in the prompt.
    """
    text = re.sub(r"```.*?```", " ", text, flags=re.S)       # code fences
    text = re.sub(r"`([^`]*)`", r"\1", text)                  # inline code
    text = re.sub(r"^\s*[-*•]\s+", "", text, flags=re.M)  # list bullets
    text = re.sub(r"\{[^{}]*\}", " ", text)                   # stray JSON
    # Underscores become spaces on purpose: a leaked tool name should be read
    # as "move linear", not spelled out with underscores.
    text = re.sub(r"[*_#>]+", " ", text)                      # markdown marks
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > MAX_SPOKEN_CHARS:
        text = text[:MAX_SPOKEN_CHARS].rsplit(" ", 1)[0] + "..."
    return text


class NullSpeaker:
    """Prints instead of speaking. The fallback, and useful for debugging."""

    name = "none"

    def say(self, text: str) -> None:
        # stderr, never stdout: voice_mcp.py serves the MCP protocol on stdout,
        # and a stray line there corrupts the session. Terminal users still see
        # this, because stderr goes to the same terminal.
        print(f"robot> {text}", file=sys.stderr)

    def close(self) -> None:
        pass


class EdgeSpeaker:
    """Speaks with an Edge neural voice.

    Args:
        voice: An Edge voice id. ``edge-tts --list-voices`` shows them all.
        rate: Speed adjustment as a signed percentage, e.g. "+15%" to speed the
            demo up, "-10%" to make it easier to follow in a noisy room.
    """

    name = "edge"

    def __init__(self, voice: str = DEFAULT_VOICE, rate: str = "+0%") -> None:
        import pygame

        self.voice = voice
        self.rate = rate
        self._pygame = pygame
        # 24 kHz matches the MP3 edge-tts returns, so SDL does not resample.
        pygame.mixer.init(frequency=24_000)

    async def _synthesize(self, text: str) -> bytes:
        import edge_tts

        communicate = edge_tts.Communicate(text, self.voice, rate=self.rate)
        buffer = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buffer.write(chunk["data"])
        return buffer.getvalue()

    def say(self, text: str) -> None:
        """Speak one line, blocking until playback finishes.

        Blocking is deliberate: the next microphone capture must not start
        while the speakers are still talking, or the robot transcribes itself.
        """
        spoken = clean_for_speech(text)
        if not spoken:
            return
        print(f"robot> {spoken}", file=sys.stderr)   # never stdout, see above
        try:
            audio = _run_coroutine(self._synthesize(spoken))
            if not audio:
                return
            self._pygame.mixer.music.load(io.BytesIO(audio), "mp3")
            self._pygame.mixer.music.play()
            while self._pygame.mixer.music.get_busy():
                self._pygame.time.wait(50)
        except Exception as exc:  # offline, endpoint down, no audio sink
            print(f"  (voice unavailable: {exc})", file=sys.stderr)

    def close(self) -> None:
        try:
            self._pygame.mixer.quit()
        except Exception:
            pass


def make_speaker(kind: str = "edge", voice: str = DEFAULT_VOICE,
                 rate: str = "+0%"):
    """Build a speaker, degrading to :class:`NullSpeaker` if audio is missing.

    Args:
        kind: "edge" for neural voices, "none" to print replies instead.
        voice: Edge voice id, ignored when kind is "none".
        rate: Speed adjustment, e.g. "+10%".
    """
    if kind == "none":
        return NullSpeaker()
    try:
        return EdgeSpeaker(voice=voice, rate=rate)
    except Exception as exc:
        # No audio sink (headless box, occupied device, no SDL): keep working
        # in text, but say why -- silent failure here looks like a dead robot.
        print(f"  (no audio output, falling back to printed replies: {exc})",
              file=sys.stderr)
        return NullSpeaker()
