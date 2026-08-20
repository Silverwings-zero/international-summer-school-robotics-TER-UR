"""Speech to text: push-to-talk capture plus faster-whisper transcription.

Three pieces, deliberately separate so each can be tested alone:

    MicConfig.resolve()    find a microphone AND a sample rate it will open at
    record_until_enter()   capture audio between two ENTER presses
    Transcriber.transcribe()   turn that audio into text

Design notes, because they are the difference between this working on every
group member's laptop and only on one:

* **The sample rate is negotiated, not assumed.** Whisper wants 16 kHz, but
  most ALSA hardware devices refuse to open at it -- a raw ``hw:0,0`` capture
  device typically supports only 44.1/48 kHz and answers anything else with
  ``paInvalidSampleRate``. So we ask the device what it supports, record at its
  own rate, and resample to 16 kHz in numpy afterwards.
* **No temporary WAV files.** ``sounddevice`` hands us float32 samples and
  ``faster-whisper`` accepts a numpy array directly, so the audio never touches
  the disk. That removes a class of "permission denied in /tmp" problems and
  saves ~100 ms per utterance.
* **No ``keyboard`` library.** It reads /dev/input and therefore needs root on
  Linux. Gating on ENTER through plain stdin needs no privileges, no X11, and
  survives running over SSH.
* **Silence is rejected before it reaches Whisper.** Whisper hallucinates
  confident sentences out of near-silence -- with an English decoder it likes
  to emit "Thank you for watching!". An RMS gate plus the built-in VAD filter
  stops that from ever reaching the robot as a command.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass

import numpy as np

# What Whisper is trained on. Audio is resampled to this before decoding,
# whatever the sound card gave us.
SAMPLE_RATE = 16_000

# Capture rates to try, in order, after the device's own preferred rate.
# 48 kHz first: it is what nearly all built-in laptop codecs run at natively.
RATE_CANDIDATES = (48_000, 44_100, 32_000, 16_000, 22_050, 8_000)

# Utterances shorter or quieter than these are treated as a misfire (a stray
# ENTER, a bumped table) rather than transcribed.
MIN_DURATION_S = 0.35
MIN_RMS = 0.004

# Set VOICE_INPUT_DEVICE to pin a microphone once for every entry point --
# run_voice.py, check_audio.py and voice_mcp.py all honour it, so a laptop
# whose default device is the wrong one is configured in a single place.
DEVICE_ENV_VAR = "VOICE_INPUT_DEVICE"

# Voice-activity detection, used by record_until_silence() when there is no
# keyboard to press (the MCP server case).
VAD_BLOCK_S = 0.05          # analysis block; also the polling interval
VAD_NOISE_SAMPLE_S = 0.30   # leading audio measured to learn the room's floor
VAD_NOISE_FACTOR = 4.0      # speech must exceed this multiple of that floor
VAD_SILENCE_S = 1.2         # hang-up delay after the talking stops
VAD_START_TIMEOUT_S = 6.0   # give up if nobody starts speaking


def device_from_env() -> int | str | None:
    """Read VOICE_INPUT_DEVICE, as an index when it looks like one."""
    raw = os.environ.get(DEVICE_ENV_VAR)
    if raw is None or not raw.strip():
        return None
    raw = raw.strip()
    return int(raw) if raw.isdigit() else raw


def list_input_devices() -> str:
    """Human-readable table of the audio devices, for --list-devices."""
    import sounddevice as sd

    return str(sd.query_devices())


# --------------------------------------------------------------------------- #
# Choosing a microphone that actually opens
# --------------------------------------------------------------------------- #
def _candidate_devices(device: int | str | None) -> list:
    """Devices to try, best first.

    An explicit choice is honoured and nothing else is attempted -- if someone
    passes --input-device they want to know it failed, not to be silently
    redirected elsewhere.

    Otherwise: the system default first, then ALSA's plug layers
    ("sysdefault", "default", "pulse", "pipewire"), which resample in the
    driver and therefore accept any rate, then every remaining input.
    """
    import sounddevice as sd

    if device is not None:
        return [device]

    devices = sd.query_devices()
    order: list = []
    try:
        default_in = sd.default.device[0]
        if default_in is not None and default_in >= 0:
            order.append(default_in)
    except Exception:
        pass

    plug_layers = ("sysdefault", "default", "pulse", "pipewire")
    for index, info in enumerate(devices):
        if info["max_input_channels"] > 0 and \
                any(k in info["name"].lower() for k in plug_layers):
            order.append(index)
    for index, info in enumerate(devices):
        if info["max_input_channels"] > 0:
            order.append(index)

    return list(dict.fromkeys(order))


@dataclass
class MicConfig:
    """A microphone and a sample rate that are known to open together.

    Resolve this ONCE at startup, not per utterance: probing rates opens and
    closes streams, which is slow and makes ALSA print to stderr.
    """

    device: int | str | None
    rate: int
    name: str = ""

    @classmethod
    def resolve(cls, device: int | str | None = None,
                preferred_rate: int = SAMPLE_RATE) -> "MicConfig":
        """Find a working (device, rate) pair, or explain why none exists.

        Args:
            device: Index, name substring, or None to fall back to the
                VOICE_INPUT_DEVICE environment variable, and then to an
                automatic search.
            preferred_rate: Tried first; recording at 16 kHz directly saves the
                resampling step when the hardware happens to allow it.

        Raises:
            RuntimeError: No device opened at any rate, with the per-device
                reasons collected so the message is actionable.
        """
        import sounddevice as sd

        if device is None:
            device = device_from_env()

        problems: list[str] = []
        for candidate in _candidate_devices(device):
            try:
                info = sd.query_devices(candidate)
            except Exception as exc:
                problems.append(f"  device {candidate}: {exc}")
                continue
            if info["max_input_channels"] < 1:
                problems.append(f"  {info['name']}: no input channels")
                continue

            native = int(info.get("default_samplerate") or 0)
            rates = [r for r in (preferred_rate, native, *RATE_CANDIDATES) if r]
            for rate in dict.fromkeys(rates):
                try:
                    sd.check_input_settings(device=candidate, channels=1,
                                            dtype="float32", samplerate=rate)
                    return cls(device=candidate, rate=int(rate),
                               name=info["name"])
                except Exception:
                    continue
            problems.append(
                f"  {info['name']}: opens at none of "
                f"{', '.join(str(r) for r in dict.fromkeys(rates))} Hz"
            )

        raise RuntimeError(
            "No usable microphone found.\n" + "\n".join(problems) +
            f"\n\nList the devices with --list-devices and pick one "
            f"explicitly with --input-device <index>, or set "
            f"{DEVICE_ENV_VAR}=<index> to pin it for every entry point. On "
            f"Linux the 'sysdefault' entry is usually the safe choice, because "
            f"ALSA resamples for you."
        )

    def describe(self) -> str:
        """One line for the startup banner and the self-test."""
        extra = "" if self.rate == SAMPLE_RATE else \
            f", resampled to {SAMPLE_RATE} Hz"
        return f"microphone: {self.name or self.device} @ {self.rate} Hz{extra}"


# --------------------------------------------------------------------------- #
# Capture
# --------------------------------------------------------------------------- #
def resample_to(audio: np.ndarray, src_rate: int,
                dst_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Resample mono float32 audio, numpy only.

    Downsampling is low-passed with a moving average first, otherwise the
    frequencies above the new Nyquist alias down into the speech band and
    Whisper's accuracy drops. A box filter is crude next to a proper polyphase
    FIR, but it is inaudible here and costs no extra dependency (scipy would be
    ~30 MB for this one line).
    """
    if audio.size == 0 or src_rate == dst_rate:
        return audio.astype(np.float32, copy=False)

    if src_rate > dst_rate:
        width = max(1, int(round(src_rate / dst_rate)))
        if width > 1:
            kernel = np.ones(width, dtype=np.float32) / width
            audio = np.convolve(audio, kernel, mode="same")

    count = int(round(audio.size * dst_rate / src_rate))
    if count < 1:
        return np.zeros(0, dtype=np.float32)
    source = np.arange(audio.size, dtype=np.float64)
    target = np.linspace(0.0, audio.size - 1.0, count)
    return np.interp(target, source, audio).astype(np.float32)


def wait_for_key(prompt: str) -> str:
    """Block until ENTER, returning whatever was typed before it, lowercased.

    The return value is what lets the caller offer single-key commands (stop,
    quit) on the same prompt that starts a recording, with no key-event
    library and no second input channel.
    """
    try:
        return input(prompt).strip().lower()
    except EOFError:  # stdin closed (piped input): behave like "quit"
        return "q"


def record_until_enter(mic: MicConfig) -> np.ndarray:
    """Record from the microphone until ENTER is pressed.

    The PortAudio callback fills the buffer on its own thread, which is exactly
    why the main thread is free to block on ``input()``. That is the whole
    push-to-talk mechanism -- there is no polling and no key-event library.

    Args:
        mic: A resolved device/rate pair from :meth:`MicConfig.resolve`.

    Returns:
        Mono float32 samples at :data:`SAMPLE_RATE`, in [-1, 1]. Empty array if
        nothing was captured.
    """
    import sounddevice as sd

    frames: list[np.ndarray] = []

    def _callback(indata, _frames, _time, status) -> None:
        if status:  # overflows are survivable; report and keep going
            print(f"  (audio status: {status})", file=sys.stderr)
        frames.append(indata.copy())

    with sd.InputStream(
        samplerate=mic.rate,
        channels=1,
        dtype="float32",
        device=mic.device,
        callback=_callback,
    ):
        try:
            input("  recording... [ENTER] to send")
        except EOFError:
            pass

    if not frames:
        return np.zeros(0, dtype=np.float32)
    audio = np.concatenate(frames, axis=0).reshape(-1)
    return resample_to(audio, mic.rate)


def record_push_to_talk(mic: MicConfig | None = None) -> np.ndarray:
    """Convenience combo: wait for ENTER, record, stop on the next ENTER."""
    mic = mic or MicConfig.resolve()
    wait_for_key("\n[ENTER] to speak")
    return record_until_enter(mic)


def is_usable(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bool:
    """True when the clip is long enough and loud enough to be real speech."""
    if audio.size < MIN_DURATION_S * sample_rate:
        return False
    return float(np.sqrt(np.mean(np.square(audio)))) >= MIN_RMS


# --------------------------------------------------------------------------- #
# Transcription
# --------------------------------------------------------------------------- #
class Transcriber:
    """faster-whisper wrapper, loaded once and reused for the whole session.

    The model load takes a few seconds and downloads ~75 MB on first use
    (cached afterwards in ~/.cache/huggingface). Build this before the
    conversation starts, never inside the loop.
    """

    def __init__(
        self,
        model_size: str = "base",
        language: str | None = "en",
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        """Load the model.

        Args:
            model_size: "tiny", "base", "small", "medium", "large-v3". "base"
                is the sweet spot on a laptop CPU: ~150 MB RAM, roughly
                real-time. Drop to "tiny" on very slow machines, go to "small"
                if it mishears cooking vocabulary.
            language: ISO code forced on the decoder ("en", "it"), or None to
                auto-detect. Forcing it is both faster and more accurate;
                auto-detection is unreliable on short commands.
            device: "cpu", or "cuda" if the laptop has an NVIDIA GPU.
            compute_type: "int8" quantisation for CPU (fastest, negligible
                accuracy loss here); use "float16" on CUDA.
        """
        from faster_whisper import WhisperModel

        self.language = language
        self.model = WhisperModel(model_size, device=device,
                                  compute_type=compute_type)

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe mono 16 kHz float32 audio into a single line of text.

        Returns an empty string when the clip is silence, too short, or the
        decoder produced nothing -- callers should treat that as "say again"
        rather than as a command.
        """
        if not is_usable(audio):
            return ""
        segments, _info = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=1,        # greedy: the quality cost is invisible here,
                                # the latency saving is not
            vad_filter=True,    # drop non-speech before decoding
            condition_on_previous_text=False,  # stops repetition loops
        )
        return " ".join(s.text.strip() for s in segments).strip()


def record_until_silence(
    mic: MicConfig,
    max_seconds: float = 15.0,
    silence_s: float = VAD_SILENCE_S,
    start_timeout_s: float = VAD_START_TIMEOUT_S,
) -> np.ndarray:
    """Record until the speaker stops talking, with no key to press.

    ``record_until_enter`` cannot be used everywhere: inside an MCP server,
    stdin carries the protocol, so there is no keyboard to gate on. This is the
    alternative -- start capturing, wait for speech, and hang up once it stops.

    The threshold adapts to the room. The first :data:`VAD_NOISE_SAMPLE_S` of
    audio is measured as the noise floor and speech must exceed
    :data:`VAD_NOISE_FACTOR` times it, so a noisy kitchen raises the bar
    instead of triggering on the extractor fan. A fixed threshold would need
    re-tuning at every venue.

    Args:
        mic: A resolved device/rate pair from :meth:`MicConfig.resolve`.
        max_seconds: Hard cap, so a stuck-open microphone cannot block forever.
        silence_s: How long the quiet must last before the recording ends.
        start_timeout_s: Give up if speech never starts.

    Returns:
        Mono float32 samples at :data:`SAMPLE_RATE`. Empty when nobody spoke.
    """
    import sounddevice as sd

    frames: list[np.ndarray] = []
    blocks: list[float] = []          # per-block RMS, for the noise floor
    state = {"started": False, "last_loud": 0.0, "threshold": 0.0}
    begin = time.monotonic()
    noise_blocks = max(1, int(VAD_NOISE_SAMPLE_S / VAD_BLOCK_S))

    def _callback(indata, _frames, _time, status) -> None:
        if status:
            print(f"  (audio status: {status})", file=sys.stderr)
        frames.append(indata.copy())
        rms = float(np.sqrt(np.mean(np.square(indata))))
        blocks.append(rms)

        if len(blocks) <= noise_blocks:      # still learning the room
            return
        if state["threshold"] == 0.0:
            floor = float(np.median(blocks[:noise_blocks]))
            state["threshold"] = max(MIN_RMS, floor * VAD_NOISE_FACTOR)
        if rms >= state["threshold"]:
            state["started"] = True
            state["last_loud"] = time.monotonic()

    with sd.InputStream(
        samplerate=mic.rate,
        channels=1,
        dtype="float32",
        device=mic.device,
        blocksize=int(mic.rate * VAD_BLOCK_S),
        callback=_callback,
    ):
        while True:
            time.sleep(VAD_BLOCK_S)
            now = time.monotonic()
            if now - begin >= max_seconds:
                break
            if state["started"]:
                if now - state["last_loud"] >= silence_s:
                    break
            elif now - begin >= start_timeout_s:
                return np.zeros(0, dtype=np.float32)

    if not frames:
        return np.zeros(0, dtype=np.float32)
    audio = np.concatenate(frames, axis=0).reshape(-1)
    return resample_to(audio, mic.rate)
