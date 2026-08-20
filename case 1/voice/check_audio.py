"""Self-test for the voice stack. Run this FIRST, before run_voice.py.

    python voice/check_audio.py
    python voice/check_audio.py --device 7        # force a microphone
    python voice/check_audio.py --voice en-GB-SoniaNeural

Checks each piece independently and prints OK or FAIL with the reason, so a
group member who cannot get voice working can paste one output instead of
guessing. It touches neither the robot nor the LLM: no simulator, no API key,
no network except the Edge TTS endpoint in step 5.

Steps:
    1  imports      are the packages installed
    2  devices      which microphone and sample rate actually open
    3  record       does that microphone capture sound
    4  transcribe   does Whisper turn the sound into text
    5  speak        does Edge TTS come back and play
"""
from __future__ import annotations

import argparse
import sys

PASS = "  OK  "
FAIL = " FAIL "

TEST_PHRASE = "Hey I'm your chef robot, how can I help you today?."


def step(n: int, title: str) -> None:
    print(f"\n[{n}] {title}")


def ok(msg: str) -> None:
    print(f"[{PASS}] {msg}")


def fail(msg: str, hint: str = "") -> None:
    print(f"[{FAIL}] {msg}")
    if hint:
        print(f"        -> {hint}")


def check_imports() -> bool:
    step(1, "packages")
    missing = []
    for module, package, hint in [
        ("numpy", "numpy", ""),
        ("sounddevice", "sounddevice",
         "PortAudio is also needed: sudo apt install -y libportaudio2"),
        ("faster_whisper", "faster-whisper", ""),
        ("edge_tts", "edge-tts", ""),
        ("pygame", "pygame", ""),
    ]:
        try:
            __import__(module)
            ok(package)
        except Exception as exc:
            fail(f"{package}: {exc}", hint or f"pip install {package}")
            missing.append(package)
    return not missing


def check_devices(device):
    """List the devices, then negotiate one that really opens."""
    step(2, "audio devices")
    try:
        import sounddevice as sd

        print(sd.query_devices())
    except Exception as exc:
        fail(str(exc), "check that PulseAudio/PipeWire is running")
        return None

    try:
        from stt import MicConfig

        mic = MicConfig.resolve(device)
        ok(mic.describe())
        print(f"        pin it with: export VOICE_INPUT_DEVICE="
              f"'{mic.name}'   (names survive reboots, indices do not)")
        if mic.rate != 16_000:
            print("        (16 kHz is not supported by this device, so audio "
                  "is captured at its native rate and resampled)")
        return mic
    except RuntimeError as exc:
        fail("no usable microphone", str(exc))
        return None


def check_record(mic, seconds: float = 3.0):
    step(3, f"microphone ({seconds:.0f} seconds)")
    if mic is None:
        fail("skipped: no microphone from step 2")
        return None
    try:
        import numpy as np
        import sounddevice as sd

        from stt import MIN_RMS, SAMPLE_RATE, resample_to

        input(f"    [ENTER], then speak for {seconds:.0f} seconds...")
        raw = sd.rec(int(seconds * mic.rate), samplerate=mic.rate,
                     channels=1, dtype="float32", device=mic.device)
        sd.wait()
        audio = resample_to(raw.reshape(-1), mic.rate)
        rms = float(np.sqrt(np.mean(np.square(audio))))
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        print(f"    captured {audio.size / SAMPLE_RATE:.1f}s at "
              f"{mic.rate} Hz  RMS {rms:.4f}  peak {peak:.3f}")
        if rms < MIN_RMS:
            fail("signal too weak: the microphone is muted or it is the wrong "
                 "one",
                 "raise the input volume, or pick another device with "
                 "--device <index> (and --input-device for run_voice.py)")
            return None
        if peak > 0.99:
            fail("signal is clipping", "lower the microphone gain")
            return audio
        ok("microphone works")
        return audio
    except Exception as exc:
        fail(str(exc))
        return None


def check_transcribe(audio, model_size: str, language: str) -> bool:
    step(4, f"transcription (whisper '{model_size}', language '{language}')")
    if audio is None:
        fail("skipped: no audio from step 3")
        return False
    try:
        from stt import Transcriber

        print("    first run downloads ~75 MB, then it is cached...")
        text = Transcriber(model_size=model_size,
                           language=language).transcribe(audio)
        if not text:
            fail("nothing recognised",
                 "speak closer to the microphone and try again")
            return False
        ok(f'recognised: "{text}"')
        return True
    except Exception as exc:
        fail(str(exc))
        return False


def check_speak(voice: str) -> bool:
    step(5, f"speech synthesis ({voice})")
    try:
        from tts import EdgeSpeaker

        speaker = EdgeSpeaker(voice=voice)
        speaker.say(TEST_PHRASE)
        speaker.close()
        ok("if you heard the phrase, audio output works")
        return True
    except Exception as exc:
        fail(str(exc),
             "edge-tts needs internet; without it run with --tts none")
        return False


def main() -> int:
    from tts import DEFAULT_VOICE

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default=None,
                    help="microphone index or name substring to force; "
                         "defaults to $VOICE_INPUT_DEVICE")
    ap.add_argument("--voice", default=DEFAULT_VOICE, help="Edge voice id")
    ap.add_argument("--stt-model", default="base", help="faster-whisper size")
    ap.add_argument("--language", default="en", help="spoken language code")
    args = ap.parse_args()

    device = args.device
    if device is not None and device.isdigit():
        device = int(device)

    print("voice stack self-test")
    if not check_imports():
        print("\nInstall the missing packages and try again:")
        print("  pip install -r voice/requirements.txt")
        return 1

    mic = check_devices(device)
    audio = check_record(mic)
    check_transcribe(audio, args.stt_model, args.language)
    check_speak(args.voice)

    print("\nIf every step is OK:  python voice/run_voice.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
