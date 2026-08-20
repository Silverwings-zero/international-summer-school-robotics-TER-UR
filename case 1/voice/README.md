# Voice layer — talk to the UR10e out loud

An optional speech front-end for the Case 1 MCP server. You hold a
conversation; the model decides which of the server's 29 tools to call; the
robot moves; the reply comes back as speech.

Speech-to-text is [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper)
running locally on the CPU. Text-to-speech is
[`edge-tts`](https://github.com/rany2/edge-tts), Microsoft's neural voices.
Both are free and need no API key.

---

## 1. The server stays independent

**`server.py` is unchanged and imports nothing from this folder.** The voice
layer is an MCP *client*, exactly like Claude Code or the Case 3 agent — the
server cannot tell the difference.

```
            microphone                                     speakers
                |                                              ^
                v                                              |
        +---------------+                              +---------------+
        | stt.py        |   faster-whisper (local)     | tts.py        |
        +---------------+                              +---------------+
                |                                              ^
                v                                              |
        +--------------------------------------------------------------+
        | conversation.py   one persistent chat + the reason/act loop   |
        +--------------------------------------------------------------+
                |                                              ^
                | tool calls                                   | results
                v                                              |
        +--------------------------------------------------------------+
        | bridge.py         MCP stdio client   +   LLM chat client      |
        +--------------------------------------------------------------+
                |                                              ^
                v                                              |
   ==================== process boundary (stdio) ======================
                |                                              |
        +--------------------------------------------------------------+
        | ../server.py      UNCHANGED. Diamond safety layer, 29 tools   |
        +--------------------------------------------------------------+
                                      |
                                 UR10e / simulator
```

Verify the independence at any time — this must keep working with none of the
voice packages installed:

```bash
cd "case 1"
python server.py          # starts, waits on stdio, no audio imports
```

---

## 2. Install

Python 3.10+ (tested on 3.13.5). From the repository root:

```bash
# 1. system audio library — sounddevice binds PortAudio, pip cannot supply it
sudo apt install -y libportaudio2          # Debian/Ubuntu
# macOS:  brew install portaudio
# Windows: nothing to do, the wheel ships it

# 2. python packages
cd "case 1"
pip install -r requirements.txt            # the server itself (fastmcp)
pip install -r voice/requirements.txt      # the voice layer
```

Then choose a brain. There are two, and the default needs no API key.

### Default: your Claude Code subscription (`--backend claude`)

Nothing to configure. `claude-agent-sdk` runs the `claude` CLI you are already
logged into, so tokens come out of your Claude plan rather than a separate API
bill. Confirm the CLI is there:

```bash
claude --version          # e.g. 2.1.235 (Claude Code)
```

`ANTHROPIC_API_KEY` is neither needed nor used. If one is exported it would be
billed separately, so leave it unset.

### Alternative: any OpenAI-compatible endpoint (`--backend openai`)

Same three variables Case 3 uses, so one setup serves both — see
[`../../llm-client/`](../../llm-client/) for free options:

```bash
export AGENT_BASE_URL=https://integrate.api.nvidia.com/v1
export AGENT_API_KEY=nvapi-...
export AGENT_MODEL=z-ai/glm-5.2
```

> Put those three lines in a file you `source`, not in your shell history —
> the key is a secret.

---

## 3. Verify before you need it

Run the self-test **first**. It touches neither the robot nor the LLM, so it
isolates audio problems from everything else:

```bash
cd "case 1"
python voice/check_audio.py
python voice/check_audio.py --device 7     # force one microphone
```

It checks five things in order and prints `OK` or `FAIL` with a fix for each:
packages installed → a device *and a sample rate* that open → microphone
captures sound → Whisper transcribes it → Edge TTS speaks. The first run
downloads the Whisper model (~75 MB), cached afterwards in
`~/.cache/huggingface`.

Step 2 prints the negotiated rate, e.g. `microphone: sof-hda-dsp: - (hw:0,0) @
48000 Hz, resampled to 16000 Hz`. That is normal: see §6.

If a teammate cannot get voice working, ask for this output. It localises the
problem in one paste.

---

## 4. Run

### On the real robot

Run the preflight first — it checks the ports, the dashboard and the RTDE read
path, and never moves the arm:

```bash
cd "case 1"
python3 preflight_real_robot.py 192.168.1.100
python3 voice/run_voice.py --robot 192.168.1.100
```

Export `UR_HOST=192.168.1.100` to make that the default and drop the flag. A
real address selects the **ur5e** safety envelope (±0.95 m workspace) and the
reduced speed caps in `run_server.sh` — never let a real UR5e run under ur10e
geometry, which would approve targets 40 cm past its reach.

### On the simulator

The simulator must be up (see [`../../simulation environment/`](../../simulation%20environment/))
and the robot powered on with brakes released.

```bash
cd "case 1"
python voice/run_voice.py                 # loopback, ur10e
python voice/run_voice.py --robot 127.0.0.1   # force it when UR_HOST is set
```

Useful variants:

```bash
python voice/run_voice.py --text          # type instead of speaking (no mic)
python voice/run_voice.py --tts none      # print replies instead of speaking
python voice/run_voice.py --list-devices  # find your microphone's index
python voice/run_voice.py --quiet         # hide tool calls, cleaner for a demo
python voice/run_voice.py --progress none # no spoken commentary, answer only
python voice/run_voice.py --input-device 7   # force a microphone by index
python voice/run_voice.py --language it --voice it-IT-DiegoNeural  # another language
```

`--text` is the one to reach for when debugging the *robot* side: it removes
audio from the picture entirely while keeping the identical tool loop.

### While talking

| Key | Effect |
|---|---|
| `ENTER` | start recording; `ENTER` again sends it |
| `!` then `ENTER` | **stop the robot now** — calls `pause_motion` directly, no model in the path |
| `q` then `ENTER` | quit (leaves the robot as it is) |
| `Ctrl-C` | stop the robot, then exit |

### While it is working

One request often becomes several tool calls — check the state, move, confirm —
and that takes ten to twenty seconds. The assistant does not wait in silence
for it: it speaks as it goes.

```
you> put the pan on the hob
robot> Let me check where the arm is first.      <- the model's own sentence
robot> Moving the arm.                           <- a line for the tool call
robot> One moment.                               <- the arm is still moving
robot> The pan is on the hob.                    <- the answer
```

Three sources, in that order of preference:

1. **What the model wrote.** Claude usually narrates before it acts. That
   sentence is already the right progress message, so it is spoken rather than
   replaced by something invented.
2. **The tool being called**, when the model goes straight to it without
   commentary — `TOOL_PHRASES` in `progress.py` maps `move_linear` to *"Moving
   the arm."* and so on.
3. **A filler**, if a single call blocks for longer than `--heartbeat` seconds
   (9 by default). This is the one that covers a slow trajectory.

`--progress print` shows the commentary on screen but keeps the voice for the
answer alone; `--progress none` restores the old behaviour. Only one line is
ever queued, and the answer discards it — speech is slower than tool calls, so
a queue would end up narrating a move that finished ten seconds ago.

### Things to try

> "Where are you right now?" · "Go to the home position" · "Stir the pot, it is
> twenty centimetres across" · "Faster" · "Stop" · "Save this position as
> cutting board"

---

## 4b. Using it from Claude Code (no LLM key needed)

If you already drive the robot from Claude Code, you do **not** need
`run_voice.py` — that is a standalone client with its own model. Claude Code
owns its conversation, so speech cannot be injected into its loop. Instead,
`voice_mcp.py` offers the microphone and the speaker to Claude Code as three
more tools, sitting next to the robot's:

| Tool | What it does |
|---|---|
| `listen(max_seconds)` | Records until you stop talking, returns the transcript |
| `speak(text)` | Says one or two sentences out loud, blocking until finished |
| `voice_check()` | Reports the microphone, model and voice in use |

```
Claude Code
   |-- ur-tools   (../server.py)   move, stir, stop, ...
   `-- voice-io   (voice_mcp.py)   listen, speak, voice_check
```

**First, prove the audio works on its own.** Do not run `voice_mcp.py` bare and
wait — an MCP stdio server prints its banner and then sits silently waiting for
a client, which is indistinguishable from a hang. Use the self-test instead:

```bash
cd "case 1"
python voice/voice_mcp.py --selftest    # speaks, then listens for 5 s
```

It reports the microphone, says a line, records you, and prints what it heard.
Only once that passes is it worth wiring into a client.

Register it next to `ur-tools` in your `.mcp.json`:

```json
{
  "mcpServers": {
    "voice-io": {
      "command": "/absolute/path/to/python",
      "args": ["/absolute/path/to/case 1/voice/voice_mcp.py"],
      "env": {
        "VOICE_INPUT_DEVICE": "hw:0,6",
        "VOICE_TTS_VOICE": "en-US-AriaNeural",
        "VOICE_LANGUAGE": "en",
        "VOICE_STT_MODEL": "base"
      }
    }
  }
}
```

> **The `env` block is not optional.** The MCP SDK does not pass your shell
> environment to a server it launches — it forwards only a small safe subset.
> Exporting `VOICE_INPUT_DEVICE` in your terminal has no effect on a server
> started by Claude Code; it must be written here.

Then start a conversation with something like:

> "Use voice_check to confirm the audio, then go into voice mode: listen to me,
> do what I ask with the robot tools, speak the answer in one short sentence,
> and listen again. Keep going until I say we are done."

The first `listen` takes a few seconds longer than the rest — that is Whisper
loading, once per session.

### Which of the two should you use?

`run_voice.py --backend claude` now uses the same subscription as Claude Code,
so the main reason to prefer the MCP route has gone. The standalone app keeps
the hard-stop key, which the MCP route cannot have.

| | `run_voice.py` | `voice_mcp.py` in Claude Code |
|---|---|---|
| Model | Claude subscription, or your own endpoint | Claude Code's |
| Setup | none (`--backend claude`) | `.mcp.json` |
| Push-to-talk | yes, `ENTER` | no — it stops on silence |
| Hard-stop key | yes, `!` | no; say "stop", or Esc in Claude Code |
| Runs unattended | yes | needs you in the Claude Code session |

Use Claude Code for development and demos where you are at the keyboard
anyway. Use `run_voice.py` when the assistant has to stand on its own — and
note it is the only one of the two with a stop key that bypasses the model.

---

## 5. Options

| Flag | Default | What it does |
|---|---|---|
| `--text` | off | Typed input, no microphone |
| `--tts` | `edge` | `none` prints replies instead of speaking |
| `--voice` | `en-US-AriaNeural` | Any id from `edge-tts --list-voices` |
| `--rate` | `+0%` | Speech speed, e.g. `+15%` for a brisker demo |
| `--stt-model` | `base` | `tiny` / `base` / `small` / `medium` |
| `--language` | `en` | Language code, or `auto` to detect |
| `--input-device` | negotiated | Microphone index or name substring |
| `--quiet` | off | Hide tool calls |
| `--robot` | `$UR_HOST`, else sim | REAL arm at this IP; implies the `ur5e` safety envelope |
| `--ur-model` | from `--robot` | Which arm the safety layer guards (`ur10e`/`ur5e`/`ur5`) |
| `--vision` | on | Merged camera+robot server (37 tools); `--no-vision` gives 29 and no camera |
| `--progress` | `speak` | Feedback while the model works: `speak`, `print` (screen only), `none` |
| `--heartbeat` | `9.0` | Seconds of silence in a turn before a filler is spoken; `0` disables |
| `--backend` | `claude` | `claude` = your Claude subscription (no key); `openai` = `AGENT_*` endpoint |
| `--model` | backend default | Model id override |
| `--server` | `../server.py` | Point at a different MCP server |
| `--server-log` | `voice/server.log` | `-` keeps server stderr on screen |

Behaviour constants worth knowing, all near the top of their file:
`MAX_TOOL_STEPS`, `MAX_HISTORY_MESSAGES`, `STOP_TOOL` and the system prompt in
`conversation.py`; `MIN_RMS`, `MIN_DURATION_S` and `RATE_CANDIDATES` in
`stt.py`; `MAX_SPOKEN_CHARS` in `tts.py`; `HEARTBEAT_SECONDS`, `FILLERS` and
`TOOL_PHRASES` in `progress.py`.

The assistant answers in whatever language it is spoken to, so `--language`
only sets what Whisper expects to hear and `--voice` only sets the accent it
replies in.

---

## 6. Why it is built this way

Six decisions that are not obvious, and that you should not undo without
knowing what they cost.

**One long-lived MCP session, not one per utterance.** The server holds live
state: the active pattern inside `PatternRunner`, the trajectory job registry,
the freedrive flag. Reconnecting between utterances would discard it, so "stir
the pan" followed by "faster" would fail with *no pattern is running*. The
session opens once and spans the whole conversation.

**One thread speaks, and it owns the speaker.** `pygame.mixer` has a single
music channel, so a progress line and an answer synthesised on two threads
would cut each other off mid-word. Everything spoken goes through the
`Narrator`'s worker thread, including the final reply — which is also why
`narrator.finish()` blocks until playback really ends: the microphone must not
open while the speakers are still talking, or the robot transcribes itself.

**Push-to-talk on `ENTER`, not the `keyboard` library.** `keyboard` reads
`/dev/input` and therefore needs root on Linux — an unacceptable setup step for
a shared repo. Blocking on `input()` needs no privileges, no X11, and works
over SSH. The recording itself runs on PortAudio's own thread, which is why the
main thread is free to wait for the second `ENTER`.

**Always push-to-talk, never always-listening.** Servo noise and bystander
conversation would feed the model commands nobody gave. Push-to-talk also gives
a natural place for the hard-stop key.

**The emergency stop does not go through the model.** `!` calls `pause_motion`
directly through the MCP session. A round trip to a cloud endpoint is seconds,
and this is the one command that cannot wait. `pause_motion` is the right
choice over `finish_motion`: it keeps the anchor and phase so the motion can be
resumed, and `PatternRunner.pause()` falls through to `_stop_stray()`, so it
stops the arm even if a pattern is still looping from an earlier session.

**The sample rate is negotiated, never assumed.** Whisper wants 16 kHz, but
most ALSA *hardware* devices refuse to open at it — a raw `hw:0,0` capture
device typically supports only 44.1/48 kHz and answers anything else with
`Invalid sample rate [PaErrorCode -9997]`. `MicConfig.resolve()` tries 16 kHz
first, falls back to the device's own preferred rate, and `resample_to()`
converts in numpy afterwards. Downsampling is low-passed with a moving average
first: without it, everything above the new 8 kHz Nyquist aliases straight down
into the speech band and accuracy drops. A box filter is crude next to a
polyphase FIR, but it is inaudible here and saves a ~30 MB scipy dependency.

Resolution happens **once** at startup, not per utterance — probing rates opens
and closes streams, which is slow and makes ALSA write to stderr. If the
default device fails at every rate, the search falls back to ALSA's plug layers
(`sysdefault`, `pulse`, `pipewire`), which resample in the driver and accept
anything. An explicit `--input-device` is never silently redirected: it fails
loudly instead, because someone who named a device wants to know it did not
work.

**Audio never touches the disk.** `sounddevice` produces float32 samples,
`faster-whisper` accepts them directly, and `edge-tts` MP3 goes into a
`BytesIO` that pygame decodes in-process. No temp files, no ffmpeg, no external
player.

### The failure modes it defends against

*Whisper hallucinating on silence.* Whisper invents confident sentences from
near-silence — in English it reliably produces *"Thank you for watching!"*,
which would then be sent to the robot as a command. `is_usable()` rejects clips
below `MIN_RMS` or `MIN_DURATION_S` before decoding, and the VAD filter catches
the rest.

*Losing the voice must not lose the robot.* `edge-tts` needs internet.
Every synthesis failure degrades to printing the line instead of raising, so
flaky conference wifi costs you the voice and nothing else.

*Context growth.* Tool results here are verbose — trajectory logs, joint dumps.
`_trim()` keeps a bounded window and always cuts on a `user` message, never in
the middle of a tool round-trip, which would otherwise leave orphaned
`tool_call_id`s that most endpoints reject outright.

*Server noise over the conversation.* FastMCP prints a full traceback for every
tool call the Diamond layer rejects. That goes to `voice/server.log`.

---

## 7. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `OSError: PortAudio library not found` | `sudo apt install -y libportaudio2` |
| `No model set. Export AGENT_MODEL...` | The three `AGENT_*` variables are not exported in *this* shell |
| `Invalid sample rate [PaErrorCode -9997]` | The device will not open at 16 kHz. Handled automatically now; if it still appears, `--list-devices` then `--input-device 7` (the `sysdefault` entry) |
| Records fine, transcribes nothing | Wrong microphone. `--list-devices`, then `--input-device <n>` |
| Transcribes "Thank you for watching!" | It heard silence. Speak closer, or lower `MIN_RMS` in `stt.py` |
| Wrong language transcribed | `--language en` forces it; `auto` guesses from short clips and is often wrong |
| Whisper too slow | `--stt-model tiny`, or `device="cuda"` in `stt.py` with an NVIDIA GPU |
| No sound out, no error | `--tts none` to confirm the text path works, then check step 5 of `check_audio.py` |
| `voice unavailable: ...` on every reply | No internet, or the Edge endpoint is blocked. Use `--tts none` |
| Robot does not move, replies sound fine | Read `voice/server.log`. Usually the simulator is off, or the brakes are engaged |
| "nessun pattern in esecuzione" after a stir | Something restarted the server. The pattern loop lives in the controller — `!` stops it regardless |
| `python voice_mcp.py` prints "Starting MCP server ... stdio" then nothing | Working as intended: it is waiting for a client on stdin. Run `--selftest` to exercise the audio, or launch it from Claude Code |
| Microphone index changed after a reboot | Indices are not stable — the same laptop showed the mic at 4 with 9 devices and at 4 with 7 devices in another session. Pin the **name** (`VOICE_INPUT_DEVICE='hw:0,6'`), not the number |
| `voice-io` tools missing in Claude Code | Absolute paths in `.mcp.json`, and the interpreter must be the one with the voice packages. `/mcp` shows the connection state |
| `listen` always returns `silence` | Run `voice_check`. Usually `VOICE_INPUT_DEVICE` is missing from the `env` block, so it picked the wrong default |
| `listen` cuts you off mid-sentence | Raise `VAD_SILENCE_S` in `stt.py` (default 1.2 s) |
| Claude Code session hangs after adding voice-io | Something printed to stdout in `voice_mcp.py` — stdout is the protocol. Wrap it in `_stdout_to_stderr()` |
| `voice unavailable: asyncio.run() cannot be called from a running event loop` | Fixed — `tts.py` now detects a running loop and synthesizes on a private thread. If you see it again, you are on an old copy of `tts.py` |
| Model narrates tool names | Tighten the system prompt in `conversation.py`; `clean_for_speech` already strips the punctuation |

---

## 8. Files

| File | Lines | Purpose |
|---|---|---|
| `stt.py` | ~320 | Device/rate negotiation, push-to-talk capture, resampling, Whisper |
| `tts.py` | ~140 | Edge neural voice, MP3 playback, speech text cleanup |
| `bridge.py` | ~185 | MCP stdio client + OpenAI-compatible chat client |
| `conversation.py` | ~170 | Persistent conversation, tool loop, system prompt, e-stop |
| `progress.py` | ~330 | Spoken progress during a turn: commentary, tool phrases, heartbeat |
| `run_voice.py` | ~210 | CLI entry point and the input loop |
| `check_audio.py` | ~200 | Standalone self-test; no robot, no LLM |
| `claude_backend.py` | ~160 | `--backend claude`: the Claude Code subscription as the brain |
| `voice_mcp.py` | ~190 | MCP server exposing `listen`/`speak` to Claude Code |
| `requirements.txt` | — | Voice-only dependencies |

`server.log` is written at run time and is git-ignored.

---

## 9. Where this goes next

The layer is deliberately thin, so the planned modules attach without
restructuring it:

- **VLM / camera.** Add an `analyze_current_view` tool to `server.py` returning
  an image. It arrives here as another entry in the catalogue — but note the
  model behind `AGENT_MODEL` must be multimodal, and `bridge.py` currently
  forwards tool results as text only. That is the one place to extend.
- **Wake word.** Replace the `ENTER` gate in `read_utterance()` with an
  `openWakeWord` detector. Nothing else changes. Keep the hard-stop key.
- **Lower latency.** Speak the first sentence while the rest is still being
  synthesized: split the reply on `.` and stream. Worth roughly a second.
- **Case 3 agent.** `stt.py` and `tts.py` have no dependency on the rest;
  import them from the Case 3 loop if you want voice on the autonomous agent.
