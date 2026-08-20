"""Spoken progress: keep the operator company while the model works.

A turn that calls three tools takes ten to twenty seconds, and until now all of
it was silent -- the reply is only spoken once the agent has finished. In a
room, silence from a robot is indistinguishable from a crash, so people repeat
themselves or reach for the stop button.

This module speaks *while* the turn is running. Two sources feed it, in order
of preference:

* **The model's own words.** Claude usually writes "Let me check where the arm
  is" before it calls a tool. That sentence is already the perfect progress
  message, so it is spoken rather than invented.
* **The tool being called.** When the model goes straight to a tool with no
  commentary, :func:`phrase_for_tool` supplies a short line for it.

and a heartbeat covers the rest: if a single tool call blocks for longer than
``heartbeat`` seconds -- a long move, a slow trajectory -- a filler is spoken so
the silence never grows unbounded.

Two rules keep this from becoming noise of its own:

* **One voice at a time.** Every spoken line in the program goes through this
  object's single worker thread, including the final answer. ``pygame.mixer``
  has one music channel, so two threads calling ``say`` would cut each other
  off mid-word.
* **Commentary is droppable.** Only one interim line is ever queued; a newer
  one replaces it, and the final answer discards it entirely. Speech is slower
  than tool calls, so a FIFO queue would fall behind and narrate a move that
  finished ten seconds ago.
"""
from __future__ import annotations

import re
import sys
import threading
import time

# How long the operator may hear nothing before a filler is spoken. Roughly
# the length of a slow ``move_linear``: short enough that the pause never feels
# like a crash, long enough that a normal two-tool turn never triggers it.
HEARTBEAT_SECONDS = 9.0

# Cycled rather than repeated: hearing the same four words three times in a row
# sounds more broken than saying nothing at all.
FILLERS = (
    "One moment.",
    "Still working on it.",
    "Nearly there.",
)

# Commentary is a signpost, not the answer. Anything longer is the model
# ignoring its prompt, and reading a paragraph aloud would take longer than the
# work it is describing. The lower bound skips bare acknowledgements.
MIN_NARRATION_CHARS = 20
MAX_NARRATION_CHARS = 160

# Which interim line survives when two are waiting. The model's own sentence
# beats a canned tool phrase, and both beat a heartbeat filler.
RANK_FILLER = 0
RANK_CANNED = 1
RANK_MODEL = 2


def first_sentence(text: str) -> str:
    """The opening of ``text``, cut at a sentence end and capped in length.

    Not literally the first sentence: models open with "Sure!" or "Okay." and
    announcing *that* tells the operator nothing. Sentences are taken until the
    line carries some actual content.
    """
    text = " ".join(text.split())
    out = ""
    for part in re.split(r"(?<=[.!?])\s+", text):
        out = f"{out} {part}".strip()
        if len(out) >= MIN_NARRATION_CHARS:
            break
    if len(out) > MAX_NARRATION_CHARS:
        out = out[:MAX_NARRATION_CHARS].rsplit(" ", 1)[0] + "..."
    return out.strip()


# Tools worth a line of their own. Everything else falls through to the prefix
# rules in phrase_for_tool(), which cover the read/write/move families.
TOOL_PHRASES = {
    "get_robot_state": "Checking where the arm is.",
    "get_motion_status": "Checking the motion.",
    "is_within_safety_limits": "Checking that this is safe.",
    "move_linear": "Moving the arm.",
    "movel": "Moving the arm.",
    "move_robot_to_position": "Moving the arm.",
    "move_joints_relative": "Turning the joints.",
    "movej": "Turning the joints.",
    "move_joint": "Turning the joints.",
    "move_to_stored_tcp_waypoint": "Going to the saved position.",
    "move_to_stored_joint_configuration": "Going to the saved position.",
    "store_waypoint_pose_on_ur": "Saving this position.",
    "store_joint_configuration_on_ur": "Saving this position.",
    "run_trajectory": "Running the trajectory.",
    "start_trajectory_job": "Starting the trajectory.",
    "get_trajectory_job_status": "Checking the trajectory.",
    "stir": "Starting to stir.",
    "start_motion_pattern": "Starting the motion.",
    "adjust_pattern_speed": "Changing the speed.",
    "pause_motion": "Stopping.",
    "resume_motion": "Resuming.",
    "finish_motion": "Finishing the motion.",
    "start_freedrive_mode": "You can move the arm by hand now.",
    "freedrive_mode": "You can move the arm by hand now.",
    "stop_freedrive_mode": "Freedrive is off.",
}


def phrase_for_tool(name: str, arguments: dict | None = None) -> str:
    """A short spoken line for one tool call.

    Args:
        name: The tool name. Accepts the SDK's ``mcp__ur__move_linear`` form as
            well as the bare ``move_linear``.
        arguments: The call's arguments, used only where they change the verb
            -- opening and closing the gripper are the same tool, told apart
            by which tool output is being driven.
    """
    bare = name.rsplit("__", 1)[-1]
    if bare == "set_tool_digital_out":
        # The gripper IS the tool IO: pin 0 opens (True) or closes (False)
        # the jaws, pin 1 picks slow (True) or fast (False).
        args = arguments or {}
        pin, value = args.get("n"), bool(args.get("b"))
        if pin == 0:
            return "Opening the gripper." if value else "Closing the gripper."
        if pin == 1:
            return ("Setting the gripper to slow." if value
                    else "Setting the gripper to fast.")
        return "Setting the tool output."
    if bare in TOOL_PHRASES:
        return TOOL_PHRASES[bare]
    if bare.startswith(("get_", "is_", "check_")):
        return "Checking."
    if bare.startswith(("move", "go_")):
        return "Moving the arm."
    if bare.startswith(("set_", "store_", "save_")):
        return "Setting that up."
    return "Working on it."


class NullNarrator:
    """Does nothing at all.

    The default an agent gets when it is driven from something with no speaker
    -- a test, or the Case 3 agent -- so the agents can report progress
    unconditionally instead of guarding every call with ``if progress:``.
    """

    enabled = False

    def turn_started(self) -> None: ...
    def note_text(self, text: str) -> None: ...
    def note_tool(self, name: str, arguments: dict | None = None) -> None: ...
    def flush_pending(self) -> None: ...
    def finish(self, text: str) -> None: ...
    def cancel(self) -> None: ...
    def close(self) -> None: ...


class Narrator:
    """Serialises every spoken line, and fills the silence during a turn.

    Owns the speaker for the whole program: nothing else may call
    ``speaker.say`` once a Narrator exists, or the two threads will talk over
    each other.

    Args:
        speaker: Anything with ``say(text)``, from ``tts.make_speaker``.
        mode: "speak" to voice progress, "print" to show it on stderr only,
            "none" for the old behaviour (the answer, and nothing before it).
        heartbeat: Seconds of silence during a turn before a filler is spoken.
            Zero disables the heartbeat.
    """

    def __init__(self, speaker, mode: str = "speak",
                 heartbeat: float = HEARTBEAT_SECONDS) -> None:
        self.speaker = speaker
        self.mode = mode
        self.enabled = mode in ("speak", "print")
        self.heartbeat = heartbeat if self.enabled else 0.0

        self._cond = threading.Condition()
        self._slot: str | None = None      # interim line, replaceable
        self._slot_rank = 0                # see _offer()
        self._final: str | None = None     # the answer, wins over _slot
        self._pending_text: str | None = None
        self._speaking = False
        self._busy = False
        self._closed = False
        self._last_spoken = ""
        self._last_end = time.monotonic()
        self._filler = 0

        self._thread = threading.Thread(target=self._run, name="narrator",
                                        daemon=True)
        self._thread.start()

    # --- what the agent reports ------------------------------------------- #
    def turn_started(self) -> None:
        """Called once per utterance, before the model is queried."""
        with self._cond:
            self._pending_text = None
            self._busy = True
            self._last_end = time.monotonic()
            self._filler = 0
            self._cond.notify_all()

    def note_text(self, text: str) -> None:
        """Hold a sentence the model just wrote, but do not speak it yet.

        Whether this text is commentary or the answer is only knowable in
        hindsight: if a tool call follows, it was commentary; if the turn ends,
        it *was* the answer and :meth:`finish` will speak it. Speaking it now
        would say every answer twice.
        """
        text = (text or "").strip()
        if not text:
            return
        with self._cond:
            self._pending_text = text

    def note_tool(self, name: str, arguments: dict | None = None) -> None:
        """Called as each tool call starts.

        The tool call is the proof that the held text was commentary, so it is
        spoken here in preference to the canned phrase -- the model's own
        wording is better than anything in :data:`TOOL_PHRASES`.
        """
        if not self.enabled:
            return
        with self._cond:
            held, self._pending_text = self._pending_text, None
        if held:
            self._offer(first_sentence(held), rank=RANK_MODEL)
        else:
            self._offer(phrase_for_tool(name, arguments), rank=RANK_CANNED)

    def flush_pending(self) -> None:
        """Speak the held sentence now, for a turn that stalls after text."""
        if not self.enabled:
            return
        with self._cond:
            held, self._pending_text = self._pending_text, None
        if held:
            self._offer(first_sentence(held), rank=RANK_MODEL)

    # --- the answer -------------------------------------------------------- #
    def finish(self, text: str) -> None:
        """Speak the final reply and block until it has been played.

        Blocking is the same contract ``speaker.say`` had: the microphone must
        not open while the speakers are still talking, or the robot transcribes
        itself. Queued commentary is dropped -- once the answer exists, a
        signpost pointing at it is worse than nothing.
        """
        with self._cond:
            self._pending_text = None
            self._busy = False
            self._slot = None
            self._final = text
            self._cond.notify_all()
        self._wait_idle()

    def cancel(self) -> None:
        """End the turn without speaking anything.

        For the paths that abandon a turn -- Ctrl-C, a hard stop -- where the
        heartbeat would otherwise keep saying "one moment" about work that is
        no longer happening.
        """
        with self._cond:
            self._pending_text = None
            self._slot = None
            self._busy = False
            self._cond.notify_all()

    def close(self) -> None:
        with self._cond:
            self._busy = False
            self._closed = True
            self._cond.notify_all()
        self._thread.join(timeout=5.0)

    # --- the worker -------------------------------------------------------- #
    def _offer(self, line: str, rank: int = RANK_CANNED) -> None:
        """Queue an interim line, replacing the one waiting when it may.

        Replacement is how the commentary keeps up with the robot, but it must
        not lose the good line: the model emits its sentence and several tool
        calls in one message, so a plain overwrite would drop "Let me check the
        pan and then stir" in favour of the canned phrase for the second tool.
        A line only displaces one of the same rank or lower.
        """
        if not line:
            return
        with self._cond:
            if self._closed or (self._slot is not None
                                and rank < self._slot_rank):
                return
            self._slot = line
            self._slot_rank = rank
            self._cond.notify_all()

    def _wait_idle(self) -> None:
        with self._cond:
            while self._final is not None or self._slot is not None \
                    or self._speaking:
                self._cond.wait(0.1)

    def _timeout(self) -> float | None:
        """How long the worker may sleep before the heartbeat is due."""
        if not self._busy or self.heartbeat <= 0:
            return None
        return max(0.1, self.heartbeat - (time.monotonic() - self._last_end))

    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._closed and self._final is None \
                        and self._slot is None:
                    timeout = self._timeout()
                    if timeout is None:
                        self._cond.wait()
                    elif not self._cond.wait(timeout) and self._busy \
                            and time.monotonic() - self._last_end \
                            >= self.heartbeat:
                        self._slot = FILLERS[self._filler % len(FILLERS)]
                        self._slot_rank = RANK_FILLER
                        self._filler += 1
                if self._final is not None:
                    line, self._final, is_final = self._final, None, True
                elif self._slot is not None:
                    line, self._slot, is_final = self._slot, None, False
                else:
                    return                      # closed with nothing to say
                self._speaking = True
            self._say(line, is_final)
            with self._cond:
                self._speaking = False
                self._last_end = time.monotonic()
                self._cond.notify_all()

    def _say(self, line: str, is_final: bool = False) -> None:
        """Speak one line, skipping an immediate repeat of the last one.

        The repeat check matters at the end of a turn: the SDK's final result
        is usually the same sentence as the last text block, and hearing it
        twice sounds like a stutter.

        The answer always goes to the speaker. ``mode`` only decides how the
        *progress* is delivered -- whether replies are spoken at all is
        ``--tts``'s business, and it says so by handing us a NullSpeaker.
        """
        norm = " ".join(line.lower().split()).rstrip(".!?")
        if norm and norm == self._last_spoken:
            return
        self._last_spoken = norm
        try:
            if is_final or self.mode != "print":
                self.speaker.say(line)
            else:
                print(f"  ... {line}", file=sys.stderr, flush=True)
        except Exception as exc:            # a dead speaker must not end the turn
            print(f"  (progress voice failed: {exc})", file=sys.stderr)


def make_narrator(speaker, mode: str = "speak",
                  heartbeat: float = HEARTBEAT_SECONDS):
    """Build a :class:`Narrator`, or the do-nothing one when mode is "none".

    "none" still needs a speaker for the answer, so the Narrator is used with
    progress switched off rather than replaced -- one code path speaks.
    """
    return Narrator(speaker, mode=mode, heartbeat=heartbeat)
