"""Conversation compaction — a checkpoint instead of a truncated history.

The model for this is opencode's session compaction (`packages/core/src/session/compaction.ts`),
adapted to this suite's shape: sessions here talk to engines over HTTP, so the transcript *is*
the state.  A compaction rewrites that transcript at a boundary the user (or the window) chose:
the older exchanges become one summary, the newest `keep` tokens stay verbatim, and the pair is
written as a single checkpoint block:

    <conversation-checkpoint>
      <summary>…</summary>
      <recent-context>…</recent-context>
    </conversation-checkpoint>

Three triggers, as upstream: ``auto`` (the estimate crossed the model's window minus a buffer),
``overflow`` (the provider rejected the request as too long; the next attempt aims smaller), and
``manual`` (the user asked).  The summary template is enforced — a reply that ignores it gets one
nudge and then counts as a failure, so a checkpoint is never a half-written block.

Nothing here loads a model: the summarizer is an injected stream function, the token counter is
injected, and events go to an injected sink.  The surgery hook in :mod:`harness.compaction.surgery`
subscribes to the same event, which is where an engine switch is offered.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence

#: Rough characters per token when no counter is injected (this suite's texts, English + code).
CHARS_PER_TOKEN = 4

#: Fills the summary; a reply must contain at least one of these headings to count.
SUMMARY_HEADINGS = ("## Goal", "## Facts", "## Decided", "## Open", "## State")

SUMMARY_TEMPLATE = (
    "Summarize the conversation above for your own future reference.  Keep it short and concrete.\n"
    "Use exactly these headings:\n"
    + "\n".join(SUMMARY_HEADINGS)
    + "\n\nDo not add anything after the last heading's section."
)

#: Sent when a reply ignored the template; one retry, then the compaction fails.
NUDGE = "Your reply did not use the required headings. Repeat the summary with the headings exactly as given."

CHECKPOINT_RE = re.compile(
    r"<conversation-checkpoint>\s*<summary>(?P<summary>.*?)</summary>\s*"
    r"<recent-context>(?P<recent>.*?)</recent-context>\s*</conversation-checkpoint>",
    re.S,
)


@dataclass(frozen=True)
class Message:
    """One turn as the transcript holds it; `role` is "user" | "assistant" | "system"."""

    role: str
    text: str


@dataclass
class Settings:
    """Compaction settings, transformable by plugins (see :class:`Editor`)."""

    auto: bool = True
    #: Tokens kept free below the model's window before auto compaction fires.  None keeps 10%.
    buffer: Optional[int] = None
    #: Tokens of the newest conversation kept verbatim beside the summary.
    keep: int = 15_000


@dataclass
class Editor:
    """What a settings plugin may change (opencode's ``editor.configure``)."""

    settings: Settings

    def configure(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if not hasattr(self.settings, key):
                raise KeyError(f"unknown compaction setting: {key}")
            setattr(self.settings, key, value)


@dataclass(frozen=True)
class Trigger:
    """Why compaction is running: "auto", "overflow" (aim at `share` of the rejected size) or "manual"."""

    reason: str
    share: float = 1.0

    @staticmethod
    def auto() -> "Trigger":
        return Trigger("auto")

    @staticmethod
    def manual() -> "Trigger":
        return Trigger("manual")

    @staticmethod
    def overflow(share: float = 1.0) -> "Trigger":
        return Trigger("overflow", share)


@dataclass
class Checkpoint:
    """A summary plus the newest exchanges, kept as one block of text."""

    summary: str
    recent: str

    def render(self) -> str:
        return (f"<conversation-checkpoint>\n<summary>{self.summary.strip()}</summary>\n"
                f"<recent-context>{self.recent.strip()}</recent-context>\n</conversation-checkpoint>")

    @staticmethod
    def parse(text: str) -> Optional["Checkpoint"]:
        match = CHECKPOINT_RE.search(text)
        return Checkpoint(match.group("summary"), match.group("recent")) if match else None


@dataclass
class Outcome:
    """What a compaction did: "skipped" (not due / auto off), "completed" (with the checkpoint), "failed"."""

    status: str
    checkpoint: Optional[Checkpoint] = None
    error: str = ""


def default_counter(text: str) -> int:
    return max(1, (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN)


def estimate(messages: Iterable[Message], counter: Callable[[str], int] = default_counter) -> int:
    return sum(counter(m.text) for m in messages)


def ceiling(limit: int, buffer: Optional[int] = None) -> int:
    """The token count that triggers auto compaction: the window less `buffer` (or 10%)."""
    return max(1, limit - (buffer if buffer is not None else limit // 10))


def due(messages: Sequence[Message], limit: int, settings: Settings,
        counter: Callable[[str], int] = default_counter) -> bool:
    return settings.auto and estimate(messages, counter) >= ceiling(limit, settings.buffer)


def split(messages: Sequence[Message], keep: int, counter: Callable[[str], int] = default_counter
          ) -> Optional[tuple[list[Message], list[Message]]]:
    """(older, recent): the newest messages that fit `keep` tokens, and everything before them.

    None when there is nothing to compact: fewer than two messages, or every message fits `keep`.
    """
    if len(messages) < 2:
        return None
    kept: list[Message] = []
    used = 0
    for message in reversed(messages):
        size = counter(message.text)
        if kept and used + size > keep:
            break
        kept.append(message)
        used += size
    recent = list(reversed(kept))
    older = list(messages[: len(messages) - len(recent)])
    return (older, recent) if older else None


def build_prompt(update: bool) -> str:
    """The summarization prompt; `update` asks a re-summary to fold in the previous checkpoint."""
    if update:
        return ("Update the conversation checkpoint: fold the newer exchanges into the previous summary "
                "(do not repeat it verbatim) and keep the same headings.\n\n" + SUMMARY_TEMPLATE)
    return SUMMARY_TEMPLATE


def uses_template(text: str) -> bool:
    return any(line.strip().startswith(h) for line in text.splitlines() for h in SUMMARY_HEADINGS)


@dataclass
class SummaryRequest:
    """What the injected stream function sees; `retry` marks the nudge attempt."""

    messages: list[Message]
    prompt: str
    retry: bool = False


#: The summarizer: takes a request, returns the reply text (streamed or not — the caller's choice).
Summarizer = Callable[[SummaryRequest], str]

#: Event sink: (kind, payload) — "compaction.started", ".completed", ".failed", ".skipped".
EventSink = Callable[[str, dict], None]


def _noop(_kind: str, _payload: dict) -> None:
    return None


def compact(messages: Sequence[Message], trigger: Trigger, settings: Settings, summarize: Summarizer,
            counter: Callable[[str], int] = default_counter, emit: EventSink = _noop,
            limit: Optional[int] = None) -> Outcome:
    """Run one compaction: split, summarize (one nudge), and return the checkpoint.

    `limit` is the model's window when the caller knows it; an "auto" trigger with a known limit is
    re-checked against `due` here, so a caller that fires early gets a clean "skipped".  The caller
    owns the transcript: on "completed" it replaces the covered messages with the checkpoint (that
    is the surgery boundary; see :mod:`harness.compaction.surgery`).  Events let observers see the
    checkpoint before it is applied.
    """
    if trigger.reason != "manual" and not settings.auto:
        return Outcome("skipped", error="automatic compaction is off")
    if trigger.reason == "auto" and limit is not None and not due(messages, limit, settings, counter):
        return Outcome("skipped", error="the context fits")

    parts = split(messages, settings.keep, counter)
    if parts is None:
        if trigger.reason == "auto":
            return Outcome("skipped", error="nothing to compact yet")
        return Outcome("failed", error="nothing to compact yet")
    older, recent = parts

    previous = Checkpoint.parse("\n".join(m.text for m in messages))
    prompt = build_prompt(previous is not None)
    base = list(older)
    emit("compaction.started", {"reason": trigger.reason, "older": len(older), "recent": len(recent),
                                "previous": previous is not None})
    request = SummaryRequest(messages=base, prompt=prompt)
    reply = summarize(request)
    if not uses_template(reply):
        request = SummaryRequest(messages=base, prompt=prompt, retry=True)
        reply = summarize(request)
        if not uses_template(reply):
            emit("compaction.failed", {"reason": trigger.reason, "error": "template"})
            return Outcome("failed", error="compaction summary did not match the required headings")

    checkpoint = Checkpoint(summary=reply.strip(),
                            recent="\n\n".join(f"{m.role}: {m.text}" for m in recent))
    emit("compaction.completed", {"reason": trigger.reason, "checkpoint": checkpoint})
    return Outcome("completed", checkpoint=checkpoint)
