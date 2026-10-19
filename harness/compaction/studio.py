"""Studio wiring for compaction and the engine-surgery offer.

`harness/compaction/core.py` and `harness/compaction/surgery.py` hold the
logic (split, summarize, checkpoint; probe, gate, commit).  This module
supplies the Studio's side of the contract: the summarizer, the engine
probe, the canary gate, and the switch itself.

The docs (`docs/COMPACTION.md`) are explicit that the *measurement* which
makes the gate meaningful — the mid-conversation hand-off between engines —
is still open.  The gate implemented here is the conservative one: a
candidate must reproduce the current engine's answer on the canary before
the switch is committed, so a local engine swap cannot change what the
session is doing.
"""

from __future__ import annotations

import re
from typing import Callable, Optional

from harness.compaction.core import (
    SUMMARY_HEADINGS,
    Checkpoint,
    Message,
    SummaryRequest,
)
from harness.compaction.surgery import Candidate, GateResult, Probe

#: The final answer of a reply, for the canary comparison.
_FINAL_RE = re.compile(
    r"(?:final answer|answer)\s*[:=]\s*(.+)|\$([\-\d.,]+)|\\boxed\{([^}]*)\}",
    re.IGNORECASE,
)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _final_answer(text: str) -> str:
    """The answer-ish tail of a reply, for comparing two engines."""
    match = _FINAL_RE.search(text or "")
    if match:
        return _norm(next((g for g in match.groups() if g), ""))
    return _norm((text or "")[-200:])


def deterministic_summary(messages: list[Message]) -> str:
    """An extractive summary that always carries the template headings.

    Used when no model is loaded: the compaction must still produce a
    well-formed checkpoint (the suite never accepts a half-written one).
    """
    goal = next((m.text.strip() for m in messages if m.role == "user"), "")
    facts, decided = [], []
    for m in messages:
        text = m.text.strip()
        if not text:
            continue
        if m.role == "assistant":
            facts.append(text.splitlines()[0][:160])
            if re.search(r"\b(decided|we will|let's|done|created|fixed)\b", text, re.I):
                decided.append(text.splitlines()[0][:160])
    lines = [f"## {SUMMARY_HEADINGS[0].lstrip('# ')}", goal[:400] or "(no request captured)"]
    lines += [f"## {SUMMARY_HEADINGS[1].lstrip('# ')}", *(f"- {f}" for f in facts[-6:] or ["(none)"])]
    lines += [f"## {SUMMARY_HEADINGS[2].lstrip('# ')}", *(f"- {d}" for d in decided[-3:] or ["(none recorded)"])]
    lines += [f"## {SUMMARY_HEADINGS[3].lstrip('# ')}", "- (see the recent context below)"]
    lines += [f"## {SUMMARY_HEADINGS[4].lstrip('# ')}", f"- {len(messages)} messages compacted"]
    return "\n".join(lines)


def make_summarizer(call_model: Optional[Callable[[str, list[Message]], str]],
                    ) -> Callable[[SummaryRequest], str]:
    """The summarizer the compaction core calls.

    ``call_model(prompt, messages)`` returns the model's reply, or None when
    no model is reachable — then the deterministic summary is used.
    """

    def summarize(request: SummaryRequest) -> str:
        if call_model is not None:
            try:
                reply = call_model(request.prompt, request.messages)
                if reply and reply.strip():
                    return reply
            except Exception:  # noqa: BLE001 - fall back to the local summary
                pass
        return deterministic_summary(request.messages)

    return summarize


def probe_engine(base_url: str) -> Probe:
    """Ask an engine which model it is (and whether it is reachable)."""
    import requests

    base = (base_url or "").rstrip("/")
    if not base:
        return Probe(False, detail="no base url")
    try:
        resp = requests.get(f"{base}/models", timeout=5)
    except Exception as exc:  # noqa: BLE001
        return Probe(False, detail=str(exc)[:120])
    if not resp.ok:
        return Probe(False, detail=f"HTTP {resp.status_code}")
    try:
        data = resp.json()
    except ValueError:
        return Probe(False, detail="unreadable /models")
    ids = [m.get("id") for m in (data.get("data") or []) if m.get("id")]
    return Probe(bool(ids), model=ids[0] if ids else "", detail=f"{len(ids)} model(s)")


def make_candidate(
    name: str,
    base_url: str,
    ask: Callable[[str], str],
    current_ask: Optional[Callable[[str], str]] = None,
) -> Candidate:
    """One engine that could serve the session.

    The canary is the newest user message; the candidate must reproduce what
    the current engine answers for it, otherwise the switch is declined.
    """

    def probe() -> Probe:
        return probe_engine(base_url)

    def gate(checkpoint: Checkpoint, candidate_name: str) -> GateResult:
        canary = ""
        for line in (checkpoint.recent or "").splitlines():
            if line.lower().startswith("user:"):
                canary = line.split(":", 1)[1].strip()
                break
        if not canary:
            return GateResult(True, 1.0, 1.0, "no canary in the checkpoint")
        try:
            answer = ask(canary)
        except Exception as exc:  # noqa: BLE001
            return GateResult(False, 0.0, 1.0, f"canary failed: {exc}"[:120])
        if current_ask is None:
            return GateResult(True, 1.0, 1.0, "no baseline engine to compare")
        try:
            baseline_answer = current_ask(canary)
        except Exception as exc:  # noqa: BLE001
            return GateResult(False, 0.0, 1.0, f"baseline failed: {exc}"[:120])
        same = _final_answer(answer) == _final_answer(baseline_answer)
        contains = (_final_answer(baseline_answer) in _final_answer(answer)
                    or _final_answer(answer) in _final_answer(baseline_answer))
        score = 1.0 if same else (0.5 if contains and _final_answer(answer) else 0.0)
        return GateResult(score >= 1.0, score, 1.0,
                          "answers agree" if same else "answers differ")

    return Candidate(name=name, probe=probe, gate=gate)
