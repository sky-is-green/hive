"""Mechanical engine surgery at a compaction checkpoint.

Compaction is the one moment a session is deliberately re-serialized: the covered history becomes
a checkpoint, and in this suite the transcript *is* the state (sessions talk to stateless engines),
so pointing the next request at another engine costs nothing structurally.  That makes the
checkpoint the natural place to offer what this module calls surgery: change *what serves the
session* — a different stack tier, a different engine build, a llama.cpp-to-Strata hand-off — and
let tests decide whether the change sticks.

The offer is deliberately transactional:

1. a candidate engine is probed (reachable, the model it claims, a health line);
2. the **gate** runs: deterministic checks plus a canary replay of the last exchanges on the
   candidate, scored the way the suite scores everything (fixture facts, not vibes);
3. only a positive gate calls `apply` (the stack layer owns the actual switch); anything else
   leaves the session on the engine it had, with the checkpoint in place as if nothing happened.

Nothing here spawns a server or loads a model: probes, gates and the switch itself are injected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from harness.compaction.core import Checkpoint


@dataclass(frozen=True)
class Probe:
    """What a candidate answered when asked who it is."""

    ok: bool
    model: str = ""
    detail: str = ""


@dataclass(frozen=True)
class GateResult:
    """The tests' verdict on a candidate: did the canary/checks stay whole through the switch?"""

    passed: bool
    score: float = 0.0
    baseline: float = 0.0
    detail: str = ""


@dataclass(frozen=True)
class Candidate:
    """One engine that could serve the session: a stack tier, a port, another build."""

    name: str
    probe: Callable[[], Probe]
    gate: Callable[[Checkpoint, str], GateResult]


@dataclass
class SurgicalOutcome:
    """What the offer did: "committed" (with the chosen engine), "declined", or "unavailable"."""

    status: str
    engine: str = ""
    results: list[tuple[str, Probe, Optional[GateResult]]] = field(default_factory=list)
    reason: str = ""


#: Event sink, the same shape as core's: "surgery.offered", ".probed", ".gate", ".committed", ".declined".
EventSink = Callable[[str, dict], None]


def _noop(_kind: str, _payload: dict) -> None:
    return None


def offer_surgery(session: str, checkpoint: Checkpoint, current: str, candidates: Sequence[Candidate],
                  apply: Callable[[str], None], emit: EventSink = _noop,
                  min_score: Optional[float] = None) -> SurgicalOutcome:
    """Offer the engine switch at a checkpoint; commit only when the gate passes.

    `candidates` are the engines that could serve the session (the current one is skipped); `apply`
    is the stack's switch, called exactly once, only on success.  `min_score`, when given, is an
    absolute floor in addition to the gate's own baseline comparison.
    """
    if not candidates:
        emit("surgery.declined", {"session": session, "reason": "no candidate"})
        return SurgicalOutcome("unavailable", reason="no candidate engines")
    emit("surgery.offered", {"session": session, "current": current,
                             "candidates": [c.name for c in candidates]})
    results: list[tuple[str, Probe, Optional[GateResult]]] = []
    for candidate in candidates:
        if candidate.name == current:
            continue
        probe = candidate.probe()
        emit("surgery.probed", {"session": session, "engine": candidate.name,
                                "ok": probe.ok, "model": probe.model})
        if not probe.ok:
            results.append((candidate.name, probe, None))
            continue
        gate = candidate.gate(checkpoint, candidate.name)
        emit("surgery.gate", {"session": session, "engine": candidate.name, "passed": gate.passed,
                              "score": gate.score, "baseline": gate.baseline, "detail": gate.detail})
        results.append((candidate.name, probe, gate))
        below_floor = min_score is not None and gate.score < min_score
        if gate.passed and not below_floor:
            apply(candidate.name)
            emit("surgery.committed", {"session": session, "engine": candidate.name,
                                       "score": gate.score, "baseline": gate.baseline})
            return SurgicalOutcome("committed", engine=candidate.name, results=results)
    emit("surgery.declined", {"session": session, "reason": "no candidate passed the gate"})
    return SurgicalOutcome("declined", results=results, reason="no candidate passed the gate")
