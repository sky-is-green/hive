"""The escalation lane: a durable harvest of everything the ladder resolved.

Every task whose served answer did not come from a detector-silent local
draft is appended here at run time — the task (checker included: this file
is training-side and nothing in the serving loop reads it), the full draft
that was judged, the structured detector findings, the repair attempt, and
the ladder's resolution (human answer, API answer, or a persistent flag).
One JSON line per item, written next to ``report.json`` as
``escalation.jsonl``.

This is the training feed the cascade grows on its own failures with.  A
discriminating head needs labels for the *silent* distribution — the wrongs
the detectors cannot see, which by construction never escalate.  Escalation
gold seeds that: sample k fresh local drafts on an escalated task (a mined
hard task), checker-label each against the escalated answer, and the head
sees the deployed distribution's fragile corrects and silent wrongs without
a human in the loop.  The runtime stays gold-free by construction — the
serving loop appends and forgets; only offline training/auditing reads.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional


def in_lane(row: dict[str, Any]) -> bool:
    """Did the ladder resolve this item's *local* failure?

    A router-gated row (the router sent it to the API before any local
    generation) is a skip, not a failure - nothing there for the harvest,
    so it is excluded even though ``escalated`` is True.
    """
    if row.get("scion_ok") is None:
        return False
    return bool(
        row.get("flagged")
        or row.get("escalated")
        or row.get("human_gate") is not None
    )


def lane_record(
    row: dict[str, Any],
    task: dict[str, Any],
    *,
    run: str,
    draft: str,
    cascade_answer: str,
    findings: Optional[list[dict[str, Any]]],
) -> dict[str, Any]:
    """The full training-side record of one ladder-resolved item.

    ``row`` is the finished report row; ``task`` the full task dict —
    checker included, because the lane is the one place the answer key is
    stored on purpose (offline sampling re-checks fresh drafts against it);
    ``draft``/``cascade_answer`` are the untruncated texts (the report
    truncates at 4000 chars; training wants the whole thing).
    """
    human = row.get("human_gate")
    if row.get("escalated"):
        source = "api"
    elif human is not None and human.get("accepted_locally"):
        source = "human"
    elif (row.get("repair") or {}).get("cleared"):
        source = "repair"
    else:
        source = "draft"
    return {
        "ts": time.strftime("%Y%m%dT%H%M%S"),
        "run": run,
        "item_id": row["id"],
        "bucket": row.get("bucket"),
        "path": row.get("planned_path"),
        "task": task,
        "draft": draft,
        "findings": list(findings or []),
        "verdict": {
            "decision": row.get("verdict"),
            "confidence": row.get("verdict_confidence"),
            "reason": row.get("verdict_reason"),
            "truncated": row.get("truncated"),
        },
        "repair": row.get("repair"),
        "resolution": {
            "source": source,
            "answer": cascade_answer,
            "human": human,
            "api_answer": row.get("api_answer"),
        },
        "checker": {
            "cascade_ok": row.get("cascade_ok"),
            "false_accept": row.get("false_accept"),
            "false_reject": row.get("false_reject"),
            "human_corrected": row.get("human_corrected"),
        },
    }


class EscalationLane:
    """Append-only JSONL of ladder-resolved items, next to the run's report."""

    def __init__(self, path: Optional[Path]):
        self.path = None if path is None else Path(path)
        self._fh = None
        self.count = 0
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("a", encoding="utf-8")

    @classmethod
    def disabled(cls) -> "EscalationLane":
        """A lane that drops everything (explicit opt-out; tests)."""
        return cls(None)

    def append(self, record: dict[str, Any]) -> None:
        """One line per item; flushed immediately so a killed run keeps its data."""
        if self._fh is None:
            return
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()
        self.count += 1

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "EscalationLane":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
