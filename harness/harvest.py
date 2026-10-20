"""The harvest: the harness's own turns, with the curation that was used.

The encoder the harness runs on is chosen by measurement
(``experiments/encoder_suite.py``) and then tuned on the harness's own
traffic.  For that there has to be a record of what the traffic looked
like: the query, the reply, the chunks the curation selected, and the
settings it ran with.  The store keeps the replies but not the queries,
and it keeps no decision log - so the Studio writes one, here, when the
user asks it to.

The file is ``harness_state/harvest.jsonl`` - one JSON object per turn,
appended as the turns happen.  It is off by default (``harvest_enabled``):
the file holds the user's own words.

What is recorded:

``query``        what the user asked
``reply``        what the model answered (empty if the turn failed)
``selected``     the ids of the chunks the curation put in the context
``candidates``   the ids the curation scored, and their raw scores
``curated``      the assembled context the model was given
``config``       the curation baseline the turn ran with
``encoder``      the model that embedded the chunks
``ts``           when the turn happened
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional


def harvest_path(state_dir: str | Path | None = None) -> Path:
    """Where the turns are written."""
    base = Path(state_dir) if state_dir else Path(__file__).resolve().parents[1] / "harness_state"
    return base / "harvest.jsonl"


def append_turn(state_dir: str | Path | None, record: dict[str, Any]) -> Optional[Path]:
    """Append one turn to the harvest.  Never raises: a failed write is
    not worth losing the turn over."""
    path = harvest_path(state_dir)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        return path
    except OSError:
        return None


def read_turns(state_dir: str | Path | None = None,
               limit: int = 0) -> list[dict]:
    """The harvested turns (newest last)."""
    path = harvest_path(state_dir)
    if not path.exists():
        return []
    out: list[dict] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        return []
    return out[-limit:] if limit else out


def count_turns(state_dir: str | Path | None = None) -> int:
    """How many turns have been harvested."""
    path = harvest_path(state_dir)
    if not path.exists():
        return 0
    try:
        with path.open("r", encoding="utf-8") as fh:
            return sum(1 for line in fh if line.strip())
    except OSError:
        return 0


def clear(state_dir: str | Path | None = None) -> bool:
    """Throw the harvest away (the user's call, not ours)."""
    path = harvest_path(state_dir)
    try:
        if path.exists():
            path.unlink()
        return True
    except OSError:
        return False


def build_record(*, query: str, reply: str, conversation_id: str,
                 turn: int, selected: list[str], curated: str,
                 candidates: dict[str, float], config: dict,
                 encoder: str, token_count: int = 0,
                 budget: int = 0, mode: str = "") -> dict[str, Any]:
    """One turn, in the shape the dataset builder wants."""
    return {
        "ts": time.time(),
        "conversation_id": conversation_id,
        "turn": int(turn),
        "query": query,
        "reply": reply or "",
        "selected": list(selected or []),
        "candidates": {k: round(float(v), 4) for k, v in (candidates or {}).items()},
        "curated": curated or "",
        "config": config or {},
        "encoder": encoder or "",
        "token_count": int(token_count or 0),
        "budget": int(budget or 0),
        "mode": mode or "",
    }
