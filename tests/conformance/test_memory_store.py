"""Rust retention-store parity with the Python reference.

The first native memory slice covers chunk storage, fingerprint dedup,
eviction, turn indexing, decay, and cosine retrieval. Full hygiene/codec,
comb, drift, dedup and assembly remain Python.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SPLINTER = REPO / "splinter" / "splinter"
DEFAULT_MEMORY = REPO / "rust" / "target" / "debug" / "hive-memory"


def _binary(name: str) -> Path:
    raw = os.environ.get(f"HIVE_{name.upper()}_BIN")
    path = Path(raw) if raw else REPO / "rust" / "target" / "debug" / f"hive-{name}"
    if not path.is_file():
        pytest.skip(f"hive-{name} is not built")
    return path


def _normalize_store(payload: dict) -> dict:
    for chunk in payload.get("chunks", []):
        chunk["timestamp"] = ""
    return payload


def test_store_add_dedup_and_eviction_match_python(tmp_path):
    if str(SPLINTER) not in sys.path:
        sys.path.insert(0, str(SPLINTER))
    from retention.store import ContextStore

    entries = [
        (1, "alpha chunk about retention"),
        (2, "beta chunk about sandboxing"),
        (3, "alpha chunk about retention"),
        (4, "gamma chunk about routing"),
        (5, "delta chunk about decoding"),
    ]
    python_store = ContextStore(
        max_chunks=3, sanitize=False, ingest_block_prefixes=[]
    )
    for turn, content in entries:
        python_store.add_chunk(turn, content)

    operations = [{"op": "sanitize", "n": 0}, {"op": "set_max", "n": 3}]
    operations += [
        {"op": "add", "turn": turn, "content": content}
        for turn, content in entries
    ]
    ops = tmp_path / "ops.json"
    ops.write_text(json.dumps(operations))
    result = subprocess.run(
        [str(_binary("memory")), "replay", "--file", str(ops), "--json"],
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    rust_store = json.loads(result.stdout)
    assert _normalize_store(rust_store) == _normalize_store(python_store.to_dict())


def test_decay_retrieval_matches_python(tmp_path):
    if str(SPLINTER) not in sys.path:
        sys.path.insert(0, str(SPLINTER))
    from retention.decay import DecayMatrix
    from retention.store import ContextStore
    from sieve.ultra_small import UltraSmallDrone

    chunks = [
        "the retention store keeps chunks with decay metadata",
        "def parse_stack(data): return data.get('tiers', [])",
        "cosine similarity compares normalized vectors",
        "the sandbox confines subprocesses with Landlock",
        "a stale chunk moves into the comb archive",
    ]
    turns = [1, 2, 3, 4, 8]
    query = "how does the retention store decay chunks?"
    current_turn = 12

    python_store = ContextStore(sanitize=False, ingest_block_prefixes=[])
    for turn, content in zip(turns, chunks):
        python_store.add_chunk(turn, content)
    drone = UltraSmallDrone(confidence_mode="off", vocab_boost=0.0)
    scores = drone.score(query, chunks)
    raw = {chunk.id: score.relevance_score for chunk, score in zip(python_store.chunks.values(), scores)}
    effective = DecayMatrix().apply(
        list(python_store.chunks.values()), current_turn, raw
    )
    expected = sorted(effective.items(), key=lambda item: item[1], reverse=True)

    rows = [
        {
            "id": chunk.id,
            "content": chunk.content,
            "turn": chunk.turn,
            "decay_multiplier": 1.0,
            "last_referenced_turn": chunk.turn,
        }
        for chunk in python_store.chunks.values()
    ]
    chunk_file = tmp_path / "chunks.json"
    chunk_file.write_text(json.dumps(rows))
    result = subprocess.run(
        [
            str(_binary("memory")),
            "retrieve",
            "--chunks",
            str(chunk_file),
            "--query",
            query,
            "--top-k",
            str(len(rows)),
            "--current-turn",
            str(current_turn),
            "--json",
        ],
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    rust = json.loads(result.stdout)["results"]
    assert [row["id"] for row in rust] == [chunk_id for chunk_id, _ in expected]
    for row, (_, score) in zip(rust, expected):
        assert math.isclose(row["effective"], score, rel_tol=1e-5, abs_tol=1e-6)
