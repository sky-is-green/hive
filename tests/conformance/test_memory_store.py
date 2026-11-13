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


def _fixture_chunks():
    if str(SPLINTER) not in sys.path:
        sys.path.insert(0, str(SPLINTER))
    from retention.store import ContextStore

    contents = [
        "the retention store keeps chunks with decay metadata",
        "the retention store keeps chunks with decay metadata and fingerprints",
        "the sandbox confines subprocesses with Landlock",
        "cosine similarity compares normalized vectors",
        "the router escalates below the confidence threshold",
        "quantization trades precision for memory bandwidth",
    ]
    store = ContextStore(sanitize=False, ingest_block_prefixes=[])
    for turn, content in enumerate(contents, 1):
        store.add_chunk(turn, content)
    return list(store.chunks.values())


def test_deduplication_matches_python(tmp_path):
    from membrane.dedup import ContextDeduplicator
    from sieve.ultra_small import UltraSmallDrone

    chunks = _fixture_chunks()
    drone = UltraSmallDrone(confidence_mode="off", vocab_boost=0.0)
    embeddings = {chunk.id: drone.embed(chunk.content) for chunk in chunks}
    python_survivors, python_refresh = ContextDeduplicator(threshold=0.92).deduplicate(
        chunks, embeddings
    )

    rows = [
        {
            "id": chunk.id,
            "content": chunk.content,
            "turn": chunk.turn,
            "last_referenced_turn": chunk.turn,
        }
        for chunk in chunks
    ]
    chunk_file = tmp_path / "chunks.json"
    chunk_file.write_text(json.dumps(rows))
    embedding_file = tmp_path / "embeddings.json"
    embedding_file.write_text(
        json.dumps(
            [
                {"id": chunk.id, "vector": [float(v) for v in embeddings[chunk.id]]}
                for chunk in chunks
            ]
        )
    )
    result = subprocess.run(
        [
            str(_binary("memory")),
            "dedup",
            "--chunks",
            str(chunk_file),
            "--embeddings",
            str(embedding_file),
            "--json",
        ],
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["survivors"] == [chunk.id for chunk in python_survivors]
    assert payload["refresh"] == {
        key: value for key, value in python_refresh.items()
    }


def test_topic_drift_matches_python(tmp_path):
    from membrane.drift import TopicDriftDetector
    from sieve.ultra_small import UltraSmallDrone

    chunks = _fixture_chunks()
    drone = UltraSmallDrone(confidence_mode="off", vocab_boost=0.0)
    expected = TopicDriftDetector(embed_fn=drone.embed, threshold=0.6).check(
        chunks[-3:], chunks, drone
    )
    rows = [
        {
            "id": chunk.id,
            "content": chunk.content,
            "turn": chunk.turn,
            "last_referenced_turn": chunk.turn,
        }
        for chunk in chunks
    ]
    chunk_file = tmp_path / "chunks.json"
    chunk_file.write_text(json.dumps(rows))
    result = subprocess.run(
        [
            str(_binary("memory")),
            "drift",
            "--chunks",
            str(chunk_file),
            "--recent",
            "3",
            "--json",
        ],
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["should_reset"] == expected.should_reset
    assert math.isclose(payload["drift_score"], expected.drift_score, rel_tol=1e-5)


def test_remembrance_pass_matches_python(tmp_path):
    if str(SPLINTER) not in sys.path:
        sys.path.insert(0, str(SPLINTER))
    from retention.remembrance import RemembrancePass
    from retention.store import ContextStore
    from sieve.ultra_small import UltraSmallDrone
    from sieve.vocabulary import Vocabulary

    contents = [
        "the retention store keeps chunks with decay metadata",
        "quantization trades precision for memory bandwidth",
        "the sandbox confines subprocesses with Landlock",
    ]
    topic = "retention decay metadata"
    store = ContextStore(sanitize=False, ingest_block_prefixes=[])
    for turn, content in enumerate(contents, 1):
        store.add_chunk(turn, content)
    candidates = list(store.chunks.values())
    drone = UltraSmallDrone(confidence_mode="off", vocab_boost=0.0)
    vocab = Vocabulary.load("code", "general")
    expected = RemembrancePass(vocab).process(candidates, topic, drone)

    rows = [
        {
            "id": chunk.id,
            "content": chunk.content,
            "turn": chunk.turn,
            "last_referenced_turn": chunk.turn,
        }
        for chunk in candidates
    ]
    chunk_file = tmp_path / "chunks.json"
    chunk_file.write_text(json.dumps(rows))
    result = subprocess.run(
        [
            str(_binary("memory")),
            "remembrance",
            "--chunks",
            str(chunk_file),
            "--topic",
            topic,
            "--json",
        ],
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    actual = json.loads(result.stdout)["results"]
    assert [row["chunk_id"] for row in actual] == [
        row.chunk_id for row in expected
    ]
    for rust, python in zip(actual, expected):
        assert rust["saved"] == python.saved
        assert math.isclose(
            rust["relevance_score"], python.relevance_score, rel_tol=1e-5, abs_tol=1e-6
        )
        assert rust["compressed_content"] == python.compressed_content
        if python.new_decay is None:
            assert rust["new_decay"] is None
        else:
            assert math.isclose(rust["new_decay"], python.new_decay)


def test_context_assembly_matches_python(tmp_path):
    if str(SPLINTER) not in sys.path:
        sys.path.insert(0, str(SPLINTER))
    from cortex.routing import RoutingDecision
    from focal.assembly import ContextAssembler
    from focal.budget import AdaptiveBudget
    from membrane.dedup import ContextDeduplicator
    from membrane.drift import TopicDriftDetector
    from retention.store import ContextStore
    from sieve.ultra_small import UltraSmallDrone
    from sieve.vocabulary import Vocabulary

    chunks = _fixture_chunks()
    query = "how does the retention store decay chunks?"
    current_turn = 12

    class StubRouter:
        def route(self, _query, _history=None):
            return RoutingDecision("ultra_small", 1.0, "fixture")

    drone = UltraSmallDrone(confidence_mode="off", vocab_boost=0.0)
    vocab = Vocabulary.load("code", "general")
    store = ContextStore(
        embed_fn=drone.embed, sanitize=False, ingest_block_prefixes=[]
    )
    for chunk in chunks:
        store.add_chunk(chunk.turn, chunk.content, chunk_id=chunk.id)
        store.chunks[chunk.id].last_referenced_turn = chunk.last_referenced_turn
    expected = ContextAssembler(collect_timings=False).assemble(
        query=query,
        current_turn=current_turn,
        store=store,
        router=StubRouter(),
        ultra_small=drone,
        medium=None,
        escalation=None,
        dedup=ContextDeduplicator(vocab=vocab, threshold=0.92),
        drift_detector=TopicDriftDetector(embed_fn=drone.embed, threshold=0.6),
        budget=AdaptiveBudget(),
        max_context=8192,
        skip_remembrance=True,
    )

    rows = [
        {
            "id": chunk.id,
            "content": chunk.content,
            "turn": chunk.turn,
            "decay_multiplier": chunk.decay_multiplier,
            "last_referenced_turn": chunk.last_referenced_turn,
        }
        for chunk in chunks
    ]
    chunk_file = tmp_path / "chunks.json"
    chunk_file.write_text(json.dumps(rows))
    result = subprocess.run(
        [
            str(_binary("memory")),
            "assemble",
            "--chunks",
            str(chunk_file),
            "--query",
            query,
            "--current-turn",
            str(current_turn),
            "--json",
        ],
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    actual = json.loads(result.stdout)
    assert actual["selected_ids"] == expected.selected_chunk_ids
    assert actual["content"] == expected.content
    assert actual["token_count"] == expected.token_count
    assert actual["budget"] == expected.budget
    assert actual["drift_detected"] == expected.drift_detected
