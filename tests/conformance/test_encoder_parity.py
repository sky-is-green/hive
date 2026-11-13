"""Phase 3 encoder parity: Rust ONNX vs Python sentence-transformers.

The corpus is deterministic and intentionally varied (natural language, code,
math, short/long). Python is the oracle here; the runtime path itself does not
need Python.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ENCODER = REPO / "rust" / "target" / "debug" / "hive-encoder"


def _encoder_bin() -> Path:
    raw = os.environ.get("HIVE_ENCODER_BIN")
    binary = Path(raw).expanduser() if raw else DEFAULT_ENCODER
    if not binary.is_file():
        pytest.skip("hive-encoder not built; run `cargo build -p hive-encoder`")
    return binary


def _corpus() -> tuple[list[str], list[str]]:
    topics = [
        "the retention store keeps chunks until the budget evicts them",
        "def parse_stack(data): return data.get('tiers', [])",
        "cosine similarity is dot(a, b) / (norm(a) * norm(b))",
        "a stale chunk is archived into the comb only when it was once curated",
        "the decoder writes JSONL checkpoints with atomic renames",
        "use Landlock and seccomp to confine the sandbox subprocess",
        "the router escalates when confidence is below the threshold",
        "drain the engine slots before terminating the model server",
        "the cache share tells you whether a model is warm in page cache",
        "quantization trades precision for memory bandwidth",
        "the sidecar restarts with capped exponential backoff",
        "an embedding of dimension 384 is normalized before retrieval",
    ]
    sentences = []
    for index in range(1000):
        base = topics[index % len(topics)]
        sentences.append(f"{base} (case {index:04d})")
    queries = [
        f"how does {topics[index % len(topics)].split()[0]} work?" for index in range(20)
    ]
    return sentences, queries


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    return dot / (left_norm * right_norm + 1e-12)


def test_rust_encoder_matches_python_on_1k_texts(tmp_path):
    probe = subprocess.run(
        [str(_encoder_bin()), "info", "--json"],
        text=True,
        capture_output=True,
        timeout=30,
    )
    if probe.returncode != 0:
        pytest.skip(f"encoder model unavailable: {probe.stderr.strip()}")
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as error:  # noqa: BLE001 - optional oracle
        pytest.skip(f"sentence-transformers unavailable: {error}")

    sentences, queries = _corpus()
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join(sentences + queries) + "\n")

    result = subprocess.run(
        [
            str(_encoder_bin()),
            "encode",
            "--file",
            str(corpus),
            "--json",
        ],
        text=True,
        capture_output=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    rust_rows = payload["rows"]
    assert len(rust_rows) == len(sentences) + len(queries)
    assert payload["dimension"] == 384

    model = SentenceTransformer(
        "sentence-transformers/paraphrase-MiniLM-L3-v2", device="cpu"
    )
    python_vectors = model.encode(
        sentences + queries,
        batch_size=32,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    rust_vectors = [row["vector"] for row in rust_rows]
    assert len(rust_vectors[0]) == len(python_vectors[0])

    cosines = [
        _cosine(rust, [float(value) for value in python])
        for rust, python in zip(rust_vectors, python_vectors)
    ]
    assert min(cosines) > 0.99999, min(cosines)
    max_abs = max(
        abs(rust - float(python))
        for rust_vector, python_vector in zip(rust_vectors, python_vectors)
        for rust, python in zip(rust_vector, python_vector)
    )
    assert max_abs < 1e-4, max_abs

    # Retrieval parity: the same top-5 document ids for 20 query vectors.
    def top5(query: list[float], rows: list[list[float]]) -> list[int]:
        scored = [(_cosine(query, row), index) for index, row in enumerate(rows)]
        scored.sort(reverse=True)
        return [index for _, index in scored[:5]]

    python_docs = [[float(value) for value in row] for row in python_vectors[:1000]]
    rust_docs = [list(row["vector"]) for row in rust_rows[:1000]]
    for index in range(20):
        python_query = [float(value) for value in python_vectors[1000 + index]]
        rust_query = list(rust_rows[1000 + index]["vector"])
        assert top5(rust_query, rust_docs) == top5(python_query, python_docs), index
