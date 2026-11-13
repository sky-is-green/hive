#!/usr/bin/env python3
"""Compare the Rust ONNX encoder with Python sentence-transformers.

The Rust binary should be the release build for the headline numbers:

    cargo build --release -p hive-encoder
    HIVE_ENCODER_BIN=rust/target/release/hive-encoder \
        .venv/bin/python tests/conformance/bench_encoder.py
"""

from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"
TOPICS = [
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


def corpus(count: int) -> list[str]:
    return [f"{TOPICS[i % len(TOPICS)]} (case {i:04d})" for i in range(count)]


def rust_bench(
    binary: Path, texts: list[str], batch: int, repeat: int = 3
) -> dict:
    path = Path("/tmp/opencode/encoder-bench.txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(texts) + "\n")
    result = subprocess.run(
        [
            str(binary),
            "bench",
            "--file",
            str(path),
            "--repeat",
            str(repeat),
            "--batch",
            str(batch),
            "--json",
        ],
        text=True,
        capture_output=True,
        timeout=300,
    )
    result.check_returncode()
    return json.loads(result.stdout)


def python_bench(texts: list[str], batch: int) -> dict:
    from sentence_transformers import SentenceTransformer

    started = time.perf_counter()
    model = SentenceTransformer(MODEL, device="cpu")
    load = time.perf_counter() - started
    model.encode(texts[:1], batch_size=batch, show_progress_bar=False)
    started = time.perf_counter()
    for _ in range(3):
        vectors = model.encode(
            texts, batch_size=batch, convert_to_numpy=True, show_progress_bar=False
        )
    elapsed = time.perf_counter() - started
    return {
        "load_s": round(load, 3),
        "vectors": len(texts) * 3,
        "seconds": round(elapsed, 3),
        "texts_per_second": round(len(texts) * 3 / elapsed, 1),
        "max_rss_mb": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1
        ),
        "dimension": int(vectors.shape[1]),
    }


def python_single(texts: list[str]) -> dict:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL, device="cpu")
    model.encode([texts[0]], show_progress_bar=False)
    started = time.perf_counter()
    for text in texts:
        model.encode([text], show_progress_bar=False)
    elapsed = time.perf_counter() - started
    return {
        "texts": len(texts),
        "seconds": round(elapsed, 3),
        "texts_per_second": round(len(texts) / elapsed, 1),
    }


def main() -> int:
    binary = Path(
        os.environ.get(
            "HIVE_ENCODER_BIN", REPO / "rust" / "target" / "release" / "hive-encoder"
        )
    )
    if not binary.is_file():
        print(f"missing {binary}; build it with cargo build --release -p hive-encoder")
        return 1
    texts = corpus(1000)
    single = texts[:200]
    print(
        json.dumps(
            {
                "rust_release_batch32": rust_bench(binary, texts, 32),
                "python_batch32": python_bench(texts, 32),
                "rust_release_single": rust_bench(binary, single, 1, 1),
                "python_single": python_single(single),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
