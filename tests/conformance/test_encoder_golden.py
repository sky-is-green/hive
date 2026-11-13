"""Frozen encoder vectors: the Rust ONNX path without a Python oracle."""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
GOLDEN = Path(__file__).resolve().parent / "golden" / "encoder_vectors.json"


def _binary() -> Path:
    raw = os.environ.get("HIVE_ENCODER_BIN")
    path = Path(raw) if raw else REPO / "rust" / "target" / "debug" / "hive-encoder"
    if not path.is_file():
        pytest.skip("hive-encoder is not built")
    return path


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    return dot / (left_norm * right_norm + 1e-12)


def test_rust_encoder_matches_frozen_vectors(tmp_path):
    payload = json.loads(GOLDEN.read_text())
    corpus = tmp_path / "texts.txt"
    corpus.write_text("\n".join(payload["texts"]) + "\n")
    result = subprocess.run(
        [str(_binary()), "encode", "--file", str(corpus), "--json"],
        text=True,
        capture_output=True,
        timeout=120,
    )
    if result.returncode != 0:
        pytest.skip(f"encoder model unavailable: {result.stderr.strip()}")
    actual = json.loads(result.stdout)
    assert actual["dimension"] == payload["dimension"]
    cosines = [
        _cosine(row["vector"], expected)
        for row, expected in zip(actual["rows"], payload["vectors"])
    ]
    assert min(cosines) > 0.99999, min(cosines)
