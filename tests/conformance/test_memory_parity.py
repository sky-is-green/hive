"""Memory-pipeline parity with the native encoder.

The memory code is still Python in this slice, but swapping its embedding drone
from sentence-transformers to `hive-encoder serve` must not change selection or
the assembled context.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SPLINTER = REPO / "splinter" / "splinter"
DEFAULT_ENCODER = REPO / "rust" / "target" / "debug" / "hive-encoder"

CHUNKS = [
    (1, "The retention store keeps chunks with decay metadata and a content fingerprint."),
    (2, "def parse_stack(data): return data.get('tiers', [])"),
    (3, "Cosine similarity is dot(a,b) / (norm(a) * norm(b))."),
    (4, "The sandbox uses Landlock and seccomp to confine subprocesses."),
    (5, "A stale chunk is archived into the comb only when it was once curated."),
    (6, "The router escalates when the similarity confidence is below the threshold."),
    (7, "Drain the engine slots before terminating the model server."),
    (8, "Quantization trades precision for memory bandwidth."),
]
QUERY = "how does the retention store decay chunks?"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _encoder_bin() -> Path:
    raw = os.environ.get("HIVE_ENCODER_BIN")
    binary = Path(raw).expanduser() if raw else DEFAULT_ENCODER
    if not binary.is_file():
        pytest.skip("hive-encoder not built; run `cargo build -p hive-encoder`")
    return binary


def _start_encoder(binary: Path) -> tuple[subprocess.Popen, int]:
    port = _free_port()
    process = subprocess.Popen(
        [str(binary), "serve", "--host", "127.0.0.1", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.time() + 20
    while time.time() < deadline:
        if process.poll() is not None:
            raise AssertionError(process.stderr.read() if process.stderr else "encoder exited")
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=0.5
            ) as response:
                if response.status == 200:
                    return process, port
        except OSError:
            time.sleep(0.05)
    process.kill()
    raise AssertionError("encoder service did not become ready")


def _run(drone):
    from cortex.config import SplinterConfig
    from cortex.splinter import Splinter

    config = SplinterConfig(
        confidence_mode="off",
        comb_enabled=False,
        enable_medium=False,
        gate_enabled=False,
    )
    splinter = Splinter(config=config, ultra=drone, backend=None)
    for turn, text in CHUNKS:
        splinter.store.add_chunk(turn, text)
    result = splinter.process_turn(QUERY, conversation_id="parity", record_exchange=False)
    assembled = result.assembled
    return {
        "mode": result.mode,
        "selected": assembled.selected_chunk_ids if assembled else [],
        "content": assembled.content if assembled else "",
        "tokens": assembled.token_count if assembled else 0,
    }


def test_memory_pipeline_is_identical_with_the_native_encoder():
    if str(SPLINTER) not in sys.path:
        sys.path.insert(0, str(SPLINTER))
    try:
        from sieve.rust_encoder import RustEncoderDrone
        from sieve.ultra_small import UltraSmallDrone
    except Exception as error:  # noqa: BLE001 - optional oracle
        pytest.skip(f"memory modules unavailable: {error}")

    binary = _encoder_bin()
    process, port = _start_encoder(binary)
    try:
        python_result = _run(UltraSmallDrone(confidence_mode="off"))
        rust_result = _run(
            RustEncoderDrone(
                base_url=f"http://127.0.0.1:{port}", confidence_mode="off"
            )
        )
    finally:
        process.terminate()
        process.wait(timeout=10)

    assert rust_result == python_result
