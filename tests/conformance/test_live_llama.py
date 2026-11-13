"""Opt-in live smoke test: the Rust supervisor against a real `llama-server`.

These tests are skipped unless both paths are provided::

    HIVE_LLAMA_SERVER=/path/to/llama-server \
    HIVE_LIVE_MODEL=/path/to/tiny.gguf \
    python -m pytest tests/conformance/test_live_llama.py -q
"""

from __future__ import annotations

import json
import os
import socket
import time
import urllib.request
from pathlib import Path

import pytest

from conftest import parse_json, run_rust

LLAMA = os.environ.get("HIVE_LLAMA_SERVER")
MODEL = os.environ.get("HIVE_LIVE_MODEL")

pytestmark = pytest.mark.skipif(
    not LLAMA or not MODEL,
    reason="set HIVE_LLAMA_SERVER and HIVE_LIVE_MODEL to run the live engine test",
)


def _stack(tmp_path: Path) -> Path:
    stack = {
        "name": "live-cpu",
        "version": 1,
        "tiers": [
            {
                "role": "face",
                "repo": "local",
                "file": str(Path(MODEL).expanduser().resolve()),
                "ctx": 2048,
                "ngl": 0,
                "backend": "vulkan",
                "cache_k": "q8_0",
                "cache_v": "q8_0",
            }
        ],
    }
    path = tmp_path / "live.json"
    path.write_text(json.dumps(stack) + "\n")
    return path


def _wait_models(port: int, timeout: float = 60.0) -> dict:
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}/v1/models"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                return json.loads(response.read())
        except OSError:
            time.sleep(0.25)
    raise AssertionError(f"llama-server did not answer on port {port}")


def _chat(port: int) -> dict:
    body = json.dumps(
        {
            "model": "x",
            "messages": [{"role": "user", "content": "Say hi in three words."}],
            "max_tokens": 16,
            "temperature": 0,
        }
    ).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())


def test_live_serve_once(rust_bin, cli_env, tmp_path):
    env, state = cli_env
    stack = _stack(tmp_path)
    result = run_rust(
        rust_bin,
        [
            "serve",
            str(stack),
            "--once",
            "--json",
            "--llama-server",
            str(LLAMA),
        ],
        env,
    )
    assert result.returncode == 0, result.stderr
    payload = parse_json(result)
    assert payload["ok"] is True
    tier = payload["tiers"][0]
    assert tier["port"] > 0
    assert tier["model"]
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", tier["port"]), timeout=0.5)
    assert not list((state / "serve").glob("*.json"))


def test_live_detached_chat_and_stop(rust_bin, cli_env, tmp_path):
    env, state = cli_env
    stack = _stack(tmp_path)
    result = run_rust(
        rust_bin,
        [
            "serve",
            str(stack),
            "--detach",
            "--json",
            "--name",
            "live",
            "--llama-server",
            str(LLAMA),
        ],
        env,
    )
    assert result.returncode == 0, result.stderr
    record = parse_json(result)
    port = record["tiers"][0]["port"]
    try:
        models = _wait_models(port)
        assert models.get("data")
        answer = _chat(port)
        assert answer.get("choices"), answer
        status = run_rust(rust_bin, ["status", "--json"], env)
        assert status.returncode == 0
        view = parse_json(status)["serves"][0]
        assert view["state"] == "running"
        assert view["tiers"][0]["probe"]["up"] is True
    finally:
        stop = run_rust(rust_bin, ["stop", "live", "--json"], env)
        assert stop.returncode == 0, stop.stderr
    assert not (state / "serve" / "live.json").exists()
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", port), timeout=0.5)
