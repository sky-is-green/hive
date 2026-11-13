"""Shared fixtures for the Python-vs-Rust CLI conformance suite.

The suite runs the same assertions against the installed Python CLI and the
Rust binary. Set ``HIVE_RUST_BIN`` to point at a binary outside the default
``rust/target/debug/hive`` build.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DEFAULT_RUST = REPO / "rust" / "target" / "debug" / "hive"


@pytest.fixture(scope="session")
def rust_bin() -> Path:
    raw = os.environ.get("HIVE_RUST_BIN")
    binary = Path(raw).expanduser() if raw else DEFAULT_RUST
    if not binary.is_file():
        pytest.skip(
            f"Rust binary not built at {binary}; run `cargo build --manifest-path rust/Cargo.toml`"
        )
    return binary


@pytest.fixture
def cli_env(tmp_path: Path) -> tuple[dict[str, str], Path]:
    state = tmp_path / "state"
    state.mkdir()
    env = os.environ.copy()
    env["HIVE_STATE_DIR"] = str(state)
    env["HIVE_PYTHON"] = sys.executable
    env.pop("HIVE_HOME", None)
    env.pop("HIVE_RUST_BIN", None)
    return env, state


def run_python(args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "harness.cli", *args],
        cwd=REPO,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )


def run_rust(
    binary: Path, args: list[str], env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(binary), *args],
        cwd=REPO,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )


def run_both(
    binary: Path, args: list[str], env: dict[str, str]
) -> tuple[subprocess.CompletedProcess[str], subprocess.CompletedProcess[str]]:
    return run_python(args, env), run_rust(binary, args, env)


def parse_json(result: subprocess.CompletedProcess[str]) -> object:
    assert result.returncode in (0, 1), result.stderr
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as error:
        pytest.fail(f"not JSON: {error}; stdout={result.stdout!r}; stderr={result.stderr!r}")


def write_gguf(path: Path, metadata: list[tuple[str, int, object]], *, version: int = 3) -> None:
    """Write a minimal GGUF header carrying the given metadata entries.

    ``type_id`` is the GGUF metadata type (4 uint32, 8 string, 9 array of
    strings). The file ends after the KV section; no tensor data is needed for
    the header parser.
    """
    out = bytearray(b"GGUF")
    out += struct.pack("<I", version)
    out += struct.pack("<Q", 0)
    out += struct.pack("<Q", len(metadata))
    for key, kind, value in metadata:
        encoded = key.encode()
        out += struct.pack("<Q", len(encoded))
        out += encoded
        out += struct.pack("<I", kind)
        if kind == 4:
            out += struct.pack("<I", int(value))
        elif kind == 8:
            text = str(value).encode()
            out += struct.pack("<Q", len(text))
            out += text
        elif kind == 9:
            items = list(value)
            out += struct.pack("<I", 8)
            out += struct.pack("<Q", len(items))
            for item in items:
                text = str(item).encode()
                out += struct.pack("<Q", len(text))
                out += text
        else:  # pragma: no cover - only the types above are used here
            raise ValueError(f"unsupported GGUF type {kind}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(out))


def sample_gguf_metadata() -> list[tuple[str, int, object]]:
    return [
        ("general.architecture", 8, "llama"),
        ("general.name", 8, "Tiny Test Model"),
        ("general.file_type", 4, 15),
        ("llama.context_length", 4, 8192),
        ("llama.block_count", 4, 32),
        ("llama.attention.head_count", 4, 32),
        ("llama.attention.head_count_kv", 4, 8),
        ("llama.embedding_length", 4, 4096),
        ("llama.attention.key_length", 4, 128),
        ("llama.attention.value_length", 4, 128),
        ("tokenizer.ggml.tokens", 9, ["a", "b"]),
    ]


def proc_cmdline(pid: int) -> str:
    raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    return raw.replace(b"\0", b" ").decode("utf-8", "replace").strip()


def write_record(state: Path, name: str, record: dict) -> None:
    directory = state / "serve"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")


@pytest.fixture
def live_proc():
    process = subprocess.Popen(["sleep", "30"])
    try:
        yield process
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


@pytest.fixture
def fixture_library(tmp_path: Path) -> Path:
    library = tmp_path / "models"
    write_gguf(
        library / "Tiny-1.7B-Q4_K_M.gguf",
        sample_gguf_metadata(),
    )
    nested = library / "nested"
    nested.mkdir()
    write_gguf(
        nested / "Tiny-VL-2B-Q8_0.gguf",
        sample_gguf_metadata(),
    )
    # A truncated header must yield no metadata, not crash the listing.
    (library / "broken.gguf").write_bytes(b"GGUF\x03\x00\x00\x00")
    (library / "readme.txt").write_text("not a model")
    # Stable mtimes keep the newest-first ordering identical between runs.
    os.utime(library / "Tiny-1.7B-Q4_K_M.gguf", (1_700_000_000, 1_700_000_000))
    os.utime(nested / "Tiny-VL-2B-Q8_0.gguf", (1_700_100_000, 1_700_100_000))
    os.utime(library / "broken.gguf", (1_700_200_000, 1_700_200_000))
    return library


@pytest.fixture
def fixture_stacks(tmp_path: Path) -> Path:
    stacks = tmp_path / "stacks"
    stacks.mkdir()

    def write(name: str, payload: object) -> None:
        path = stacks / f"{name}.json"
        if isinstance(payload, str):
            path.write_text(payload)
        else:
            path.write_text(json.dumps(payload))

    write(
        "b-second",
        {
            "name": "b-second",
            "version": 1,
            "tiers": [{"role": "face", "repo": "local", "file": "a.gguf"}],
        },
    )
    write(
        "a-first",
        {
            "name": "a-first",
            "version": 1,
            "tiers": [
                {"role": "worker", "repo": "local", "file": "b.gguf"},
                {"role": "face", "repo": "local", "file": "a.gguf"},
            ],
        },
    )
    write("c-broken", "{not json")
    write(
        "d-unknown",
        {
            "name": "d-unknown",
            "version": 1,
            "tiers": [],
            "typo": True,
        },
    )
    write(
        "e-bad-role",
        {
            "name": "e-bad-role",
            "version": 1,
            "tiers": [{"role": "boss", "repo": "local", "file": "a.gguf"}],
        },
    )
    write(
        "f-bad-engine-arg",
        {
            "name": "f-bad-engine-arg",
            "version": 1,
            "tiers": [
                {
                    "role": "face",
                    "repo": "local",
                    "file": "a.gguf",
                    "engine_args": ["--host", "0.0.0.0"],
                }
            ],
        },
    )
    write(
        "g-bad-launch",
        {
            "name": "g-bad-launch",
            "version": 1,
            "tiers": [
                {
                    "role": "face",
                    "repo": "local",
                    "file": "a.gguf",
                    "endpoint": "http://127.0.0.1:8080/v1",
                    "launch": {
                        "kind": "boss",
                        "repo": "repo",
                        "config": "cfg.json",
                    },
                }
            ],
        },
    )
    return stacks


@pytest.fixture
def fake_engine(tmp_path: Path) -> Path:
    script = tmp_path / "fake-llama-server"
    script.write_text(
        f"""#!{sys.executable}
import argparse, json, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model")
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, required=True)
parser.add_argument("-ngl")
parser.add_argument("-c")
args, _ = parser.parse_known_args()

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/v1/models", "/models"):
            payload = {{"data": [{{"id": "fake-model"}}]}}
        elif self.path == "/health":
            payload = {{"status": "ok", "loaded": True, "model": "fake-model"}}
        elif self.path == "/slots":
            payload = [{{"id": 0, "is_processing": False}}]
        else:
            self.send_error(404)
            return
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, fmt, *args):
        print(fmt % args, file=sys.stderr, flush=True)

ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
"""
    )
    script.chmod(0o755)
    return script


@pytest.fixture
def serve_fixture(tmp_path: Path, fake_engine: Path) -> dict:
    models = tmp_path / "serve-models"
    models.mkdir()
    write_gguf(models / "tiny.gguf", sample_gguf_metadata())
    stacks = tmp_path / "serve-stacks"
    stacks.mkdir()
    stack = {
        "name": "fake-one",
        "version": 1,
        "tiers": [
            {
                "role": "face",
                "repo": "local",
                "file": "tiny.gguf",
                "ctx": 2048,
                "ngl": 0,
                "backend": "vulkan",
                "cache_k": "q8_0",
                "cache_v": "q8_0",
            }
        ],
    }
    (stacks / "fake-one.json").write_text(json.dumps(stack) + "\n")
    return {
        "models": models,
        "stacks": stacks,
        "stack": stacks / "fake-one.json",
        "engine": fake_engine,
    }
