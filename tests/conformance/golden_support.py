"""Frozen-contract helpers for the Rust CLI golden corpus.

The golden files are generated from the Python CLI once (see ``generate.py``)
and then compared against the Rust binary without starting Python. Volatile
values (absolute fixture paths, the interpreter version, the platform string)
are replaced with placeholders so the corpus is portable.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GOLDEN = Path(__file__).resolve().parent / "golden"


def write_gguf(path: Path, metadata: list[tuple[str, int, object]], *, version: int = 3) -> None:
    """Write a minimal GGUF header carrying the given metadata entries."""
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


def build_library(root: Path) -> Path:
    library = root / "models"
    write_gguf(library / "Tiny-1.7B-Q4_K_M.gguf", sample_gguf_metadata())
    nested = library / "nested"
    nested.mkdir()
    write_gguf(nested / "Tiny-VL-2B-Q8_0.gguf", sample_gguf_metadata())
    (library / "broken.gguf").write_bytes(b"GGUF\x03\x00\x00\x00")
    (library / "readme.txt").write_text("not a model")
    os.utime(library / "Tiny-1.7B-Q4_K_M.gguf", (1_700_000_000, 1_700_000_000))
    os.utime(nested / "Tiny-VL-2B-Q8_0.gguf", (1_700_100_000, 1_700_100_000))
    os.utime(library / "broken.gguf", (1_700_200_000, 1_700_200_000))
    return library


def build_stacks(root: Path) -> Path:
    stacks = root / "stacks"
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
    return stacks


def build_env(root: Path, python: str | None = None) -> dict[str, str]:
    home = root / "home"
    state = root / "state"
    home.mkdir()
    state.mkdir()
    env = os.environ.copy()
    env.pop("HIVE_RUST_BIN", None)
    env["HIVE_HOME"] = str(home)
    env["HIVE_STATE_DIR"] = str(state)
    env["HIVE_PYTHON"] = python or sys.executable
    env["TZ"] = "UTC"
    return env


def normalize(value: object, root: Path, python: str) -> object:
    """Replace volatile values with stable placeholders."""
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key in ("platform", "python") and isinstance(item, str) and "/" not in item:
                # version --json reports a version string; config reports a path.
                placeholder = "<PLATFORM>" if key == "platform" else "<PYTHON_VERSION>"
                out[key] = placeholder
            else:
                out[key] = normalize(item, root, python)
        return out
    if isinstance(value, list):
        return [normalize(item, root, python) for item in value]
    if isinstance(value, str):
        for path, placeholder in (
            (str(root), "<TMP>"),
            (str(REPO), "<REPO>"),
            (python, "<PYTHON>"),
        ):
            if path and value.startswith(path):
                return placeholder + value[len(path) :]
        return value
    return value


#: Top-level commands the Rust binary has ported. The Python root catalog also
#: lists Python-only verbs; the frozen corpus covers the ported subset and the
#: per-topic parity test allows the rest to be absent.
RUST_COMMANDS = {
    "serve",
    "tui",
    "status",
    "stop",
    "logs",
    "models",
    "stacks",
    "backends",
    "setup",
    "doctor",
    "config",
    "ui",
    "version",
    "help",
}


def trim_help_root(payload: dict) -> dict:
    trimmed = dict(payload)
    trimmed["commands"] = [
        command
        for command in payload.get("commands", [])
        if command.get("name") in RUST_COMMANDS
    ]
    return trimmed


CASES: list[tuple[str, list[str]]] = [
    ("version", ["version", "--json"]),
    ("help", ["help", "--json"]),
    (
        "config",
        [
            "config",
            "--models-dir",
            "{models}",
            "--stacks-dir",
            "{stacks}",
            "--json",
        ],
    ),
    ("models_list", ["models", "list", "--models-dir", "{models}", "--json"]),
    ("models_dir", ["models", "dir", "--models-dir", "{models}", "--json"]),
    ("stacks_list", ["stacks", "list", "--stacks-dir", "{stacks}", "--json"]),
    ("stacks_show", ["stacks", "show", "a-first", "--stacks-dir", "{stacks}", "--json"]),
]


def expand(args: list[str], root: Path) -> list[str]:
    values = {
        "{models}": str(root / "models"),
        "{stacks}": str(root / "stacks"),
    }
    return [values.get(arg, arg) for arg in args]


def render(
    runner: list[str],
    case_args: list[str],
    root: Path,
    env: dict[str, str],
    *,
    module: bool = False,
) -> object:
    command = [*runner, *expand(case_args, root)]
    if module:
        command = [*runner, "-m", "harness.cli", *expand(case_args, root)]
    result = subprocess.run(
        command,
        cwd=REPO,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{' '.join(command)} failed ({result.returncode}): {result.stderr}"
        )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"not JSON: {error}; stdout={result.stdout!r}") from error
    return normalize(payload, root, env["HIVE_PYTHON"])
