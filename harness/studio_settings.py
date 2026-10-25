"""Studio settings — the toggles the console owns.

Persisted next to the conversations (``harness_state/studio_settings.json``)
so they survive a restart.  The set is small and fixed; unknown keys are
ignored, so an older sidecar reading a newer file is harmless.

The keys:

``sandbox_enabled``
    Let the Studio run commands in the sandbox (Docker when the daemon is up).
    Off: ``/v1/sandbox/exec`` refuses.
``tools_enabled``
    Let the agent call tools (bash, files, code).  Off: the agent endpoint
    refuses and the Agent toggle is disabled.
``auto_compact``
    Compact a conversation into a checkpoint when it outgrows the window.
``strata_base``
    Where the Strata engine (the local 176B serve) lives.
``dev_mode``
    Reveal the developer surface: the deep-research queue, the Docker/VHDX
    setup tab, and the raw status.  Off by default; the ``/dev`` console
    command turns it on.
``harvest_enabled``
    Record every chat turn (the query, the reply, the curation that was
    used) to ``harness_state/harvest.jsonl``.  This is the data the
    encoder is trained on, so it is off by default: the file holds the
    user's own words.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

#: every boolean the Studio stores
BOOL_KEYS = ("sandbox_enabled", "tools_enabled", "auto_compact", "dev_mode",
             "harvest_enabled")
#: every string the Studio stores
STR_KEYS = ("strata_base", "strata_repo")
#: the grouped control blocks the Global tab owns
DICT_KEYS = ("memory_defaults", "quality_defaults", "mcp", "runtime_defaults",
             "encoder")

DEFAULTS: dict[str, Any] = {
    "sandbox_enabled": True,
    "tools_enabled": True,
    "auto_compact": True,
    "dev_mode": False,
    # the encoder's training data: every turn, with its curation
    "harvest_enabled": False,
    "strata_base": os.environ.get("CASCADE_SCION_BASE") or "http://127.0.0.1:8081/v1",
    "strata_repo": "",          # the Strata checkout (the sibling by default)
    # the harness-wide baseline: what a conversation gets when it overrides
    # nothing.  The Splinter tab layers this conversation's overrides on top.
    "memory_defaults": {},
    # the cascade's defaults: the judge, the repair, the gate, the escalation
    "quality_defaults": {},
    # the memory over MCP: the endpoint, its tools, what the agent may do
    "mcp": {"enabled": True, "tools": ["remember", "search"], "read_only": False},
    # the runtime baseline: the context, the GPU layers, the sampling
    "runtime_defaults": {},
    # the encoder: the small model that does the retrieval and the memory
    "encoder": {
        "embedding_model": "sentence-transformers/paraphrase-MiniLM-L3-v2",
        "embedding_backend": "local",
        "embedding_url": "",
        "medium_model": "microsoft/graphcodebert-base",
        # Controller-only lifecycle: unload the lazily loaded encoder after
        # this many idle seconds.  The serving engine is never touched.
        "controller_idle_unload": True,
        "controller_idle_seconds": 300,
    },
}


def settings_path(state_dir: str | Path | None = None) -> Path:
    base = Path(state_dir) if state_dir else Path(__file__).resolve().parents[1] / "harness_state"
    return base / "studio_settings.json"


def load_settings(state_dir: str | Path | None = None) -> dict[str, Any]:
    """The Studio's toggles, with the defaults filled in."""
    try:
        raw = json.loads(settings_path(state_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raw = {}
    if not isinstance(raw, dict):
        raw = {}
    out = dict(DEFAULTS)
    for key in BOOL_KEYS:
        if key in raw:
            out[key] = bool(raw[key])
    for key in STR_KEYS:
        value = str(raw.get(key) or "").strip()
        if value:
            out[key] = value
    for key in DICT_KEYS:
        value = raw.get(key)
        if isinstance(value, dict):
            merged = dict(DEFAULTS[key]) if isinstance(DEFAULTS.get(key), dict) else {}
            merged.update(value)
            out[key] = merged
    return out


def save_settings(data: dict[str, Any],
                  state_dir: str | Path | None = None, *,
                  remove: Mapping[str, Iterable[str]] | None = None
                  ) -> dict[str, Any]:
    """Apply the given keys to the stored settings and return the result.

    ``remove`` is used by controller rollback: a setting that did not exist
    before a proposal must be removed, not left behind as ``null``.  It is a
    mapping of block name to keys, and only known blocks may be touched.
    """
    path = settings_path(state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    # The Studio and a confirmed controller apply can write this file from
    # different threads/processes.  Keep the read-modify-write under the same
    # per-file lock the controller state uses.
    from harness.state_store import file_lock

    with file_lock(path):
        current = load_settings(state_dir)
        for key in BOOL_KEYS:
            if key in data:
                current[key] = bool(data[key])
        for key in STR_KEYS:
            value = str(data.get(key) or "").strip()
            if value:
                current[key] = value
        for key in DICT_KEYS:
            value = data.get(key)
            if isinstance(value, dict):
                merged = dict(current.get(key) or {})
                merged.update(value)
                current[key] = merged
        for block, keys in (remove or {}).items():
            if block not in DICT_KEYS:
                continue
            target = current.get(block)
            if not isinstance(target, dict):
                continue
            for key in keys:
                target.pop(str(key), None)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(current, indent=1))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        return current
