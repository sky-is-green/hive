"""Where Hive keeps the things it downloads and changes at runtime.

A source checkout keeps the historical layout: ``stacks/``, ``models/gguf/``,
``tools/backends/`` and ``harness_state/`` live next to ``pyproject.toml``.  An
installed wheel cannot write into site-packages, so it uses ``HIVE_HOME``
instead (default ``~/.hive``), with the same directory names underneath.  Set
``HIVE_HOME`` explicitly to move a checkout's mutable state somewhere else.

Only *mutable* state moves.  The backend manifest is source data and ships
inside the package; :func:`manifest_path` prefers the checkout copy so local
edits to ``tools/backends/manifest.json`` still win.
"""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PACKAGE_ROOT.parent
PACKAGED_MANIFEST = PACKAGE_ROOT / "data" / "backends-manifest.json"


def is_source_checkout() -> bool:
    """True when the package is running from a checkout, not a wheel."""
    return (SOURCE_ROOT / "pyproject.toml").is_file() and (
        SOURCE_ROOT / "harness" / "paths.py").is_file()


def home() -> Path:
    """The writable Hive root (``HIVE_HOME``, else repo or ``~/.hive``)."""
    raw = os.environ.get("HIVE_HOME")
    if raw:
        return Path(raw).expanduser()
    return SOURCE_ROOT if is_source_checkout() else Path.home() / ".hive"


def ensure_home() -> Path:
    root = home()
    root.mkdir(parents=True, exist_ok=True)
    return root


def state_dir() -> Path:
    """Serve records and other runtime state (``HIVE_STATE_DIR`` overrides)."""
    raw = os.environ.get("HIVE_STATE_DIR")
    return Path(raw).expanduser() if raw else home() / "harness_state"


def stacks_dir() -> Path:
    return home() / "stacks"


def models_dir() -> Path:
    return home() / "models" / "gguf"


def logs_dir() -> Path:
    return home() / "logs"


def backend_root() -> Path:
    """Parent that contains ``tools/backends/<name>/llama-server``."""
    return home()


def manifest_path() -> Path:
    """The backend manifest: the checkout copy when present, else package data."""
    source = SOURCE_ROOT / "tools" / "backends" / "manifest.json"
    if source.is_file():
        return source
    return PACKAGED_MANIFEST
