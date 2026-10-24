"""What each llama-server backend build is, where it lives, and whether it is installed.

The stack schema's ``backend`` field is the executable choice; this module is
the declaration behind it.  It exists so ``hive serve`` can say *which* builds
are installed, *why* it picked one, and what a user is missing — instead of
probing a directory convention and failing with "no llama-server binary for
backend 'rocm'".

The shape follows LocalAI's backend gallery (``backend/index.yaml``): one entry
per build with its acceleration variants and a note.  It deliberately does not
follow LocalAI's gRPC transport — our engines speak OpenAI over HTTP.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "tools" / "backends" / "manifest.json"

#: Which build to prefer for an accelerator, best first.  A vendor build beats
#: the portable one; the fork is a deliberate choice, so it is not auto-picked
#: ahead of rocm/cuda; CPU is last and only matches ``cpu``.
_PREFERENCE: tuple[str, ...] = ("rocm", "cuda", "sycl", "vulkan", "scion", "cpu")


def load(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Read the manifest.  Raises ``FileNotFoundError``/``ValueError``."""
    target = Path(path) if path else MANIFEST_PATH
    data = json.loads(target.read_text(encoding="utf-8"))
    entries = data.get("backends")
    if not isinstance(entries, dict) or not entries:
        raise ValueError(f"backend manifest {target} has no 'backends' object")
    return entries


def names(path: str | Path | None = None) -> tuple[str, ...]:
    """The declared backend names, in manifest order."""
    return tuple(load(path))


def resolve(name: str, *, root: str | Path | None = None,
            path: str | Path | None = None) -> dict[str, Any]:
    """One backend with its absolute binary path and whether it is installed."""
    entries = load(path)
    key = str(name).strip().lower()
    if key not in entries:
        raise KeyError(f"unknown backend {name!r}; known: {', '.join(entries)}")
    entry = dict(entries[key])
    entry["name"] = key
    base = Path(root) if root else REPO_ROOT
    exe = "llama-server.exe" if os.name == "nt" else "llama-server"
    binary = base / str(entry.get("binary_dir") or f"tools/backends/{key}") / exe
    entry["binary"] = str(binary)
    entry["installed"] = binary.is_file()
    return entry


def installed(*, root: str | Path | None = None,
              path: str | Path | None = None) -> list[str]:
    """Declared backends whose binary is present, in manifest order."""
    return [name for name in names(path)
            if resolve(name, root=root, path=path)["installed"]]


def for_accelerator(kind: str, *, root: str | Path | None = None,
                    path: str | Path | None = None) -> Optional[str]:
    """The installed backend to use for ``amd``/``nvidia``/``intel``/``cpu``.

    ``None`` means nothing suitable is installed; the caller should fall back
    to the default llama-server binary rather than guess.
    """
    kind = str(kind or "").strip().lower()
    order = {name: index for index, name in enumerate(_PREFERENCE)}
    candidates = [
        name for name in names(path)
        if kind in (resolve(name, root=root, path=path).get("accelerators") or [])
        and resolve(name, root=root, path=path)["installed"]
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda name: order.get(name, len(order)))
    return candidates[0]


def rows(*, root: str | Path | None = None,
         path: str | Path | None = None) -> list[dict[str, Any]]:
    """Every declared backend, for ``hive serve --backends``."""
    return [resolve(name, root=root, path=path) for name in names(path)]
