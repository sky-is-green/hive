"""Launch verification — is this launch actually wired up?

Before the Studio starts a model there is a short list of things that can
be wrong, and every one of them has a fix a person can apply.  This probe
checks the list and says what to do, instead of letting a launch fail
three minutes later with a stack trace.

    python experiments/launch_verify.py --model <file> [--ctx 32768]

It answers four questions:

  1. do the weights exist, and is the first shard of a split model found
  2. does the model fit the box (the tiers, the disk guard, the cap)
  3. is the engine binary there
  4. is the port free

Exit 0 when the launch is ready, 1 with the fixes when it is not.  The
DSH side has a probe of the same name for its Windows/VHDX/Docker path;
this one is the local-box equivalent, and the shape (check, then an
actionable fix) is theirs.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import urllib.request
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.hardware import fit_plan

#: the default port the Studio's llama.cpp servers use
DEFAULT_PORT = 1234
#: where the Studio's binaries live
BIN_DIR = Path(__file__).resolve().parents[1] / "tools" / "backends"


def find_shards(directory: str | Path) -> list[Path]:
    """Every GGUF under ``directory``."""
    root = Path(directory).expanduser()
    if not root.is_dir():
        return []
    return sorted(root.rglob("*.gguf"))


def bootstrap_target(shards: list[Path]) -> Optional[Path]:
    """The file a split model starts from: its first shard.

    A model split across files is named ``<name>-00001-of-000NN.gguf``;
    the launcher has to be pointed at the first one.
    """
    if not shards:
        return None
    numbered = [p for p in shards if "-00001-of-" in p.name]
    if numbered:
        return sorted(numbered)[0]
    return sorted(shards)[0]


def port_free(port: int, host: str = "127.0.0.1") -> bool:
    """True when nothing is listening on the port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.4)
        return sock.connect_ex((host, port)) != 0


def engine_binary() -> Optional[Path]:
    """A llama.cpp binary the Studio can launch."""
    for backend in ("rocm", "vulkan", "cuda", "cpu"):
        candidate = BIN_DIR / backend / "llama-server"
        if candidate.exists():
            return candidate
    return None


def hardware() -> dict:
    """The box's VRAM/RAM/disk, as the Studio sees it."""
    from harness.app import _hardware_summary

    return _hardware_summary()


def verify(model: str | None, ctx: int = 8192, port: int = DEFAULT_PORT,
           models_dir: str | None = None) -> dict:
    """Check the launch; return the checks and the fixes."""
    checks: list[dict] = []
    fixes: list[str] = []

    # 1. the weights
    model_path: Optional[Path] = None
    if model:
        candidate = Path(model).expanduser()
        if candidate.is_file():
            model_path = candidate
        else:
            shards = find_shards(models_dir or candidate.parent)
            model_path = bootstrap_target(shards)
            if model_path is None:
                fixes.append(
                    f"no GGUF found for '{model}' - check the path, or set the "
                    f"library folder in the Local Library tab")
    size_gb = 0.0
    if model_path is not None and model_path.exists():
        size_gb = model_path.stat().st_size / (1024 ** 3)
    checks.append({"name": "the weights", "ok": model_path is not None,
                   "detail": str(model_path) if model_path else "missing",
                   "fix": "" if model_path else "point at a GGUF that exists"})

    # 2. does it fit
    hw = hardware()
    vram = float(hw.get("combined_vram_gb") or hw.get("vram_gb") or 0.0)
    ram = float(hw.get("available_ram_gb") or hw.get("total_ram_gb") or 0.0)
    disk = (hw.get("disk") or {}).get("repo") or {}
    plan = fit_plan(size_gb, ctx, vram_gb=vram, ram_gb=ram,
                    disk_free_gb=disk.get("free_gb"))
    fits = plan["flags"]["disk_full"] is False
    checks.append({"name": "it fits the box", "ok": fits,
                   "detail": f"{plan['metrics']['total_gb']:.1f} GB needed "
                             f"(vram {plan['metrics']['tier1_vram_gb']:.1f}, "
                             f"ram {plan['metrics']['tier2_ram_gb']:.1f}, "
                             f"disk {plan['metrics']['tier3_disk_gb']:.1f})",
                   "fix": "" if fits else "the disk is nearly full - free it, "
                                           "or use a smaller context"})
    if plan["flags"]["io_latency_warning"]:
        fixes.append("part of the KV cache lands on the disk - long prompts "
                     "get slower")

    # 3. the binary
    binary = engine_binary()
    checks.append({"name": "the engine binary", "ok": binary is not None,
                   "detail": str(binary) if binary else "none found",
                   "fix": "" if binary else "build or install llama-server "
                                           "under tools/backends/"})

    # 4. the port
    free = port_free(port)
    checks.append({"name": "the port", "ok": free,
                   "detail": f"{port} {'is free' if free else 'is taken'}",
                   "fix": "" if free else f"something already listens on "
                                           f"{port} - unload it, or pick "
                                           f"another port"})

    return {"ok": all(c["ok"] for c in checks), "checks": checks,
            "fixes": fixes, "plan": plan}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", help="a GGUF file (or a folder of them)")
    parser.add_argument("--models-dir", help="where to look for the weights")
    parser.add_argument("--ctx", type=int, default=8192)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--json", action="store_true", help="the raw report")
    args = parser.parse_args()
    report = verify(args.model, args.ctx, args.port, args.models_dir)
    if args.json:
        print(json.dumps(report, indent=1))
    else:
        for check in report["checks"]:
            mark = "ok " if check["ok"] else "FAIL"
            print(f"[{mark}] {check['name']}: {check['detail']}")
            if check["fix"]:
                print(f"        fix: {check['fix']}")
        for fix in report["fixes"]:
            print(f"  note: {fix}")
        print("launch_verify:", "ready" if report["ok"] else "NOT ready")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
