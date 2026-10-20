"""Leak probe — does an unload really give the memory back?

The Studio can load a model and unload it again.  That is only useful if
the unload actually returns the memory: a server that lingers (or a
process that never dies) looks unloaded and eats the box.  This probe
loads and unloads a model a number of times and asserts that the VRAM and
the RAM come back each time.

It is opt-in — it loads a real model, so it is not part of the offline
suite.  Point it at a running Studio::

    python experiments/leak_probe.py --model qwen35-release-v2-cur05-soup --iters 3

The shape (a tolerance-bounded reclaim check over a history) is the one
the DSH side uses for its own leak gate; the numbers here are measured,
not simulated.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request

#: the memory must come back within this share (+ this many MB)
RECLAIM_TOL = 0.05
RECLAIM_SLACK_MB = 50.0


def is_reclaimed(before: dict, after: dict, tol: float = RECLAIM_TOL,
                 slack_mb: float = RECLAIM_SLACK_MB) -> bool:
    """True when the memory came back after the unload."""
    vram_ok = (abs(after["vramMb"] - before["vramMb"])
               <= before["vramMb"] * tol + slack_mb)
    ram_ok = (abs(after["ramMb"] - before["ramMb"])
              <= before["ramMb"] * tol + slack_mb)
    return vram_ok and ram_ok


def detect_leak(history: list[dict], tol: float = RECLAIM_TOL) -> bool:
    """True when the memory never came back over the whole history."""
    if len(history) < 2:
        return False
    first, last = history[0], history[-1]
    return not is_reclaimed(first, last, tol) and last["vramMb"] > first["vramMb"]


class Studio:
    """The few Studio calls the probe needs."""

    def __init__(self, base: str = "http://127.0.0.1:8765") -> None:
        self.base = base.rstrip("/")

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            self.base + path, data=json.dumps(payload).encode(),
            headers={"content-type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=900) as resp:
            return json.loads(resp.read().decode() or "{}")

    def _get(self, path: str) -> dict:
        with urllib.request.urlopen(self.base + path, timeout=30) as resp:
            return json.loads(resp.read().decode() or "{}")

    def memory(self) -> dict:
        """The box's VRAM and RAM right now, in MB."""
        status = self._get("/v1/server/status")
        hw = status.get("hardware") or {}
        vram = float(hw.get("vram_free_gb") or 0.0) * 1024.0
        ram = float(hw.get("available_ram_gb") or 0.0) * 1024.0
        return {"vramMb": vram, "ramMb": ram, "timestamp": time.time()}

    def load(self, model: str) -> dict:
        return self._post("/v1/server/start", {"model": model})

    def unload(self) -> dict:
        return self._post("/v1/server/unload-loaded", {})


def run_probe(studio: Studio, model: str, iterations: int = 3) -> dict:
    """Load and unload ``model`` ``iterations`` times; the memory must return."""
    history: list[dict] = []
    for index in range(iterations):
        before = studio.memory()
        studio.load(model)
        loaded = studio.memory()
        studio.unload()
        after = studio.memory()
        if not is_reclaimed(before, after):
            raise AssertionError(
                f"iter {index}: the memory did not come back "
                f"(before {before['vramMb']:.0f}/{before['ramMb']:.0f} MB, "
                f"after {after['vramMb']:.0f}/{after['ramMb']:.0f} MB)")
        history.append(after)
        print(f"iter {index}: vram {loaded['vramMb']:.0f} -> {after['vramMb']:.0f} MB, "
              f"ram {loaded['ramMb']:.0f} -> {after['ramMb']:.0f} MB (reclaimed)")
    if detect_leak(history):
        raise AssertionError("a leak across the history")
    return {"iterations": iterations, "reclaimed": True, "history": history}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--studio", default="http://127.0.0.1:8765")
    parser.add_argument("--model", required=True, help="a local model file to load")
    parser.add_argument("--iters", type=int, default=3)
    args = parser.parse_args()
    try:
        out = run_probe(Studio(args.studio), args.model, args.iters)
    except (AssertionError, urllib.error.URLError) as exc:
        print(f"leak_probe: FAILED — {exc}")
        return 1
    print(f"leak_probe: {out['iterations']} iters ok, the memory came back")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
