#!/usr/bin/env python
"""Sample host, GPU and process stability while a partner probe runs.

Writes a CSV (one row per interval) and a summary JSON on exit.  This is
deliberately read-only and cheap: it samples /proc and sysfs, never touches
the model server or the 176B serve.  It exits when --pid disappears, or on
SIGINT/SIGTERM.

    monitor_resources.py --pid 1234 --out logs/run.csv \
        --health-url http://127.0.0.1:8081/health
"""

from __future__ import annotations

import argparse
import csv
import json
import signal
import time
import urllib.request
from pathlib import Path

STOP = False


def _stop(_signum, _frame):
    global STOP
    STOP = True


def _meminfo() -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, rest = line.split(":", 1)
            out[key] = int(rest.strip().split()[0])
    except OSError:
        pass
    return out


def _vram() -> list[tuple[str, int, int]]:
    cards = []
    for total_path in sorted(Path("/sys/class/drm").glob("card*/device/mem_info_vram_total")):
        try:
            total = int(total_path.read_text())
            used = int((total_path.parent / "mem_info_vram_used").read_text())
            name = total_path.parent.parent.name
            cards.append((name, (total - used) // 1024 // 1024, total // 1024 // 1024))
        except (OSError, ValueError):
            continue
    return cards


def _rss_kb(pid: int) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except OSError:
        return None
    return None


def _health(url: str | None) -> int | None:
    if not url:
        return None
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            return 1 if response.status == 200 else 0
    except Exception:
        return 0


def _min(values):
    values = [v for v in values if v is not None]
    return min(values) if values else None


def _max(values):
    values = [v for v in values if v is not None]
    return max(values) if values else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--health-url", default="http://127.0.0.1:8081/health")
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "ts", "iso", "pid_alive", "rss_kb", "load1", "mem_available_kb",
        "swap_used_kb", "vram0_free_mib", "vram0_total_mib",
        "vram1_free_mib", "vram1_total_mib", "serve_health",
    ]
    rows: list[dict] = []
    started = time.time()
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        while not STOP:
            if args.pid is not None and not Path(f"/proc/{args.pid}").exists():
                break
            mem = _meminfo()
            cards = _vram()
            load = Path("/proc/loadavg").read_text().split()[0] if Path("/proc/loadavg").exists() else ""
            row = {
                "ts": round(time.time(), 3),
                "iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "pid_alive": 1 if args.pid and Path(f"/proc/{args.pid}").exists() else "",
                "rss_kb": _rss_kb(args.pid) if args.pid else None,
                "load1": load,
                "mem_available_kb": mem.get("MemAvailable"),
                "swap_used_kb": mem.get("SwapTotal", 0) - mem.get("SwapFree", 0),
                "vram0_free_mib": cards[0][1] if len(cards) > 0 else None,
                "vram0_total_mib": cards[0][2] if len(cards) > 0 else None,
                "vram1_free_mib": cards[1][1] if len(cards) > 1 else None,
                "vram1_total_mib": cards[1][2] if len(cards) > 1 else None,
                "serve_health": _health(args.health_url),
            }
            writer.writerow(row)
            handle.flush()
            rows.append(row)
            time.sleep(args.interval)

    ended = time.time()
    vram_names = [c[0] for c in _vram()]
    summary = {
        "label": args.label,
        "samples": len(rows),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started)),
        "ended": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(ended)),
        "duration_s": round(ended - started, 3),
        "pid": args.pid,
        "max_load1": _max([float(r["load1"]) for r in rows if r["load1"] not in ("", None)]),
        "min_mem_available_gib": round(_min([r["mem_available_kb"] for r in rows]) / 1024 / 1024, 3) if _min([r["mem_available_kb"] for r in rows]) is not None else None,
        "max_swap_used_gib": round(_max([r["swap_used_kb"] for r in rows]) / 1024 / 1024, 3) if _max([r["swap_used_kb"] for r in rows]) is not None else None,
        "max_rss_gib": round(_max([r["rss_kb"] for r in rows]) / 1024 / 1024, 3) if _max([r["rss_kb"] for r in rows]) is not None else None,
        "min_vram_free_mib": {
            name: _min([r[f"vram{i}_free_mib"] for r in rows])
            for i, name in enumerate(vram_names)
        },
        "health_failures": sum(1 for r in rows if r["serve_health"] == 0),
        "pid_exited": bool(args.pid and not Path(f"/proc/{args.pid}").exists()),
    }
    summary_path = args.out.with_name(args.out.stem + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
