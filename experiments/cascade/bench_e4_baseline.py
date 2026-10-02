#!/usr/bin/env python
"""E4 baseline protocol: Qwen3.8-Flash-Next Q2_0 on the AMD box.

Runs the protocol from ``docs/ENGINE.md`` ("E4 baseline protocol"): for each
``-ncmoe`` arm, load the prism ROCm ``llama-server`` directly, then measure
cold 1K/4K prefill, warm prefill, 256-token decode, expert slot-cache hit
rate, host-RAM / VRAM peaks and disk MB per generation. Every arm writes a
manifest with provenance under ``experiments/cascade/results/<run>/<arm>/``.

Usage:

    .venv/bin/python experiments/cascade/bench_e4_baseline.py --arms all
    .venv/bin/python experiments/cascade/bench_e4_baseline.py --arms 12 --dry-run

Arms (ENGINE.md protocol):

    ncmoe0          -ncmoe 0                     (all-resident baseline)
    ncmoe12-mscauto -ncmoe 12 -msc auto
    ncmoe24-msc256  -ncmoe 24 -msc 256

The box runs one heavy track at a time; ask the human before starting.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CASCADE = REPO / "experiments" / "cascade"
RESULTS = CASCADE / "results"
SERVER = REPO / "tools" / "backends" / "rocm" / "llama-server"
PRISM = Path.home() / "Desktop/work/prism-ml-llama.cpp"
MODEL = (
    Path.home()
    / "Desktop/work/models/qwen38-q2_0/Q2_0"
    / "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf"
)
WIKI = Path.home() / "Desktop/work/ternary-serve/wiki.test.raw"
MODEL_TREE = (
    Path.home()
    / "Desktop/work/models/qwen38-q2_0/.cache/huggingface/trees"
    / "2a55d75962e22f7a4a1d9963ab6eae3678537831.json"
)

# Protocol flags (ENGINE.md). HIP_VISIBLE_DEVICES=1,0 = headless-first, as in
# the published 125B numbers; -ts 0.522,0.478 splits with the display card
# first.
COMMON_ARGS = [
    "-ngl", "99",
    "-c", "4096",
    "-np", "1",
    "-t", "8",
    "-ts", "0.522,0.478",
    "--split-mode", "layer",
    "-lm", "mmap",
    "--lazy-mode", "on",
    "-fa", "auto",
]
ARMS = [
    {"id": "ncmoe0", "label": "all-resident", "args": ["-ncmoe", "0"]},
    {"id": "ncmoe12-mscauto", "label": "-ncmoe 12 -msc auto", "args": ["-ncmoe", "12", "-msc", "auto"]},
    {"id": "ncmoe24-msc256", "label": "-ncmoe 24 -msc 256", "args": ["-ncmoe", "24", "-msc", "256"]},
]

# Prompt slices from wiki.test.raw. ~3.8 chars/token on the 125B; the 4K slice
# stays under -c 4096 even after n_predict.
PROMPTS = {
    "cold1k": {"chars": 3800, "n_predict": 16, "cache_prompt": False},
    "cold4k": {"chars": 13500, "n_predict": 16, "cache_prompt": False},
    "warm1k": {"chars": 3800, "n_predict": 16, "cache_prompt": False},
    "decode1": {"chars": 700, "n_predict": 256, "cache_prompt": False, "ignore_eos": True},
    "decode2": {"chars": 700, "n_predict": 256, "cache_prompt": False, "ignore_eos": True},
    "cached": {"chars": 700, "n_predict": 256, "cache_prompt": True, "ignore_eos": True},
}

SLOT_INIT_RE = re.compile(
    r"MoE slot cache: (\d+) layer\(s\), (\d+) slots each, ([\d.]+) MiB on device\(s\)(.*)"
)
SLOT_STATS_RE = re.compile(
    r"MoE slot cache: (\d+) layer\(s\) x (\d+) slots, (\d+) hits / (\d+) fetches \(([\d.]+)% hit rate\)"
)


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def run_cmd(args: list[str], cwd: Path | None = None, timeout: int = 30) -> str:
    try:
        out = subprocess.run(
            args, cwd=cwd, capture_output=True, text=True, timeout=timeout
        )
        return (out.stdout or out.stderr).strip()
    except Exception as exc:  # noqa: BLE001 - provenance is best-effort
        return f"<error: {exc}>"


def git_state(path: Path) -> dict:
    def git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=path, capture_output=True, text=True, timeout=10
            ).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""

    dirty = [line for line in git("status", "--porcelain").splitlines() if line]
    return {
        "branch": git("branch", "--show-current"),
        "commit": git("rev-parse", "HEAD"),
        "dirty_files": len(dirty),
    }


def sha256_file(path: Path, cache_path: Path) -> str:
    """File sha256 with a path/size/mtime cache (GGUFs are GBs)."""
    stat = path.stat()
    key = f"{path}|{stat.st_size}|{int(stat.st_mtime)}"
    cache: dict[str, str] = {}
    if cache_path.is_file():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except ValueError:
            cache = {}
    if key in cache:
        return cache[key]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    cache[key] = digest.hexdigest()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    return cache[key]


def read_int(path: Path | str, default: int = 0) -> int:
    try:
        return int(Path(path).read_text().strip())
    except Exception:  # noqa: BLE001
        return default


def parse_meminfo() -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            out[key] = int(rest.strip().split()[0])
    except Exception:  # noqa: BLE001
        pass
    return out


def read_proc_status(pid: int, key: str) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith(key):
                return int(line.split()[1])
    except Exception:  # noqa: BLE001
        pass
    return 0


def read_proc_io(pid: int, key: str) -> int:
    try:
        for line in Path(f"/proc/{pid}/io").read_text().splitlines():
            if line.startswith(key):
                return int(line.split()[1])
    except Exception:  # noqa: BLE001
        pass
    return 0


def vram_used_kb(card: int) -> int:
    return read_int(f"/sys/class/drm/card{card}/device/mem_info_vram_used") // 1024


def gpu_busy(card: int) -> int:
    return read_int(f"/sys/class/drm/card{card}/device/gpu_busy_percent", -1)


def find_nvme() -> str:
    for entry in sorted(Path("/sys/block").glob("nvme*n1")):
        return entry.name
    return ""


NVME = find_nvme()


def disk_sectors() -> tuple[int, int]:
    """(read, written) sectors of the first NVMe device, 512 B each."""
    if not NVME:
        return 0, 0
    try:
        fields = (Path("/sys/block") / NVME / "stat").read_text().split()
        return int(fields[2]), int(fields[6])
    except Exception:  # noqa: BLE001
        return 0, 0


def now_iso() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


# --------------------------------------------------------------------------
# sampler
# --------------------------------------------------------------------------

class Sampler(threading.Thread):
    """10 Hz time series of RAM / VRAM / GPU busy / process I/O."""

    def __init__(self, pid: int, interval: float = 0.1):
        super().__init__(daemon=True)
        self.pid = pid
        self.interval = interval
        self.phase = "load"
        self.rows: list[dict] = []
        self._stop_evt = threading.Event()

    def stop(self) -> None:
        self._stop_evt.set()

    def run(self) -> None:
        started = time.monotonic()
        while not self._stop_evt.is_set():
            self.rows.append(self._sample(time.monotonic() - started))
            self._stop_evt.wait(self.interval)

    def _sample(self, t: float) -> dict:
        mem = parse_meminfo()
        sys_r, sys_w = disk_sectors()
        return {
            "t_s": round(t, 3),
            "phase": self.phase,
            "mem_available_kb": mem.get("MemAvailable", 0),
            "mem_free_kb": mem.get("MemFree", 0),
            "cached_kb": mem.get("Cached", 0),
            "server_rss_kb": read_proc_status(self.pid, "VmRSS"),
            "vram0_kb": vram_used_kb(0),
            "vram1_kb": vram_used_kb(1),
            "vram2_kb": vram_used_kb(2),
            "gpu0_busy": gpu_busy(0),
            "gpu1_busy": gpu_busy(1),
            "proc_read_bytes": read_proc_io(self.pid, "read_bytes"),
            "proc_write_bytes": read_proc_io(self.pid, "write_bytes"),
            "sys_read_sectors": sys_r,
            "sys_write_sectors": sys_w,
        }


def phase_peaks(rows: list[dict], phase: str) -> dict:
    sel = [r for r in rows if r["phase"] == phase]
    if not sel:
        return {}
    return {
        "samples": len(sel),
        "mem_available_min_kb": min(r["mem_available_kb"] for r in sel),
        "server_rss_peak_kb": max(r["server_rss_kb"] for r in sel),
        "vram_peak_kb": {
            f"card{i}": max(r[f"vram{i}_kb"] for r in sel) for i in range(3)
        },
        "gpu_busy_mean": {
            f"card{i}": round(
                sum(max(r[f"gpu{i}_busy"], 0) for r in sel) / len(sel), 1
            ) for i in range(2)
        },
    }


# --------------------------------------------------------------------------
# server + requests
# --------------------------------------------------------------------------

def start_server(arm: dict, port: int, log_path: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = "1,0"
    env["LLAMA_MOE_SLOT_STATS"] = "1"
    cmd = [
        str(SERVER), "-m", str(MODEL), *COMMON_ARGS, *arm["args"],
        "--host", "127.0.0.1", "--port", str(port),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log:
        proc = subprocess.Popen(
            cmd, cwd=str(REPO), env=env,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
    return proc


def wait_ready(proc: subprocess.Popen, log_path: Path, timeout: int, min_free_kb: int) -> dict:
    """Wait for 'listening on'; abort on process death or free-RAM floor."""
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        if proc.poll() is not None:
            return {"ok": False, "error": f"server exited rc={proc.returncode}", "load_s": time.monotonic() - started}
        try:
            text = log_path.read_text(errors="replace")
        except Exception:  # noqa: BLE001
            text = ""
        if "listening on" in text:
            return {"ok": True, "load_s": round(time.monotonic() - started, 1)}
        if parse_meminfo().get("MemAvailable", 1 << 40) < min_free_kb:
            return {"ok": False, "error": "free RAM below floor during load", "load_s": time.monotonic() - started}
        time.sleep(0.5)
    return {"ok": False, "error": f"timeout after {timeout}s", "load_s": time.monotonic() - started}


def stop_server(proc: subprocess.Popen, timeout: int = 240) -> dict:
    """SIGINT for a clean context teardown (prints the slot-cache stats)."""
    result = {"exited": False, "signal": None}
    if proc.poll() is not None:
        result.update(exited=True, rc=proc.returncode)
        return result
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=timeout)
        result.update(exited=True, rc=proc.returncode, signal="SIGINT")
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=30)
            result.update(exited=True, rc=proc.returncode, signal="SIGTERM")
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
            result.update(exited=True, rc=proc.returncode, signal="SIGKILL")
    return result


def parse_slot_stats(log_path: Path) -> dict:
    try:
        text = log_path.read_text(errors="replace")
    except Exception:  # noqa: BLE001
        return {}
    out: dict = {}
    init = SLOT_INIT_RE.findall(text)
    if init:
        layers, slots, mib, tail = init[-1]
        out["slots_per_layer"] = int(slots)
        out["layers"] = int(layers)
        out["vram_mib"] = float(mib)
        out["promoted_note"] = tail.strip()
    stats = SLOT_STATS_RE.findall(text)
    if stats:
        layers, slots, hits, fetches, rate = stats[-1]
        out["hits"] = int(hits)
        out["fetches"] = int(fetches)
        out["hit_rate_pct"] = float(rate)
        out["stats_layers"] = int(layers)
        out["stats_slots"] = int(slots)
    return out


def completion(port: int, prompt: str, n_predict: int, cache_prompt: bool, pid: int,
               ignore_eos: bool = False) -> dict:
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0,
        "cache_prompt": cache_prompt,
        "seed": 1234,
    }
    if ignore_eos:
        payload["ignore_eos"] = True
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    io_r0 = read_proc_io(pid, "read_bytes")
    io_w0 = read_proc_io(pid, "write_bytes")
    sys_r0, sys_w0 = disk_sectors()
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=1800) as resp:
            data = json.load(resp)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)[:300], "wall_s": round(time.monotonic() - t0, 2)}
    wall = time.monotonic() - t0
    timings = data.get("timings") or {}
    content = str(data.get("content") or "")
    prompt_n = int(timings.get("prompt_n") or data.get("tokens_evaluated") or 0)
    prompt_ms = float(timings.get("prompt_ms") or 0)
    pred_n = int(timings.get("predicted_n") or data.get("tokens_predicted") or 0)
    pred_ms = float(timings.get("predicted_ms") or 0)
    sys_r1, sys_w1 = disk_sectors()
    return {
        "ok": True,
        "wall_s": round(wall, 2),
        "prompt_n": prompt_n,
        "prompt_ms": round(prompt_ms, 1),
        "prefill_tps": round(1000 * prompt_n / prompt_ms, 1) if prompt_ms else 0.0,
        "predicted_n": pred_n,
        "predicted_ms": round(pred_ms, 1),
        "decode_tps": round(1000 * pred_n / pred_ms, 2) if pred_ms >= 1 else 0.0,
        "content_sha16": hashlib.sha256(content.encode()).hexdigest()[:16],
        "content_head": content[:80],
        "proc_read_mb": round((read_proc_io(pid, "read_bytes") - io_r0) / 1e6, 1),
        "proc_write_mb": round((read_proc_io(pid, "write_bytes") - io_w0) / 1e6, 1),
        "sys_read_mb": round((sys_r1 - sys_r0) * 512 / 1e6, 1),
        "sys_write_mb": round((sys_w1 - sys_w0) * 512 / 1e6, 1),
    }


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------

def collect_provenance() -> dict:
    wiki_bytes = WIKI.read_bytes() if WIKI.is_file() else b""
    upstream = {}
    try:
        tree = json.loads(MODEL_TREE.read_text(encoding="utf-8"))
        for name, entry in tree.get("files", {}).items():
            if "Q2_0" in name and entry.get("lfs_sha256"):
                upstream[name] = entry["lfs_sha256"]
    except Exception:  # noqa: BLE001
        pass
    shards = sorted(MODEL.parent.glob("Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-*.gguf"))
    model = {
        "path": str(MODEL),
        "size_bytes": MODEL.stat().st_size if MODEL.is_file() else 0,
        "mtime": int(MODEL.stat().st_mtime) if MODEL.is_file() else 0,
        "shards": [
            {"name": p.name, "size_bytes": p.stat().st_size, "mtime": int(p.stat().st_mtime)}
            for p in shards
        ],
        "upstream_lfs_sha256": upstream,
    }
    box = {
        "hostname": os.uname().nodename,
        "kernel": os.uname().release,
        "cpu": run_cmd(["bash", "-lc", "lscpu | grep -m1 'Model name' | cut -d: -f2- | xargs"]),
        "mem_total_kb": parse_meminfo().get("MemTotal", 0),
        "rocm": run_cmd(["bash", "-lc", "cat /opt/rocm/.info/version 2>/dev/null || echo unknown"]),
        "rocminfo": run_cmd(["/opt/rocm/bin/rocminfo"], timeout=60)[:4000],
        "vram_used_kb_before": {f"card{i}": vram_used_kb(i) for i in range(3)},
        "meminfo_before": parse_meminfo(),
    }
    engine = {
        "repo": str(PRISM),
        "binary": str(SERVER),
        "binary_sha256": sha256_file(SERVER.resolve(), RESULTS / ".engine-sha-cache.json")
        if SERVER.is_file() else "",
        "version": run_cmd([str(SERVER), "--version"]).splitlines()[0]
        if SERVER.is_file() else "",
        **git_state(PRISM),
    }
    prompts = {
        "source": str(WIKI),
        "source_sha256": hashlib.sha256(wiki_bytes).hexdigest(),
        "source_bytes": len(wiki_bytes),
        "slices": {k: v["chars"] for k, v in PROMPTS.items()},
    }
    return {
        "run_kind": "e4-baseline",
        "created": now_iso(),
        "hivebench": {**git_state(REPO)},
        "engine": engine,
        "model": model,
        "box": box,
        "prompts": prompts,
        "protocol": "docs/ENGINE.md E4 baseline protocol (2026-10-02)",
        "flags": {
            "common": COMMON_ARGS,
            "env": {"HIP_VISIBLE_DEVICES": "1,0", "LLAMA_MOE_SLOT_STATS": "1"},
        },
        "caveats": [
            "single sample per prompt; decode noise ~+-1 t/s on this box",
            "VRAM/RAM peaks are sampled at 10 Hz, not instantaneous",
            "disk MB via /proc/<pid>/io read_bytes (includes mmap readahead); "
            "sys_read_mb is the whole-NVMe cross-check",
            "-msc auto picks the slot count from free VRAM at load time",
            "page-cache state differs across arms; cold prefill is first-request",
            "desktop session (Xwayland on card1) active during the run",
        ],
    }


# --------------------------------------------------------------------------
# arm runner
# --------------------------------------------------------------------------

def run_arm(arm: dict, run_dir: Path, port: int, min_free_kb: int,
            load_timeout: int, dry: bool, skip_phases: set[str] | None = None) -> dict:
    log_path = run_dir / "server.log"
    skip_phases = skip_phases or set()
    record: dict = {
        "arm": arm["id"],
        "label": arm["label"],
        "args": [*COMMON_ARGS, *arm["args"]],
        "started": now_iso(),
        "requests": {},
        "skipped_phases": sorted(skip_phases),
    }
    print(f"\n=== arm {arm['id']} ({arm['label']}) ===", flush=True)
    if dry:
        print(" ".join([str(SERVER), "-m", str(MODEL), *COMMON_ARGS, *arm["args"],
                        "--host", "127.0.0.1", "--port", str(port)]))
        return record

    proc = start_server(arm, port, log_path)
    sampler = Sampler(proc.pid)
    sampler.start()
    aborted = None
    try:
        ready = wait_ready(proc, log_path, load_timeout, min_free_kb)
        record["load"] = ready
        print(f"load: {ready}", flush=True)

        prompts = {
            name: WIKI.read_text(encoding="utf-8", errors="replace")[: spec["chars"]]
            for name, spec in PROMPTS.items()
        }
        if ready.get("ok"):
            for name, spec in PROMPTS.items():
                if name in skip_phases:
                    print(f"[{name}] skipped", flush=True)
                    continue
                if parse_meminfo().get("MemAvailable", 1 << 40) < min_free_kb:
                    aborted = f"free RAM below floor before {name}"
                    break
                sampler.phase = name
                print(f"[{name}] n_predict={spec['n_predict']} cache_prompt={spec['cache_prompt']} ...",
                      flush=True)
                out = completion(port, prompts[name], spec["n_predict"],
                                 spec["cache_prompt"], proc.pid,
                                 ignore_eos=bool(spec.get("ignore_eos")))
                record["requests"][name] = out
                if not out.get("ok"):
                    aborted = f"request {name} failed: {out.get('error')}"
                    break
                print(
                    f"[{name}] pp={out['prompt_n']}t {out['prefill_tps']} t/s | "
                    f"tg={out['predicted_n']}t {out['decode_tps']} t/s | "
                    f"wall {out['wall_s']}s disk {out['proc_read_mb']}MB",
                    flush=True,
                )
        else:
            aborted = ready.get("error")

        sampler.phase = "shutdown"
        record["vmhwm_kb"] = read_proc_status(proc.pid, "VmHWM")
        record["vmsize_kb"] = read_proc_status(proc.pid, "VmSize")
    except Exception as exc:  # noqa: BLE001 - partial runs still produce artifacts
        aborted = aborted or f"{type(exc).__name__}: {exc}"
        print(f"arm {arm['id']} error: {aborted}", file=sys.stderr, flush=True)
    finally:
        if proc.poll() is None:
            record["stop"] = stop_server(proc)
        time.sleep(1.0)
        sampler.stop()
        sampler.join(timeout=5)
        record["stop"] = record.get("stop") or {
            "exited": proc.poll() is not None, "rc": proc.returncode
        }
        record["slot_stats"] = parse_slot_stats(log_path)
        record["aborted"] = aborted
        record["ended"] = now_iso()

        # per-phase peaks and I/O deltas
        record["phases"] = {
            name: phase_peaks(sampler.rows, name)
            for name in ["load", *PROMPTS.keys(), "shutdown"]
        }
        record["mem_available_min_kb"] = min(
            (r["mem_available_kb"] for r in sampler.rows), default=0
        )
        record["vram_peak_kb"] = {
            f"card{i}": max((r[f"vram{i}_kb"] for r in sampler.rows), default=0)
            for i in range(3)
        }

        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "samples.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(sampler.rows[0].keys()) if sampler.rows else []
            )
            if sampler.rows:
                writer.writeheader()
                writer.writerows(sampler.rows)
        (run_dir / "manifest.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(f"slot stats: {record['slot_stats']}", flush=True)
    print(f"artifact: {run_dir / 'manifest.json'}", flush=True)
    return record


# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------

def _tps(record: dict, name: str, key: str) -> float | None:
    req = record.get("requests", {}).get(name)
    return req.get(key) if req and req.get("ok") else None


def write_summary(run_dir: Path, provenance: dict, records: list[dict]) -> None:
    summary = {
        "run": run_dir.name,
        "created": provenance["created"],
        "engine_commit": provenance["engine"]["commit"],
        "model_path": provenance["model"]["path"],
        "model_sha256": provenance["model"].get("sha256"),
        "arms": records,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    lines = [
        f"# E4 baseline - Flash-Next Q2_0 ({run_dir.name})",
        "",
        f"Engine: `prism {provenance['engine']['branch']}` @ `{provenance['engine']['commit'][:12]}` "
        f"({provenance['engine']['version']})",
        f"Model: `{provenance['model']['path']}`",
        f"Model sha256: `{provenance['model'].get('sha256') or 'pending'}`",
        f"Protocol: {provenance['protocol']}",
        "",
        "| arm | slots | hit rate | cold pp 1K | cold pp 4K | warm pp 1K | decode1 | decode2 | cached | disk MB/gen | RAM min GB | VRAM pk c0/c1 GB | load s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for rec in records:
        slots = rec.get("slot_stats", {})
        pp1 = _tps(rec, "cold1k", "prefill_tps")
        pp4 = _tps(rec, "cold4k", "prefill_tps")
        ppw = _tps(rec, "warm1k", "prefill_tps")
        d1 = _tps(rec, "decode1", "decode_tps")
        d2 = _tps(rec, "decode2", "decode_tps")
        dc = _tps(rec, "cached", "decode_tps")
        disk = _tps(rec, "decode2", "proc_read_mb")
        ram_min = rec.get("mem_available_min_kb")
        vram = rec.get("vram_peak_kb", {})
        lines.append(
            "| {arm} | {slots} | {hr} | {pp1} | {pp4} | {ppw} | {d1} | {d2} | {dc} | {disk} | {ram} | {v0}/{v1} | {load} |".format(
                arm=rec["arm"],
                slots=slots.get("slots_per_layer") or "-",
                hr=f"{slots['hit_rate_pct']:.1f}%" if "hit_rate_pct" in slots else "-",
                pp1=f"{pp1:.0f}" if pp1 else "-",
                pp4=f"{pp4:.0f}" if pp4 else "-",
                ppw=f"{ppw:.0f}" if ppw else "-",
                d1=f"{d1:.1f}" if d1 else "-",
                d2=f"{d2:.1f}" if d2 else "-",
                dc=f"{dc:.1f}" if dc else "-",
                disk=f"{disk:.0f}" if disk is not None else "-",
                ram=f"{ram_min/1e6:.1f}" if ram_min else "-",
                v0=f"{vram.get('card0',0)/1e6:.1f}",
                v1=f"{vram.get('card1',0)/1e6:.1f}",
                load=f"{rec.get('load',{}).get('load_s','-')}",
            )
        )
    lines += ["", "Caveats:"]
    lines += [f"- {c}" for c in provenance["caveats"]]
    for rec in records:
        if rec.get("skipped_phases"):
            lines.append(f"- {rec['arm']}: phases skipped: {', '.join(rec['skipped_phases'])}")
        if rec.get("aborted"):
            lines.append(f"- {rec['arm']}: aborted: {rec['aborted']}")
    lines.append("")
    (run_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")

    index_path = RESULTS / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else []
    keep = [row for row in index if row.get("run") not in {f"{run_dir.name}/{r['arm']}" for r in records}]
    for rec in records:
        keep.append({
            "run": f"{run_dir.name}/{rec['arm']}",
            "kind": "e4-baseline",
            "created": provenance["created"],
            "model_sha256": provenance["model"].get("sha256"),
            "engine_commit": provenance["engine"]["commit"],
            "metrics": {
                "cold_pp_1k_tps": _tps(rec, "cold1k", "prefill_tps"),
                "cold_pp_4k_tps": _tps(rec, "cold4k", "prefill_tps"),
                "warm_pp_1k_tps": _tps(rec, "warm1k", "prefill_tps"),
                "decode2_tps": _tps(rec, "decode2", "decode_tps"),
                "hit_rate_pct": rec.get("slot_stats", {}).get("hit_rate_pct"),
                "disk_mb_per_gen": _tps(rec, "decode2", "proc_read_mb"),
                "vram_peak_gb": {k: round(v / 1e6, 2) for k, v in rec.get("vram_peak_kb", {}).items()},
            },
        })
    index_path.write_text(json.dumps(keep, indent=1), encoding="utf-8")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arms", default="all",
                        help="comma list of arm ids (0, 12, 24 match substrings) or 'all'")
    parser.add_argument("--name", default=f"e4-baseline-{time.strftime('%Y%m%d')}",
                        help="run directory under experiments/cascade/results/")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--min-free-gb", type=float, default=4.0,
                        help="abort when MemAvailable drops below this")
    parser.add_argument("--load-timeout", type=int, default=1800)
    parser.add_argument("--sha", choices=["end", "skip"], default="end",
                        help="hash the model shards after the runs (default) or skip")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the commands and planned requests, run nothing")
    parser.add_argument("--skip-phases", default="",
                        help="comma list of request phases to skip (e.g. cached)")
    args = parser.parse_args()
    skip_phases = {p.strip() for p in args.skip_phases.split(",") if p.strip()}

    if not SERVER.is_file():
        print(f"engine binary not found: {SERVER}", file=sys.stderr)
        return 2
    if not MODEL.is_file():
        print(f"model not found: {MODEL}", file=sys.stderr)
        return 2
    if not WIKI.is_file():
        print(f"prompt source not found: {WIKI}", file=sys.stderr)
        return 2

    if args.arms.strip().lower() == "all":
        selected = list(ARMS)
    else:
        wanted = [w.strip() for w in args.arms.split(",") if w.strip()]
        selected = [a for a in ARMS if any(w in a["id"] for w in wanted)]
    if not selected:
        print(f"no arms match {args.arms!r}; ids: {[a['id'] for a in ARMS]}", file=sys.stderr)
        return 2

    if args.dry_run:
        for arm in selected:
            run_arm(arm, RESULTS / args.name / arm["id"], args.port,
                    int(args.min_free_gb * 1e6), args.load_timeout, dry=True,
                    skip_phases=skip_phases)
        print("\nrequests per arm:")
        for name, spec in PROMPTS.items():
            print(f"  {name:8} {spec['chars']} chars, n_predict={spec['n_predict']}, cache_prompt={spec['cache_prompt']}")
        return 0

    provenance = collect_provenance()
    run_dir = RESULTS / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for arm in selected:
        records.append(
            run_arm(arm, run_dir / arm["id"], args.port,
                    int(args.min_free_gb * 1e6), args.load_timeout, dry=False,
                    skip_phases=skip_phases)
        )
        if records[-1].get("aborted"):
            print(f"arm {arm['id']} aborted: {records[-1]['aborted']}", file=sys.stderr)

    # merge arm manifests from earlier invocations so the summary covers the
    # whole run directory, not only this pass
    order = {arm["id"]: i for i, arm in enumerate(ARMS)}
    merged = {}
    for manifest in sorted(run_dir.glob("*/manifest.json")):
        try:
            rec = json.loads(manifest.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if rec.get("arm"):
            merged[rec["arm"]] = rec
    for rec in records:
        merged[rec["arm"]] = rec
    records = sorted(merged.values(), key=lambda r: order.get(r["arm"], 99))

    if args.sha == "end":
        shards = sorted(MODEL.parent.glob("Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-*.gguf"))
        for shard in shards:
            print(f"hashing {shard.name} ...", flush=True)
            provenance["model"].setdefault("local_sha256", {})[shard.name] = sha256_file(
                shard, RESULTS / ".model-sha-cache.json"
            )
        provenance["model"]["sha256"] = provenance["model"]["local_sha256"].get(MODEL.name, "")

    (run_dir / "provenance.json").write_text(json.dumps(provenance, indent=1), encoding="utf-8")
    write_summary(run_dir, provenance, records)
    print(f"\nsummary: {run_dir / 'SUMMARY.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
