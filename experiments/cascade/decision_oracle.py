#!/usr/bin/env python3
"""Decision oracle stage 0 — C2/D2 escalation judges, CPU beside the 176B.

Measures the four decision models on disk as the cascade's accept/reject
(D2, feeding the C2 "answer / escalate / refuse" gate):

    tiny-jev-1.7b         lostargon/Tiny-Jev-1.7B
    intern-decision-0.8b  internlm/Intern-Decision-0.8B
    intern-decision-2b    internlm/Intern-Decision-2B
    intern-decision-4b    internlm/Intern-Decision-4B

Protocol: the frozen J-Bench from the jev-readout session
(``results/jev-readout-20261006/judge-bench.jsonl``), **test split** (n=116;
29 correct / 87 incorrect; recorded + perturbed items), scored by the existing
``score_judges.py`` wrappers through ``experiments/cascade/decision.py``.  The
per-model numbers (AUC, accuracy @0.5, false accepts / rejects, no-leak
threshold) are directly comparable with the July readout runs — those ran the
intern models on GPU; this run measures the residency that matters now: CPU,
with the 176B live on 8081.

The driver runs one model per subprocess (RAM freed between), samples host
MemAvailable, polls each child's VmHWM, probes the serve health before/after,
then aggregates CandidateMeasurement rows (role D2, device cpu), the frontier,
and a SUMMARY.

Usage, from the repo root:

    .venv/bin/python experiments/cascade/decision_oracle.py run          # all four
    .venv/bin/python experiments/cascade/decision_oracle.py run --limit 4  # smoke
    .venv/bin/python experiments/cascade/decision_oracle.py aggregate

CPU-only: a ROCm context beside the live serve is the documented hard-reset
hazard; the earlier GPU readout numbers stay the GPU reference point.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CASCADE_DIR = REPO_ROOT / "experiments" / "cascade"
RESULTS_DIR = CASCADE_DIR / "results" / "decision-oracle-20261017"
FRONTIER_JSON = CASCADE_DIR / "decision-oracle-20261017.json"
SCORE_SCRIPT = CASCADE_DIR / "score_judges.py"
BENCH_NAME = "judge-bench.jsonl (test split)"

HEALTH_URL = "http://127.0.0.1:8081/health"

#: (family for score_judges.py, model id, registry id)
SPECS = (
    ("tinyjev", "lostargon/Tiny-Jev-1.7B", "tiny-jev-1.7b"),
    ("intern", "internlm/Intern-Decision-0.8B", "intern-decision-0.8b"),
    ("intern", "internlm/Intern-Decision-2B", "intern-decision-2b"),
    ("intern", "internlm/Intern-Decision-4B", "intern-decision-4b"),
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _probe_health() -> dict:
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=3) as resp:
            return {"url": HEALTH_URL, "status": resp.status}
    except Exception as exc:  # noqa: BLE001 — the probe is informational
        return {"url": HEALTH_URL, "status": None, "error": str(exc)}


def _meminfo_available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1048576.0
    return float("nan")


def _proc_vmhwm_gib(pid: int) -> float:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1048576.0
    except OSError:
        pass
    return 0.0


def weights_gb(model_id: str) -> float:
    """Checkpoint weight bytes on disk (safetensors/bin), in GB."""
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(model_id))
    total = 0
    for pattern in ("*.safetensors", "*.bin"):
        total += sum(p.stat().st_size for p in snapshot.glob(pattern))
    return round(total / 1e9, 2)


def _judge_json(name: str) -> Path:
    return RESULTS_DIR / f"judge-{name}.json"


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------

def run_one(family: str, model_id: str, name: str, limit: int | None) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS_DIR / f"{name}.log"
    cmd = [
        sys.executable, str(SCORE_SCRIPT),
        "--family", family, "--model", model_id, "--name", name,
        "--device", "cpu", "--split", "test", "--out", str(RESULTS_DIR),
    ]
    if limit:
        cmd += ["--limit", str(limit)]
    print(f"[decision-oracle] {name} <- {model_id}", flush=True)

    peak_rss = 0.0
    with log_path.open("w") as log:
        proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=log,
                                stderr=subprocess.STDOUT)
        while proc.poll() is None:
            peak_rss = max(peak_rss, _proc_vmhwm_gib(proc.pid))
            time.sleep(1.5)
    if proc.returncode != 0:
        tail = "\n".join(log_path.read_text().splitlines()[-8:])
        raise SystemExit(f"[decision-oracle] {name} failed; log tail:\n{tail}")
    return {"peak_rss_gib": round(peak_rss, 2)}


def _run_all(limit: int | None) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    health_before = _probe_health()
    print(f"[decision-oracle] serve health before: {health_before}", flush=True)

    samples: list[tuple[float, float]] = []
    stop = threading.Event()

    def memwatch() -> None:
        t_start = time.time()
        while not stop.is_set():
            samples.append((time.time() - t_start, _meminfo_available_gib()))
            stop.wait(2.0)

    watcher = threading.Thread(target=memwatch, daemon=True)
    watcher.start()
    t0 = time.time()

    records: dict[str, dict] = {}
    for family, model_id, name in SPECS:
        records[name] = run_one(family, model_id, name, limit)

    elapsed = time.time() - t0
    stop.set()
    watcher.join(timeout=5)
    health_after = _probe_health()
    print(f"[decision-oracle] serve health after: {health_after}", flush=True)

    if samples:
        min_avail = min(v for _, v in samples)
        (RESULTS_DIR / "memwatch.log").write_text(
            f"# MemAvailable during the decision oracle run ({RESULTS_DIR.name})\n"
            f"# min {min_avail:.2f} GiB | serve {HEALTH_URL} before="
            f"{health_before['status']} after={health_after['status']}\n"
            + "".join(f"{t:8.1f}s  MemAvailable={v:.2f} GiB\n" for t, v in samples)
        )

    meta = {
        "role": "D2 (accept/reject; feeds C2 answer/escalate/refuse gating)",
        "date": time.strftime("%Y-%m-%d", time.gmtime()),
        "device": "cpu",
        "bench": BENCH_NAME,
        "split": "test",
        "limit": limit,
        "run_elapsed_s": round(elapsed, 1),
        "serve": {"before": health_before, "after": health_after},
        "mem_available_min_gib": round(min((v for _, v in samples), default=float("nan")), 2),
        "peak_rss_gib": {name: rec.get("peak_rss_gib") for name, rec in records.items()},
    }
    aggregate(meta)


# --------------------------------------------------------------------------
# aggregate
# --------------------------------------------------------------------------

def _notes(name: str, metrics: dict, lat_ms: list[float], peak_rss: float | None) -> str:
    at = metrics["at_0.5"]
    nl = metrics.get("no_leak") or {}
    nl_txt = (f", no-leak thr {nl['threshold']:.2f} -> "
              f"{nl['correct']}/{nl['n']}" if nl else "")
    lat_med = statistics.median(lat_ms)
    rss = f"; peak RSS {peak_rss} GiB" if peak_rss else ""
    return (
        f"CPU bf16, J-Bench test n={metrics['n']} (29 correct/87 wrong): AUC "
        f"{metrics['auc']:.3f}, acc@0.5 {at['correct']}/{metrics['n']}, "
        f"FA {at['false_accepts']}, FR {at['false_rejects']}{nl_txt}; "
        f"median {lat_med:.0f} ms/verdict{rss}; via harness validator "
        f"(threshold 0.5) as the C2 escalation gate"
    )


def aggregate(meta: dict | None = None) -> None:
    sys.path.insert(0, str(REPO_ROOT))
    from harness.cascade.oracle import CandidateMeasurement, frontier_by_role
    from harness.cascade.roles import Device

    recs: dict[str, dict] = {}
    rows = []
    for _family, model_id, name in SPECS:
        path = _judge_json(name)
        if not path.exists():
            print(f"[decision-oracle] no record for {name} — skipped")
            continue
        data = json.loads(path.read_text())
        recs[name] = data
        metrics = data["metrics"]
        lat = [float(i["judge_ms"]) for i in data["items"]]
        peak = (meta or {}).get("peak_rss_gib", {}).get(name)
        rows.append(CandidateMeasurement(
            candidate=name,
            role="D2",
            device=Device.CPU,
            quality=round(float(metrics["auc"]), 4),
            latency_ms=round(statistics.median(lat), 1),
            unit="call",
            ctx=0,
            precision="bf16",
            bytes_gb=weights_gb(model_id),
            notes=_notes(name, metrics, lat, peak),
        ))

    payload = {"measurements": [row.to_dict() for row in rows], "meta": meta or {}}
    (RESULTS_DIR / "measurements.json").write_text(json.dumps(payload, indent=2) + "\n")
    FRONTIER_JSON.write_text(json.dumps(payload, indent=2) + "\n")

    frontiers = frontier_by_role(rows)
    frontier_rows = {role: [r.to_dict() for r in group]
                     for role, group in frontiers.items()}
    on_front = {row["candidate"] for row in frontier_rows.get("D2", [])}

    lines = [
        "# Decision oracle — stage 0 (C2/D2 escalation judges)",
        "",
        "Measured on CPU beside the resident 176B serve (port 8081), 2026-10-17.",
        "Frozen J-Bench test split (n=116: 29 correct / 87 incorrect answers,",
        "recorded + perturbed). AUC/accuracy are directly comparable with the",
        "GPU readout runs in `results/jev-readout-20261006/`.",
        "",
        "| model | bytes | AUC | acc@0.5 | FA | FR | no-leak thr | median ms | peak RSS |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        m = recs[row.candidate]["metrics"]
        at = m["at_0.5"]
        nl = m.get("no_leak") or {}
        nl_txt = f"{nl['threshold']:.2f} ({nl['correct']}/{nl['n']})" if nl else "-"
        peak = (meta or {}).get("peak_rss_gib", {}).get(row.candidate)
        lines.append(
            f"| {row.candidate} | {row.to_dict()['bytes_gb']} GB | {m['auc']:.3f} "
            f"| {at['correct']}/{m['n']} | {at['false_accepts']} | {at['false_rejects']} "
            f"| {nl_txt} | {row.latency_ms:.0f} ms | {peak or '-'} GiB |"
        )

    lines += ["", "## Frontier (role=D2, unit=call)", ""]
    for role, group in frontier_rows.items():
        lines.append(f"Non-dominated rows for `{role}`:")
        lines.append("")
        for r in group:
            lines.append(f"- **{r['candidate']}** — AUC {r['quality']}, "
                         f"{r['latency_ms']} ms/verdict, {r['bytes_gb']} GB")
        lines.append("")
    dominated = [r.candidate for r in rows if r.candidate not in on_front]
    if dominated:
        lines.append("Dominated (kept for the record): " + ", ".join(dominated) + ".")
        lines.append("")

    if meta:
        lines += [
            "## Coexistence with the live 176B serve",
            "",
            f"- `/health` before: {meta['serve']['before']['status']}, after: "
            f"{meta['serve']['after']['status']} (serve stayed up).",
            f"- MemAvailable across the run: min {meta['mem_available_min_gib']} GiB "
            f"(see `memwatch.log`); total run {meta['run_elapsed_s']} s.",
            "- All four ran CPU-only (no ROCm context beside the serve). The July",
            "  readout runs scored the intern models on GPU; those numbers remain",
            "  the GPU reference.",
            "",
        ]
    lines += [
        "## Artifacts",
        "",
        "- `judge-<name>.json` — per-item verdicts + metrics (`score_judges.py` schema).",
        "- `<name>.log` — per-model run logs.",
        "- `measurements.json` — CandidateMeasurement rows (also copied to",
        "  `experiments/cascade/decision-oracle-20261017.json` for `/v1/cascade/frontier`).",
        "- `memwatch.log` — host MemAvailable samples.",
        "",
    ]
    (RESULTS_DIR / "SUMMARY.md").write_text("\n".join(lines))
    print(f"[decision-oracle] wrote {RESULTS_DIR/'measurements.json'}")
    print(f"[decision-oracle] wrote {RESULTS_DIR/'SUMMARY.md'}")
    print(f"[decision-oracle] wrote {FRONTIER_JSON}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=["run", "aggregate"])
    parser.add_argument("--limit", type=int, default=0,
                        help="items per model (smoke); 0 = full test split")
    args = parser.parse_args()

    if args.action == "run":
        _run_all(args.limit or None)
    else:
        aggregate()


if __name__ == "__main__":
    main()
