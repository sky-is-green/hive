#!/usr/bin/env python
"""Strata feasibility bench: mirror the E4 baseline sequence via the OpenAI API.

Runs against a running Strata server (default http://127.0.0.1:8080) with its
engine log, sending the same wiki prompts and request order as
``bench_e4_baseline.py`` and parsing Strata's engine timing line:

    prompt N tokens = R reused + F read in X ms (P tok/s), G generated in Y ms (D tok/s)

Records system RAM/VRAM/GPU/disk at 10 Hz plus ``/metrics`` draft totals per
request. Artifacts under ``experiments/cascade/results/<name>/``.

Usage:

    .venv/bin/python experiments/cascade/bench_strata.py \
        --engine-log ~/Desktop/work/strata/strata-q2_0.log \
        --name strata-feasibility-20261002
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from experiments.cascade.bench_e4_baseline import (  # noqa: E402
    MODEL,
    PROMPTS,
    RESULTS,
    WIKI,
    Sampler,
    disk_sectors,
    git_state,
    now_iso,
    parse_meminfo,
    sha256_file,
)

STRATA_REPO = Path.home() / "Desktop/work/strata"
STRATA_DATA = Path.home() / "Desktop/work/strata-data"

PATTERN = re.compile(
    r"prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([0-9.]+) tok/s\), "
    r"(\d+) generated in (\d+) ms \(([0-9.]+) tok/s\)"
)
FIELDS = (
    "prompt_tokens", "reused", "fresh", "prefill_ms", "prefill_tps",
    "generated", "decode_ms", "decode_tps",
)


def get_json(url: str, timeout: float = 30) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.load(resp)


def read_timing(log_path: Path, offset: int, deadline_s: float = 3.0) -> dict | None:
    deadline = time.monotonic() + deadline_s
    while True:
        try:
            with log_path.open("rb") as handle:
                handle.seek(offset)
                matches = PATTERN.findall(handle.read().decode(errors="replace"))
        except OSError:
            matches = []
        if matches:
            return dict(zip(FIELDS, (float(v) for v in matches[-1])))
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.05)


def completion(base: str, model: str, prompt: str, max_tokens: int,
               timeout: float = 900) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_k": 1,
        "top_p": 1,
        "min_p": 0,
        "seed": 1234,
        "reasoning_effort": "none",
    }
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)[:300], "wall_s": round(time.monotonic() - t0, 2)}
    wall = time.monotonic() - t0
    choice = (data.get("choices") or [{}])[0]
    usage = data.get("usage") or {}
    return {
        "ok": True,
        "wall_s": round(wall, 2),
        "finish_reason": choice.get("finish_reason"),
        "usage": usage,
        "content_head": str((choice.get("message") or {}).get("content") or "")[:80],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="strata")
    parser.add_argument("--engine-log", type=Path, required=True)
    parser.add_argument("--name", default=f"strata-feasibility-{time.strftime('%Y%m%d')}")
    parser.add_argument("--repo", type=Path, default=STRATA_REPO)
    parser.add_argument("--data-dir", type=Path, default=STRATA_DATA)
    parser.add_argument("--skip-phases", default="")
    parser.add_argument("--flags", default="",
                        help="the engine args the server was started with (provenance)")
    args = parser.parse_args()
    skip = {p.strip() for p in args.skip_phases.split(",") if p.strip()}

    run_dir = RESULTS / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    sampler = Sampler(pid=0)  # system-wide; the engine runs under the serve process
    sampler.start()
    sampler.phase = "idle"
    time.sleep(0.5)

    wiki = WIKI.read_text(encoding="utf-8", errors="replace")
    prompts = {name: wiki[: spec["chars"]] for name, spec in PROMPTS.items()}
    record: dict = {
        "run": args.name,
        "kind": "strata-feasibility",
        "started": now_iso(),
        "base": args.base,
        "model": args.model,
        "engine_log": str(args.engine_log),
        "requests": {},
        "skipped_phases": sorted(skip),
    }

    for name, spec in PROMPTS.items():
        if name in skip:
            print(f"[{name}] skipped", flush=True)
            continue
        sampler.phase = name
        offset = args.engine_log.stat().st_size if args.engine_log.is_file() else 0
        sys_r0, sys_w0 = disk_sectors()
        print(f"[{name}] max_tokens={spec['n_predict']} ...", flush=True)
        out = completion(args.base, args.model, prompts[name], spec["n_predict"])
        sys_r1, sys_w1 = disk_sectors()
        out["sys_read_mb"] = round((sys_r1 - sys_r0) * 512 / 1e6, 1)
        out["sys_write_mb"] = round((sys_w1 - sys_w0) * 512 / 1e6, 1)
        timing = read_timing(args.engine_log, offset)
        if timing:
            out["timing"] = timing
        try:
            metrics = get_json(args.base.rstrip("/") + "/metrics", timeout=30)
            totals = metrics.get("totals") or {}
            out["metrics_totals"] = {
                k: totals.get(k) for k in ("requests", "drafts_offered", "drafts_accepted")
            }
            hist = metrics.get("requests") or []
            if hist:
                out["metrics_last"] = {k: hist[0].get(k) for k in
                                       ("prompt_tokens", "generated", "prefill_tps", "decode_tps",
                                        "drafts_offered", "drafts_accepted")}
        except Exception as exc:  # noqa: BLE001
            out["metrics_error"] = str(exc)[:200]
        record["requests"][name] = out
        if out.get("ok"):
            t = out.get("timing") or {}
            print(
                f"[{name}] wall {out['wall_s']}s | pp {t.get('prompt_tokens','?')}t "
                f"({t.get('prefill_tps','?')} t/s, {t.get('reused','?')} reused) | "
                f"tg {t.get('generated','?')}t ({t.get('decode_tps','?')} t/s) | "
                f"finish {out.get('finish_reason')}",
                flush=True,
            )
        else:
            print(f"[{name}] FAILED: {out.get('error')}", flush=True)
            break

    sampler.phase = "idle"
    time.sleep(0.5)
    sampler.stop()
    sampler.join(timeout=5)

    record["mem_available_min_kb"] = min((r["mem_available_kb"] for r in sampler.rows), default=0)
    record["vram_peak_kb"] = {
        f"card{i}": max((r[f"vram{i}_kb"] for r in sampler.rows), default=0) for i in range(3)
    }
    record["phases"] = {}
    for phase in ["idle", *PROMPTS.keys()]:
        sel = [r for r in sampler.rows if r["phase"] == phase]
        if sel:
            record["phases"][phase] = {
                "samples": len(sel),
                "mem_available_min_kb": min(r["mem_available_kb"] for r in sel),
                "vram_peak_kb": {f"card{i}": max(r[f"vram{i}_kb"] for r in sel) for i in range(3)},
                "gpu_busy_mean": {f"card{i}": round(sum(max(r[f"gpu{i}_busy"], 0) for r in sel) / len(sel), 1)
                                  for i in range(2)},
            }

    engine_bins = sorted(args.repo.glob("build-hip/strata")) + sorted(args.repo.glob("engine/strata"))
    provenance = {
        "created": now_iso(),
        "strata": {
            "repo": str(args.repo),
            **git_state(args.repo),
            "engine_binary": str(engine_bins[0]) if engine_bins else "",
            "engine_binary_sha256": sha256_file(engine_bins[0], RESULTS / ".engine-sha-cache.json")
            if engine_bins and engine_bins[0].is_file() else "",
            "data_dir": str(args.data_dir),
            "packs": [str(p) for p in sorted((args.data_dir / "packs").glob("*"))]
            if (args.data_dir / "packs").is_dir() else [],
        },
        "model": {
            "source_gguf": str(MODEL),
            "sha256": sha256_file(MODEL, RESULTS / ".model-sha-cache.json") if MODEL.is_file() else "",
        },
        "flags": args.flags,
        "box": {
            "mem_total_kb": parse_meminfo().get("MemTotal", 0),
            "meminfo_before": parse_meminfo(),
        },
        "caveats": [
            "single sample per prompt",
            "Strata's engine log timings are authoritative; API wall time includes HTTP",
            "no ignore_eos on the API - decode may stop at EOS before max_tokens",
            "system-wide 10 Hz sampling (engine runs under the serve process)",
        ],
    }

    (run_dir / "manifest.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    (run_dir / "provenance.json").write_text(json.dumps(provenance, indent=1), encoding="utf-8")
    if sampler.rows:
        with (run_dir / "samples.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(sampler.rows[0].keys()))
            writer.writeheader()
            writer.writerows(sampler.rows)

    lines = [
        f"# Strata feasibility - {args.name}",
        "",
        f"Engine: `{provenance['strata']['repo']}` @ `{provenance['strata']['commit'][:12]}`",
        f"Model: `{provenance['model']['source_gguf']}` sha `{provenance['model']['sha256'][:16]}…`",
        f"Flags: `{args.flags}`",
        "",
        "| phase | wall s | prompt t | reused | prefill t/s | generated | decode t/s | finish |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, out in record["requests"].items():
        t = out.get("timing") or {}
        lines.append(
            "| {n} | {w} | {p} | {r} | {pp} | {g} | {d} | {f} |".format(
                n=name, w=out.get("wall_s", "?"),
                p=t.get("prompt_tokens", "?"), r=t.get("reused", "?"),
                pp=t.get("prefill_tps", "?"), g=t.get("generated", "?"),
                d=t.get("decode_tps", "?"), f=out.get("finish_reason", "-"),
            )
        )
    lines += [
        "",
        f"RAM available min: {record['mem_available_min_kb'] / 1e6:.1f} GB",
        f"VRAM peak: card0 {record['vram_peak_kb']['card0'] / 1e6:.1f} GB, "
        f"card1 {record['vram_peak_kb']['card1'] / 1e6:.1f} GB",
        "",
        "Caveats:",
    ]
    lines += [f"- {c}" for c in provenance["caveats"]]
    (run_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nsummary: {run_dir / 'SUMMARY.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
