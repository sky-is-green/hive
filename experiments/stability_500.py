"""The 500-turn stability: does the memory survive a long conversation?

The published claim (the canonical paper's §8): over 500+ turns the
memory pipeline holds its throughput, its resident memory stays small and
nothing is OOM-killed.  This re-measures it, on the current code, with
the real drone and a scripted conversation (no model, no network - the
memory pipeline is what is under test, not the generator).

What it reports:

``turns``           how many turns ran
``peak_rss_mb``     the process's peak resident memory
``store_chunks``    how much the store kept (the curation is what bounds it)
``context_tokens``  the assembled context's size, early vs late
``p50_ms`` / ``p95_ms``  the per-turn curation cost
``throughput_drift``  the late turns' cost against the early turns'

Usage::

    python -m experiments.stability_500 --turns 500
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import sys
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

#: A growing conversation: every turn states a *new* fact, and every
#: fifth turn asks about an *old* one - so the store really grows and the
#: retrieval has something to do.  A script that repeated itself would
#: measure the dedup, not the memory.
TOPICS = [
    ("the graphics card", "an RX 7900 XTX with 24 GB of VRAM, in a 340mm slot"),
    ("the power supply", "an 850W unit: the card alone draws 355W"),
    ("the CPU cooler", "a 360mm AIO, or a dual-tower air cooler if the case allows"),
    ("the case fans", "three 140mm intakes at the front, two exhausts at the rear"),
    ("the motherboard", "a B650 board with three M.2 slots and full ATX clearance"),
    ("the memory", "64 GB of DDR5-6000 for the models and the memory work"),
    ("the primary drive", "a 2 TB NVMe for the model weights"),
    ("the secondary drive", "a cheaper SATA SSD for the run bundles and the logs"),
    ("the operating system", "Linux: the local inference tooling is built for it"),
    ("the network card", "a 2.5 GbE board, because the model downloads are large"),
    ("the display", "a 4K panel driven by the second card, not the compute one"),
    ("the UPS", "a 900 VA unit, so a brownout does not take the run with it"),
    ("the thermal paste", "a cheap ceramic paste is fine under a cold plate"),
    ("the case glass", "a tempered side panel, so the airflow is unobstructed"),
    ("the desk", "a standing desk, because a build takes a long evening"),
]


def build_script(turns: int) -> list[tuple[str, str]]:
    """``turns`` of a growing conversation."""
    out: list[tuple[str, str]] = []
    stated: list[tuple[str, str]] = []
    for turn in range(turns):
        if stated and turn % 5 == 4:
            topic, fact = stated[(turn // 5) % len(stated)]
            out.append((f"Remind me what we said about {topic}.",
                        f"You said {topic} is {fact}."))
            continue
        topic, fact = TOPICS[turn % len(TOPICS)]
        detail = f" (revision {turn // len(TOPICS) + 1})"
        out.append((f"Turn {turn + 1}: what are we using for {topic}?",
                    f"For {topic} we settled on {fact}{detail}."))
        stated.append((topic, fact))
    return out


def rss_mb() -> float:
    """The process's resident memory in MB (the peak, on Linux)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB
    return round(peak / 1024.0, 1)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--turns", type=int, default=500)
    parser.add_argument("--encoder",
                        default="sentence-transformers/paraphrase-MiniLM-L3-v2")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    from focal.budget import AdaptiveBudget
    from cortex.routing import DroneRouter, EscalationHandler
    from focal.assembly import ContextAssembler
    from membrane.dedup import ContextDeduplicator
    from membrane.drift import TopicDriftDetector
    from retention.store import ContextStore
    from sieve.ultra_small import UltraSmallDrone

    drone = UltraSmallDrone(model_name=args.encoder, confidence_mode="off",
                            device="cpu")
    drone._ensure_loaded()
    store = ContextStore(embed_fn=drone.embed)
    assembler = ContextAssembler()
    router = DroneRouter()
    budget = AdaptiveBudget()
    escalation = EscalationHandler()
    dedup = ContextDeduplicator()
    drift = TopicDriftDetector()

    costs: list[float] = []
    contexts: list[int] = []
    turns = max(1, args.turns)
    script = build_script(turns)
    t0 = time.perf_counter()
    for turn in range(1, turns + 1):
        query, answer = script[turn - 1]
        started = time.perf_counter()
        assembled = assembler.assemble(
            query=query, current_turn=turn, store=store, router=router,
            ultra_small=drone, medium=None, escalation=escalation, dedup=dedup,
            drift_detector=drift, budget=budget, max_context=8192,
        )
        costs.append((time.perf_counter() - started) * 1000.0)
        contexts.append(int(getattr(assembled, "token_count", 0) or 0))
        store.add_chunk(turn, answer)
        if turn % 50 == 0:
            print(f"  turn {turn}/{turns}  {time.perf_counter() - t0:.0f}s  "
                  f"rss {rss_mb()} MB  store {len(store.chunks)} chunks  "
                  f"ctx {contexts[-1]} tok", flush=True)

    early = costs[: max(1, len(costs) // 5)]
    late = costs[-max(1, len(costs) // 5):]
    result = {
        "turns": len(costs),
        "seconds": round(time.perf_counter() - t0, 1),
        "peak_rss_mb": rss_mb(),
        "store_chunks": len(store.chunks),
        "context_tokens": {
            "early_p50": statistics.median(contexts[: max(1, len(contexts) // 5)]),
            "late_p50": statistics.median(contexts[-max(1, len(contexts) // 5):]),
            "max": max(contexts) if contexts else 0,
        },
        "curation_ms": {
            "p50": round(statistics.median(costs), 2),
            "p95": round(sorted(costs)[int(len(costs) * 0.95)], 2),
            "early_p50": round(statistics.median(early), 2),
            "late_p50": round(statistics.median(late), 2),
        },
        "throughput_drift": round(statistics.median(late)
                                 / max(statistics.median(early), 1e-9), 3),
        "oom": False,
    }
    print(json.dumps(result, indent=1))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
