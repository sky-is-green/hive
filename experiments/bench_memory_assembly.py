"""Benchmark native daemon memory assembly against the cached Python path.

This is the P3-13 reproducer. It fills a store from a generated fixture, warms
both implementations, then measures:

- **warm**: the same query against an unchanged store (the best case for the
  in-process Python cache);
- **evolving**: a fresh query each turn plus one new chunk, so drift text and
  the candidate pool change every call (the realistic case).

Prerequisites: a daemon with native memory and a managed encoder, e.g.::

    HIVE_MEMORY_BACKEND=rust \
    HIVE_ENCODER_BIN=rust/target/release/hive-encoder \
    HIVE_ENCODER_PORT=8198 \
    rust/target/release/hive-daemon --port 8918 --no-sidecar

and the Python environment used for the reference (torch or a prebuilt
sentence-transformers install)::

    PYTHONPATH=splinter/splinter python -m experiments.bench_memory_assembly \
        --memory-url http://127.0.0.1:8918
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import requests

from cortex.routing import RoutingDecision
from focal.assembly import ContextAssembler
from focal.budget import AdaptiveBudget
from membrane.dedup import ContextDeduplicator
from membrane.drift import TopicDriftDetector
from retention.store import ContextStore
from sieve.ultra_small import UltraSmallDrone
from sieve.vocabulary import Vocabulary

REPO = Path(__file__).resolve().parents[1]


class _Router:
    def route(self, _query, _history=None):
        return RoutingDecision("ultra_small", 1.0, "benchmark")


def _median(values: list[float]) -> float:
    return round(statistics.median(values), 2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        default=str(REPO / "tests" / "fixtures" / "generated" / "long_001.json"),
    )
    parser.add_argument("--chunks", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--memory-url", default="http://127.0.0.1:8918")
    parser.add_argument("--conversation", default="bench")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    fixture = json.loads(Path(args.fixture).read_text())
    texts = [
        turn["content"] for turn in fixture["turns"] if turn.get("content")
    ][: args.chunks]
    drone = UltraSmallDrone(confidence_mode="off", vocab_boost=0.15)
    vocab = Vocabulary.load("code", "general")
    store = ContextStore(embed_fn=drone.embed, sanitize=False, ingest_block_prefixes=[])
    for turn, text in enumerate(texts, 1):
        store.add_chunk(turn, text)

    assembler = ContextAssembler(collect_timings=False)

    def python_assembly(query: str) -> str:
        return assembler.assemble(
            query=query,
            current_turn=args.chunks + 1,
            store=store,
            router=_Router(),
            ultra_small=drone,
            medium=None,
            escalation=None,
            dedup=ContextDeduplicator(vocab=vocab, threshold=0.92),
            drift_detector=TopicDriftDetector(embed_fn=drone.embed, threshold=0.6),
            budget=AdaptiveBudget(),
            max_context=8192,
        ).content

    rows = [
        {
            "id": chunk.id,
            "content": chunk.content,
            "turn": chunk.turn,
            "last_referenced_turn": chunk.last_referenced_turn,
        }
        for chunk in store.all_chunks()
    ]
    requests.post(
        f"{args.memory_url}/v1/memory/{args.conversation}/load",
        json={"chunks": rows},
        timeout=60,
    ).raise_for_status()

    def native_assembly(query: str) -> str:
        response = requests.post(
            f"{args.memory_url}/v1/memory/{args.conversation}/assemble",
            json={
                "query": query,
                "current_turn": args.chunks + 1,
                "route": "ultra_small",
                "max_context": 8192,
                "vocab_boost": 0.15,
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["content"]

    query = "what did we decide about the order schema and its migrations?"
    python_assembly(query)
    native_assembly(query)

    def measure(fn, count):
        values = []
        for _ in range(count):
            started = time.perf_counter()
            fn()
            values.append((time.perf_counter() - started) * 1000.0)
        return values

    python_warm = measure(lambda: python_assembly(query), args.repeat)
    native_warm = measure(lambda: native_assembly(query), args.repeat)

    python_evolving: list[float] = []
    native_evolving: list[float] = []
    for index in range(args.repeat):
        evolving_query = (
            f"turn {index}: how does the retention store decay chunk {index} "
            "and what did we decide about migrations?"
        )
        started = time.perf_counter()
        python_assembly(evolving_query)
        python_evolving.append((time.perf_counter() - started) * 1000.0)
        chunk = f"new fixture chunk {index} about retention and the order schema"
        store.add_chunk(args.chunks + 1 + index, chunk)
        requests.post(
            f"{args.memory_url}/v1/memory/{args.conversation}/add",
            json={"turn": args.chunks + 1 + index, "content": chunk},
            timeout=60,
        ).raise_for_status()
        started = time.perf_counter()
        native_assembly(evolving_query)
        native_evolving.append((time.perf_counter() - started) * 1000.0)

    result = {
        "chunks": len(rows),
        "python_warm_ms": _median(python_warm),
        "native_warm_ms": _median(native_warm),
        "warm_ratio": round(_median(native_warm) / _median(python_warm), 2),
        "python_evolving_ms": _median(python_evolving),
        "native_evolving_ms": _median(native_evolving),
        "evolving_ratio": round(
            _median(native_evolving) / _median(python_evolving), 2
        ),
    }
    if args.json:
        print(json.dumps(result))
    else:
        print(
            f"{result['chunks']} chunks · warm: python {result['python_warm_ms']} ms, "
            f"native {result['native_warm_ms']} ms ({result['warm_ratio']}x) · "
            f"evolving: python {result['python_evolving_ms']} ms, "
            f"native {result['native_evolving_ms']} ms "
            f"({result['evolving_ratio']}x)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
