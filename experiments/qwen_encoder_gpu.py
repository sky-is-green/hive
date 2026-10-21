#!/usr/bin/env python
"""The gate's *quality* numbers for our student, on the GPU.

The canonical gate (``experiments/qwen_encoder.py --stage measure``) runs
every candidate through the suite's own CPU path, because the cost is
part of the measurement and the drone is a CPU citizen.  That makes the
Qwen teacher's arm slow (7,150 texts for the drift cases, one at a
time).

This is the same measurement with the *quality* part on the GPU: the
suite's own ``measure_dedup`` / ``measure_drift`` / ``measure_cost`` are
called unchanged, and only the embedding function differs (the student's
own path, on the GPU).  The numbers are the same arithmetic on the same
vectors; the cost is not comparable across devices, so the cost line
here is the CPU one from the canonical run.

Usage::

    ../.venv-rocm/bin/python -m experiments.qwen_encoder_gpu --out DIR
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.qwen_encoder import (build_pairs, drop_degenerate,  # noqa: E402
                                      split_conversations, student_encoder)


def gpu_encoder(path: str, device: str = "cuda", batch: int = 32):
    """The student's embedding function, on the GPU."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(path, dtype=torch.bfloat16).to(device).eval()
    step = max(1, int(batch))

    def embed(texts: list[str]):
        # the suite's embed_many hands over every text in one call, and the
        # drift cases are 7,150 of them - so the batch is split here (the
        # vectors are the same; only the memory differs)
        import numpy as np

        empty = [i for i, t in enumerate(texts) if not (t or "").strip()]
        safe = [(t if (t or "").strip() else " ") for t in texts]
        out = []
        for i in range(0, len(safe), step):
            chunk = tokenizer(safe[i:i + step], padding=True, truncation=True,
                              max_length=192, return_tensors="pt")
            chunk = {k: v.to(device) for k, v in chunk.items()}
            with torch.no_grad():
                h = model(**chunk).last_hidden_state
                lengths = chunk["attention_mask"].sum(dim=1) - 1
                picked = h[torch.arange(h.size(0), device=h.device), lengths]
                out.append(F.normalize(picked.float(), dim=-1).cpu().numpy())
        vec = np.concatenate(out, axis=0) if out else np.zeros((0, 1), dtype="float32")
        for i in empty:
            vec[i] = 0.0
        return vec

    info = {"the_model": path, "the_pooling": "last token", "the_normalised": True,
            "the_device": device,
            "the_params": sum(p.numel() for p in model.parameters())}
    return embed, info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", nargs="+",
                    default=[str(REPO / "harness_state"),
                             str(REPO.parent / "splinter-memory" / "harness_state")])
    ap.add_argument("--per-conv", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--out", type=Path,
                    default=REPO / "experiments/encoder-suite/results/qwen")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    from experiments.encoder_suite import (build_drift_cases, load_conversations,
                                           measure_dedup, measure_drift)

    conversations = load_conversations(args.corpus)
    _train, holdout = split_conversations(conversations, seed=args.seed)
    pairs = build_pairs(holdout, args.per_conv, args.seed)
    drift_cases = build_drift_cases(holdout)
    print(f"holdout: {len(pairs)} pairs, {len(drift_cases)} drift cases", flush=True)

    checkpoint = args.out / "qwen-bi-encoder"
    embed, info = gpu_encoder(str(checkpoint), batch=args.batch)
    t0 = time.time()
    dedup = measure_dedup(pairs, embed)
    drift = measure_drift(drift_cases, embed)
    print(f"our student (the GPU): dedup@0.92 f1 {dedup['at_0.92']['f1']} "
          f"(precision {dedup['at_0.92']['precision']}, "
          f"recall {dedup['at_0.92']['recall']}) | "
          f"dedup best {dedup['best']['f1']} | drift auc {drift['auc']} "
          f"({time.time() - t0:.0f}s)", flush=True)
    out = {"the_info": info, "dedup": dedup, "drift": drift}
    (args.out / "qwen-student-gpu.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out / 'qwen-student-gpu.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
