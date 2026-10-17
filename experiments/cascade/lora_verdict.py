#!/usr/bin/env python
"""LoRA fine-tune the Intern-Decision backbone on the verdict objective.

The heads trained so far sit on frozen features; the organic-gap diagnosis says
the residual failures are not represented in those features. This script moves
the features: it trains low-rank adapters so the model's own decision-marker
logits (the exact forward pass the judge uses at inference) separate
checker-verified correct/incorrect candidates.

Training rows use the same request schema as ``decision.InternDecisionJudge``
(``state = {task, candidate_answer}`` + the D2 ``noul`` question), compiled by
the checkpoint's own ``inference`` module, so the prompt is byte-identical to
deployment. The loss is cross-entropy over the ``yes``/``no`` symbol logits at
the decision marker. CPU-only beside the 176B; the vision tower stays frozen
(target modules are the language ``*_proj`` / ``gate|up|down_proj`` linears).

Usage, from the repo root:

    .venv/bin/python experiments/cascade/lora_verdict.py \
        --corpus experiments/cascade/results/decision-harvest-r2-20261017/harvest-bench-r2.jsonl \
        --bench experiments/cascade/results/jev-readout-20261006/judge-bench.jsonl \
        --name intern-decision-0.8b-lora-r2 --epochs 2 --limit 4   # smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.cascade.decision_readout import load_backend  # noqa: E402
from experiments.cascade.readout_judge import metrics  # noqa: E402

DEFAULT_MODEL = "internlm/Intern-Decision-0.8B"
DEFAULT_BENCH = REPO_ROOT / "experiments/cascade/results/jev-readout-20261006/judge-bench.jsonl"
DEFAULT_OUT = REPO_ROOT / "experiments/cascade/results/lora-verdict-20261017"

LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "in_proj_qkv", "in_proj_z", "out_proj",
]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def verdict_request(item: dict) -> dict:
    return {
        "state": {"task": item["prompt"], "candidate_answer": item["candidate"]},
        "questions": {
            "verdict": {
                "type": "noul",
                "instructions": "The candidate answer is correct and complete for the task.",
                "criteria": {
                    "yes": "correct and complete",
                    "no": "incorrect, incomplete, or contradictory",
                },
            }
        },
    }


def symbol_ids(tokenizer) -> tuple[int, int]:
    ids = []
    for symbol in ("yes", "no"):
        enc = tokenizer.encode(symbol, add_special_tokens=False)
        if len(enc) != 1:
            raise SystemExit(f"symbol {symbol!r} is not a single token: {enc}")
        ids.append(enc[0])
    return ids[0], ids[1]


def predict_p_yes(model, backend, inference, torch, items: list[dict],
                  yes_id: int, no_id: int, limit: int | None = None) -> np.ndarray:
    outs = []
    model.eval()
    with torch.inference_mode():
        for item in items[:limit] if limit else items:
            row = inference.validate_request(verdict_request(item))
            _, batch, positions = backend.encode(row)
            batch = {k: v.to(backend.device) for k, v in batch.items()}
            logits = model(**batch, use_cache=False,
                           logits_to_keep=positions.to(backend.device)).logits[0]
            two = logits[0, torch.tensor([yes_id, no_id], device=backend.device)]
            outs.append(float(torch.softmax(two.float(), dim=-1)[0]))
    return np.array(outs)


def report(tag: str, p: np.ndarray, items: list[dict]) -> dict:
    rows = [{"label": int(it["label"]), "p_correct": float(pi)} for it, pi in zip(items, p)]
    m = metrics(rows)
    at = m["at_0.5"]
    wrong = sum(1 for it in items if it["label"] == 0)
    caught = sum(1 for it, pi in zip(items, p) if it["label"] == 0 and pi < 0.5)
    auc = m["auc"]
    auc_s = f"{auc:.3f}" if auc is not None else "n/a  "
    print(f"  {tag:22s} n {m['n']:4d} AUC {auc_s}  acc {at['correct']:3d}/{m['n']}  "
          f"FA {at['false_accepts']:2d} FR {at['false_rejects']:2d}  "
          f"wrongs-caught {caught}/{wrong}", flush=True)
    return {"n": m["n"], "auc": (round(auc, 4) if auc is not None else None),
            "acc": at["correct"], "fa": at["false_accepts"],
            "fr": at["false_rejects"], "wrongs_caught": caught, "wrongs": wrong,
            "p": [round(float(x), 6) for x in p]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--corpus", type=Path, action="append", required=True,
                    help="training corpus jsonl (repeatable)")
    ap.add_argument("--bench", type=Path, action="append", default=None,
                    help="evaluation jsonl (repeatable; default: frozen J-Bench)")
    ap.add_argument("--extra-eval", type=Path, action="append", default=[],
                    help="extra organic evaluation jsonl (e.g. test-math extension)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--limit", type=int, default=0, help="first N train items (smoke)")
    ap.add_argument("--eval-limit", type=int, default=0,
                    help="first N items per eval set (smoke)")
    ap.add_argument("--eval-every", type=int, default=1,
                    help="evaluate after every N epochs (0 = only at end)")
    ap.add_argument("--threads", type=int, default=0,
                    help="torch CPU threads (0 = default; leave room for the serve)")
    args = ap.parse_args()
    args.bench = args.bench or [DEFAULT_BENCH]

    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model

    if args.threads:
        torch.set_num_threads(args.threads)
        print(f"[lora] torch threads: {args.threads}", flush=True)

    items = []
    for path in args.corpus:
        loaded = load_jsonl(path)
        print(f"[lora] corpus {path.name}: {len(loaded)} items", flush=True)
        items.extend(loaded)
    train = [i for i in items if i["split"] == "train"]
    if args.limit:
        train = train[: args.limit]
    print(f"[lora] train items: {len(train)} "
          f"({sum(1 for i in train if i['label'] == 0)} wrong)", flush=True)

    evals = {}
    for path in args.bench + args.extra_eval:
        loaded = load_jsonl(path)
        label = path.stem
        evals[label] = [i for i in loaded if i["split"] == "test"]
    # organic (harvest) test items come from the corpus itself
    harvest_test = [i for i in items if i["split"] == "test"
                    and i["origin"].startswith("harvested:")]
    evals["organic-test"] = harvest_test
    if args.eval_limit:
        evals = {k: v[: args.eval_limit] for k, v in evals.items()}
    for tag, rows in evals.items():
        print(f"[lora] eval {tag}: {len(rows)} items", flush=True)

    inference, backend, torch = load_backend(args.model, args.device, args.dtype)
    tokenizer = backend.tokenizer
    yes_id, no_id = symbol_ids(tokenizer)
    print(f"[lora] yes_id {yes_id} no_id {no_id}", flush=True)

    base = backend.model
    for p in base.parameters():
        p.requires_grad_(False)
    base.eval()
    cfg = LoraConfig(r=args.rank, lora_alpha=args.alpha,
                     lora_dropout=args.dropout,
                     target_modules=LORA_TARGETS, bias="none", task_type=None)
    model = get_peft_model(base, cfg)
    model.print_trainable_parameters()
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    rng = np.random.default_rng(args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    result = {"name": args.name, "model": args.model,
              "config": {k: getattr(args, k) for k in
                         ("epochs", "lr", "rank", "alpha", "dropout", "accum",
                          "seed", "dtype", "device")},
              "train_items": len(train),
              "evals": {}, "seconds": None}
    t0 = time.time()

    def run_eval(epoch_tag: str) -> None:
        block = result["evals"].setdefault(epoch_tag, {})
        for tag, rows in evals.items():
            p = predict_p_yes(model, backend, inference, torch, rows,
                              yes_id, no_id)
            block[tag] = report(f"{epoch_tag} {tag}", p, rows)

    model.train()
    step = 0
    losses: list[float] = []
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(train))
        for n, k in enumerate(order, 1):
            item = train[int(k)]
            row = inference.validate_request(verdict_request(item))
            _, batch, positions = backend.encode(row)
            batch = {kk: v.to(backend.device) for kk, v in batch.items()}
            logits = model(**batch, use_cache=False,
                           logits_to_keep=positions.to(backend.device)).logits[0]
            two = logits[0, torch.tensor([yes_id, no_id], device=backend.device)]
            target = torch.tensor([0 if item["label"] else 1], device=backend.device)
            loss = F.cross_entropy(two.unsqueeze(0), target) / args.accum
            loss.backward()
            losses.append(float(loss.item()) * args.accum)
            step += 1
            if step % args.accum == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)
            if n % 25 == 0 or n == len(train):
                elapsed = time.time() - t0
                print(f"[lora] epoch {epoch} {n}/{len(train)} "
                      f"loss {np.mean(losses[-100:]):.4f} | {elapsed:.0f}s "
                      f"({elapsed / step:.2f} s/item)", flush=True)
        if step % args.accum:
            opt.step()
            opt.zero_grad(set_to_none=True)
        if args.eval_every and epoch % args.eval_every == 0:
            run_eval(f"epoch{epoch}")
            # Keep each epoch's adapter: if epoch 2 overfits we still have epoch 1.
            snap = args.out / f"adapter-{args.name}-epoch{epoch}"
            model.save_pretrained(snap)
            print(f"[lora] snapshot {snap.name}", flush=True)
            model.train()

    run_eval("final")
    result["seconds"] = round(time.time() - t0, 1)
    model.save_pretrained(args.out / f"adapter-{args.name}")
    (args.out / f"lora-{args.name}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"[lora] wrote {args.out / f'lora-{args.name}.json'} and adapter", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
