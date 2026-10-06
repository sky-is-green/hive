#!/usr/bin/env python
"""Prompted-readout judge over J-Bench: P(correct) from one forward/prompt.

Two backends:
  --hf <model>        transformers: one prefill, read the restricted softmax
                      over the Yes/No label tokens at the answer position;
  --server <url>      llama-server /completion: same, from logprobs.

Both run the prompt in two option orders (A/B swapped) and average the Yes
mass to cancel position bias (the AnyJev L0 trick, applied at the prompt
level).  No training, no generation beyond one token.

Outputs <out>/readout-<name>.json with the per-item probabilities plus the
test-split metrics (AUC/BA/accuracy, false accepts/rejects at 0.5 and at the
no-false-accept threshold, ECE, order-flip rate).

Usage:
    python experiments/cascade/readout_judge.py --hf Qwen/Qwen3.5-0.8B \
        --name qwen35-0.8b --bench results/jev-readout-20261006/judge-bench.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[2]
DEFAULT_BENCH = REPO / "experiments/cascade/results/jev-readout-20261006/judge-bench.jsonl"
DEFAULT_OUT = REPO / "experiments/cascade/results/jev-readout-20261006"

BODY = (
    "Question: {prompt}\n"
    "Candidate answer: {candidate}\n"
    "Is the candidate answer correct?\n"
    "{a}. Yes\n"
    "{b}. No\n"
)


def order_body(item: dict, swap: bool) -> str:
    a, b = ("B", "A") if swap else ("A", "B")
    return BODY.format(prompt=item["prompt"], candidate=item["candidate"],
                       a=a, b=b)


# ------------------------------------------------------------------ metrics
def auc(scores: list[float], labels: list[int]) -> Optional[float]:
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return None
    # rank-based AUC with ties at 0.5
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    pos_ranks = [ranks[i] for i, l in enumerate(labels) if l == 1]
    n_pos, n_neg = len(pos), len(neg)
    return (sum(pos_ranks) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def best_ba(scores: list[float], labels: list[int]) -> tuple[float, float]:
    best_t, best = 0.5, 0.0
    for step in range(20, 100):
        t = step / 100.0
        tp = sum(1 for s, l in zip(scores, labels) if l == 1 and s >= t)
        tn = sum(1 for s, l in zip(scores, labels) if l == 0 and s < t)
        ba = 0.5 * (tp / max(1, sum(labels)) + tn / max(1, len(labels) - sum(labels)))
        if ba > best:
            best, best_t = ba, t
    return best, best_t


def threshold_stats(scores: list[float], labels: list[int], t: float) -> dict:
    accepted = [s >= t for s in scores]
    tp = sum(1 for a, l in zip(accepted, labels) if a and l == 1)
    fp = sum(1 for a, l in zip(accepted, labels) if a and l == 0)
    tn = sum(1 for a, l in zip(accepted, labels) if not a and l == 0)
    fn = sum(1 for a, l in zip(accepted, labels) if not a and l == 1)
    n = len(labels)
    return {"threshold": t, "accept_rate": (tp + fp) / n,
            "false_accepts": fp, "false_rejects": fn,
            "correct": tp + tn, "n": n,
            "bal_acc": 0.5 * (tp / max(1, tp + fn) + tn / max(1, tn + fp))}


def no_leak_threshold(scores: list[float], labels: list[int]) -> Optional[dict]:
    best = None
    for step in range(20, 100):
        t = step / 100.0
        stats = threshold_stats(scores, labels, t)
        if stats["false_accepts"] == 0:
            if best is None or stats["correct"] > best["correct"]:
                best = stats
    return best


def ece(scores: list[float], labels: list[int], bins: int = 10) -> Optional[float]:
    if not scores:
        return None
    total = 0.0
    n = len(scores)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, s in enumerate(scores)
               if (s >= lo and s < hi) or (b == bins - 1 and s == 1.0)]
        if not idx:
            continue
        conf = sum(scores[i] for i in idx) / len(idx)
        acc = sum(labels[i] for i in idx) / len(idx)
        total += len(idx) / n * abs(conf - acc)
    return total


def metrics(items: list[dict]) -> dict:
    scores = [i["p_correct"] for i in items]
    labels = [i["label"] for i in items]
    out: dict = {"n": len(items), "auc": auc(scores, labels),
                 "ece": ece(scores, labels),
                 "at_0.5": threshold_stats(scores, labels, 0.5),
                 "no_leak": no_leak_threshold(scores, labels)}
    ba, t = best_ba(scores, labels)
    out["best_ba"] = {"bal_acc": ba, "threshold": t}
    out["order_flip_rate"] = (sum(1 for i in items if i.get("order_disagree"))
                              / max(1, len(items)))
    return out


# ------------------------------------------------------------------ backends
class HFReadout:
    def __init__(self, model_name: str, device: str, chat: bool = True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype).to(device).eval()
        self.device = device
        self.chat = bool(chat and getattr(self.tok, "chat_template", None))
        if chat and not self.chat:
            print("warning: tokenizer has no chat template; plain prompt",
                  file=sys.stderr)

    def _prompt(self, item: dict, swap: bool) -> str:
        body = order_body(item, swap)
        if not self.chat:
            return body + "Answer:"
        messages = [
            {"role": "system",
             "content": "You are a decision function. Reply with the answer "
                        "label only."},
            {"role": "user", "content": body},
        ]
        return self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

    def _label_ids(self) -> tuple[int, int]:
        def first_id(text: str) -> int:
            ids = self.tok.encode(text, add_special_tokens=False)
            return ids[0]
        # prefer the space-prefixed single-token forms (" A", " B")
        a_ids, b_ids = [], []
        for form in (" A", "A"):
            ids = self.tok.encode(form, add_special_tokens=False)
            if len(ids) == 1:
                a_ids.append(ids[0])
        for form in (" B", "B"):
            ids = self.tok.encode(form, add_special_tokens=False)
            if len(ids) == 1:
                b_ids.append(ids[0])
        if not a_ids or not b_ids:
            # tokenizer splits the letter: fall back to first sub-token
            return (first_id(" A"), first_id(" B"))
        return a_ids[0], b_ids[0]

    def p_yes(self, item: dict, swap: bool) -> float:
        with self.torch.no_grad():
            ids = self.tok(self._prompt(item, swap), return_tensors="pt").input_ids
            ids = ids.to(self.device)
            logits = self.model(ids).logits[0, -1]
            id_a, id_b = self._label_ids()
            pair = self.torch.tensor([id_a, id_b], device=self.device)
            probs = self.torch.softmax(logits[pair].float(), dim=-1)
            # label A is Yes when not swapped, No when swapped
            p_yes_in_this_order = float(probs[1] if swap else probs[0])
        return p_yes_in_this_order


def _server_label_probs(url: str, prompt: str, timeout: float) -> dict[str, float]:
    body = {"prompt": prompt, "n_predict": 1, "temperature": 0.0,
            "seed": 1337, "cache_prompt": False, "logprobs": True}
    req = urllib.request.Request(url.rstrip("/") + "/completion",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.load(r)
    probs: dict[str, float] = {}
    cp = data.get("completion_probabilities") or []
    if cp:
        for entry in cp[0].get("probs", []):
            probs[entry.get("tok_str", "")] = float(entry.get("prob", 0.0))
        return probs
    # older response shape
    for entry in data.get("tokens", []):
        pass
    raise RuntimeError(f"no logprobs in server response: {list(data)[:8]}")


class ServerReadout:
    def __init__(self, url: str, timeout: float = 600.0):
        self.url = url
        self.timeout = timeout

    def p_yes(self, item: dict, swap: bool) -> float:
        probs = _server_label_probs(self.url, order_body(item, swap) + "Answer:",
                                    self.timeout)
        yes = sum(v for k, v in probs.items() if k.strip() in ("A", "Yes")
                  and not swap) + sum(v for k, v in probs.items()
                                      if k.strip() in ("B", "Yes") and swap)
        no = sum(v for k, v in probs.items() if k.strip() in ("B", "No")
                 and not swap) + sum(v for k, v in probs.items()
                                     if k.strip() in ("A", "No") and swap)
        if yes + no <= 0:
            # label letters missing from top-k; fall back to a generation call
            raise RuntimeError(f"labels not in top logprobs for {item['item_id']}")
        return yes / (yes + no)


class ServerChatReadout:
    """Prompted readout through a llama-server GGUF model.

    The prompt is built with the model's own chat template
    (``/apply-template``, thinking disabled) and the answer is read from the
    top logprobabilities at the first generated position (``n_probs``).
    """

    SYSTEM = ("You are a decision function. Reply with the answer label only.")

    def __init__(self, url: str, timeout: float = 600.0):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.missing = 0

    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(self.url + path,
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)

    def _prompt(self, item: dict, swap: bool) -> str:
        messages = [
            {"role": "system", "content": self.SYSTEM},
            {"role": "user", "content": order_body(item, swap)},
        ]
        result = self._post("/apply-template", {
            "messages": messages, "add_generation_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False}})
        return result["prompt"]

    def p_yes(self, item: dict, swap: bool) -> float:
        prompt = self._prompt(item, swap)
        result = self._post("/completion", {
            "prompt": prompt, "n_predict": 1, "temperature": 0.0, "seed": 1337,
            "n_probs": 20, "cache_prompt": False})
        cp = result.get("completion_probabilities") or []
        probs = {e["token"]: math.exp(e["logprob"])
                 for e in (cp[0].get("top_logprobs", []) if cp else [])}
        yes = sum(v for k, v in probs.items() if _is_label(k, "A", "Yes", swap))
        no = sum(v for k, v in probs.items() if _is_label(k, "B", "No", swap))
        if yes + no <= 0:
            self.missing += 1
            return 0.5
        return yes / (yes + no)


def _is_label(token: str, letter: str, word: str, swap: bool) -> bool:
    t = token.strip().strip(".").strip().lower()
    want_letter = letter if not swap else ("B" if letter == "A" else "A")
    return t in (want_letter.lower(), word.lower())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bench", type=Path, default=DEFAULT_BENCH)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--name", required=True)
    ap.add_argument("--hf", default="")
    ap.add_argument("--server", default="")
    ap.add_argument("--server-chat", default="",
                    help="llama-server URL; builds the prompt with "
                         "/apply-template (thinking disabled) and reads "
                         "n_probs at the answer position")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-chat", action="store_true",
                    help="do not apply the tokenizer chat template (HF backend)")
    ap.add_argument("--sanity", action="store_true",
                    help="print a greedy generation for item 1 (HF backend)")
    ap.add_argument("--split", default="test", choices=("train", "test", "all"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    items = [json.loads(line) for line in args.bench.read_text().splitlines()]
    if args.split != "all":
        items = [i for i in items if i["split"] == args.split]
    if args.limit:
        items = items[:args.limit]

    if args.hf:
        backend = HFReadout(args.hf, args.device, chat=not args.no_chat)
        name = args.name or Path(args.hf).name
    elif args.server_chat:
        backend = ServerChatReadout(args.server_chat)
        name = args.name or "server-chat"
    elif args.server:
        backend = ServerReadout(args.server)
        name = args.name or "server"
    else:
        raise SystemExit("need --hf, --server or --server-chat")

    out_items = []
    t0 = time.time()
    if args.sanity and isinstance(backend, HFReadout):
        item = items[0]
        ids = backend.tok(backend._prompt(item, False),
                          return_tensors="pt").input_ids.to(backend.device)
        with backend.torch.no_grad():
            gen = backend.model.generate(ids, max_new_tokens=6, do_sample=False)
        print("sanity generation:",
              repr(backend.tok.decode(gen[0, ids.shape[1]:],
                                      skip_special_tokens=False)), flush=True)
    for n, item in enumerate(items, 1):
        p1 = backend.p_yes(item, swap=False)
        p2 = backend.p_yes(item, swap=True)
        p_yes = 0.5 * (p1 + (1.0 - p2))
        out_items.append({**item, "p_correct": p_yes,
                          "p_order1": p1, "p_order2": p2,
                          "order_disagree": (p1 >= 0.5) != (p2 >= 0.5)})
        if n % 10 == 0 or n == len(items):
            print(f"[{n}/{len(items)}] {time.time()-t0:.0f}s", flush=True)

    result = {"name": name, "backend": args.hf or args.server,
              "split": args.split, "items": out_items}
    by_bucket: dict[str, list] = {}
    for i in out_items:
        by_bucket.setdefault(i["bucket"], []).append(i)
    result["metrics"] = metrics(out_items)
    result["metrics_by_bucket"] = {b: metrics(v) for b, v in by_bucket.items()}
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"readout-{name}.json"
    path.write_text(json.dumps(result, indent=1))
    m = result["metrics"]
    print(f"=== {name} ({args.split}, n={m['n']}) ===")
    print(f"AUC {m['auc']:.3f}  ECE {m['ece']:.3f}  "
          f"@0.5 acc {(m['at_0.5']['correct']/m['n']):.3f} "
          f"FA {m['at_0.5']['false_accepts']} FR {m['at_0.5']['false_rejects']}")
    if m["no_leak"]:
        nl = m["no_leak"]
        print(f"no-leak thr {nl['threshold']:.2f}: acc {nl['correct']/nl['n']:.3f} "
              f"accept {nl['accept_rate']:.3f}")
    print(f"order-flip {m['order_flip_rate']:.3f}")
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
