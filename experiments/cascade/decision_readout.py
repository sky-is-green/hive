#!/usr/bin/env python
"""Trainable readout for the Intern-Decision backbones (frozen features).

Stage 1 of "train the smallest format": keep the checkpoint frozen, replace
its two-symbol verbalizer with a small trained head over the decision-position
hidden state — the same position whose next-token logits the model itself
reads (the token immediately before ``<decision>``).  Fit on the J-Bench
**train** split, score the frozen **test** split, and compare with the model's
own readout (the ``p_correct`` the decision oracle recorded).

Stage 2 (``--bench`` is repeatable): train on J-Bench plus the harvested
corpus of organic 176B answers (``experiments/cascade/harvest.py``, each item
carrying ``source``), and report the frozen test metrics split by source so
the bench and the fresh-distribution slice stay separable.

Extraction reuses the repo's own skeleton (``inference.compile_row`` /
``HFBackend.encode``) so train and serving share one prompt path; metrics come
from ``readout_judge``.  CPU-only on purpose: a ROCm context beside the live
176B serve is the documented hard-reset hazard.

Usage (from the repo root):

    .venv/bin/python experiments/cascade/decision_readout.py run \
        --model internlm/Intern-Decision-0.8B --name intern-decision-0.8b
    .venv/bin/python experiments/cascade/decision_readout.py extract --limit 12 ...
    .venv/bin/python experiments/cascade/decision_readout.py fit --name intern-decision-0.8b

Combined corpus (stage 2):

    .venv/bin/python experiments/cascade/decision_readout.py run --name intern-decision-0.8b-harvest \
        --bench experiments/cascade/results/jev-readout-20261006/judge-bench.jsonl \
        --bench experiments/cascade/results/decision-harvest-20261017/harvest-bench.jsonl \
        --apply-head experiments/cascade/results/decision-readout-20261017/readout-head-intern-decision-0.8b.npz

Reweighted refit sharing an existing extraction (no re-extract):

    .venv/bin/python experiments/cascade/decision_readout.py fit \
        --name intern-decision-0.8b-harvest-w10 \
        --states-name intern-decision-0.8b-harvest --organic-boost 10
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "cascade"))
from readout_judge import (  # noqa: E402
    auc,
    best_ba,
    ece,
    no_leak_threshold,
    threshold_stats,
)

DEFAULT_BENCH = REPO_ROOT / "experiments/cascade/results/jev-readout-20261006/judge-bench.jsonl"
RESULTS_DIR = REPO_ROOT / "experiments/cascade/results/decision-readout-20261017"
JUDGE_REF = REPO_ROOT / "experiments/cascade/results/decision-oracle-20261017"
HEALTH_URL = "http://127.0.0.1:8081/health"

#: The noul question the D2 verdict uses — verbatim from
#: ``experiments/cascade/decision.py`` (InternDecisionJudge.verdict).
VERDICT_QUESTION = {
    "type": "noul",
    "instructions": "The candidate answer is correct and complete for the task.",
    "criteria": {
        "yes": "correct and complete",
        "no": "incorrect, incomplete, or contradictory",
    },
}


# --------------------------------------------------------------------------
# small helpers (ridge/platt identical to probe_judge.py)
# --------------------------------------------------------------------------

def _probe_health() -> dict:
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=3) as resp:
            return {"url": HEALTH_URL, "status": resp.status}
    except Exception as exc:  # noqa: BLE001 — informational
        return {"url": HEALTH_URL, "status": None, "error": str(exc)}


def _mem_available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1048576.0
    return float("nan")


def layer_map(depths: list[float], n_layers: int) -> list[tuple[str, int]]:
    out = []
    for f in depths:
        idx = max(1, min(n_layers, int(round(f * n_layers))))
        out.append((f"{f:g}", idx))
    return out


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float = 1.0,
              sample_weight: np.ndarray | None = None):
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-6
    Xs = (X - mu) / sd
    if sample_weight is None:
        Xw, yw = Xs, y
    else:
        rw = np.sqrt(np.asarray(sample_weight, dtype=np.float64))
        Xw, yw = Xs * rw[:, None], y * rw
    w = np.linalg.solve(Xw.T @ Xw + lam * np.eye(Xs.shape[1]), Xw.T @ yw)
    b = float(y.mean() if sample_weight is None
              else np.average(y, weights=sample_weight))
    return w, (mu, sd, b)


def ridge_score(X: np.ndarray, model) -> np.ndarray:
    w, (mu, sd, b) = model
    return ((X - mu) / sd) @ w + b


def platt(scores: np.ndarray, labels: np.ndarray,
          sample_weight: np.ndarray | None = None) -> tuple[float, float]:
    if sample_weight is None:
        wsum, w = 1.0, np.ones(len(labels))
    else:
        w = np.asarray(sample_weight, dtype=np.float64)
        wsum = float(w.sum())
    best = (0.0, 0.0, 1e9)
    for a in (0.5, 1.0, 2.0, 4.0, 8.0):
        for b in np.linspace(-2, 2, 9):
            p = 1.0 / (1.0 + np.exp(-(a * scores + b)))
            nll = -np.sum(w * (labels * np.log(p + 1e-9)
                               + (1 - labels) * np.log(1 - p + 1e-9))) / wsum
            if nll < best[2]:
                best = (a, b, nll)
    return best[0], best[1]


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------

def load_backend(model_id: str, device: str, dtype: str):
    from huggingface_hub import snapshot_download
    import torch

    snapshot = snapshot_download(model_id)
    if snapshot not in sys.path:
        sys.path.insert(0, snapshot)
    import inference  # type: ignore

    class _TextOnlyProcessor:
        tokenizer = None

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    inference.AutoProcessor = _TextOnlyProcessor
    backend = inference.HFBackend(checkpoint=snapshot, device=device, dtype=dtype)
    return inference, backend, torch


def extract(args: argparse.Namespace) -> None:
    import torch

    items = []
    for bench in args.bench:
        loaded = [json.loads(line) for line in bench.read_text().splitlines() if line.strip()]
        print(f"[readout] bench {bench.name}: {len(loaded)} items", flush=True)
        items.extend(loaded)
    if args.limit:
        items = items[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)

    inference, backend, torch = load_backend(args.model, args.device, args.dtype)
    if args.adapter:
        from peft import PeftModel
        backend.model = PeftModel.from_pretrained(backend.model, str(args.adapter))
        backend.model.eval()
        print(f"[readout] adapter {args.adapter}", flush=True)
    temperature = float(inference.DEFAULT_TEMPERATURE)
    tok = backend.tokenizer
    print(f"[readout] {args.model} on {args.device} ({args.dtype}), "
          f"T={temperature:.4f}, {len(items)} items", flush=True)

    health_before = _probe_health()
    print(f"[readout] serve health before: {health_before}", flush=True)

    samples: list[tuple[float, float]] = []
    stop = threading.Event()

    def memwatch() -> None:
        t_start = time.time()
        while not stop.is_set():
            samples.append((time.time() - t_start, _mem_available_gib()))
            stop.wait(2.0)

    watcher = threading.Thread(target=memwatch, daemon=True)
    watcher.start()
    t0 = time.time()

    states: dict[str, dict[str, np.ndarray]] = {}
    raw: dict[str, float] = {}
    metas: dict[str, dict] = {}
    depth_specs: list[tuple[str, int]] = []
    n_layers = 0

    for n, item in enumerate(items, 1):
        request = {
            "state": {"task": item["prompt"], "candidate_answer": item["candidate"]},
            "questions": {"verdict": dict(VERDICT_QUESTION)},
        }
        row = inference.validate_request(request)
        compiled, batch, positions = backend.encode(row)
        pos = int(positions[0])

        started = time.perf_counter()
        with torch.inference_mode():
            try:
                out = backend.model(**batch, use_cache=False,
                                    output_hidden_states=True,
                                    logits_to_keep=positions)
                kept = True
            except TypeError:  # older forwards without logits_to_keep
                out = backend.model(**batch, use_cache=False,
                                    output_hidden_states=True)
                kept = False
        ms = (time.perf_counter() - started) * 1000.0

        hs = out.hidden_states
        if not depth_specs:
            n_layers = len(hs) - 1
            depth_specs = layer_map(args.depths, n_layers)
            print(f"[readout] {n_layers} layers; depths {depth_specs}", flush=True)

        feats = {f: hs[idx][0, pos].float().numpy() for f, idx in depth_specs}

        # The model's own readout at the same position: softmax over the
        # field symbols (A=no, B=yes for the noul question), then the
        # checkpoint's temperature — a replica of scale_probabilities().
        assert compiled.symbols["verdict"] == ("A", "B"), compiled.symbols
        logits_vec = out.logits[0][0] if kept else out.logits[0][pos]
        ids = [tok.encode(s, add_special_tokens=False) for s in compiled.symbols["verdict"]]
        assert all(len(i) == 1 for i in ids), ids
        probs = torch.softmax(logits_vec[[i[0] for i in ids]].float(), dim=-1).tolist()
        logs = [math.log(p) if p else -math.inf for p in probs]
        mx = max(logs)
        w = [math.exp((lp - mx) / temperature) for lp in logs]
        raw[item["item_id"]] = w[1] / sum(w)  # P(yes) after temperature

        states[item["item_id"]] = feats
        metas[item["item_id"]] = {
            "input_tokens": int(batch["input_ids"].shape[-1]),
            "ms": round(ms, 1),
        }
        if n % 25 == 0 or n == len(items):
            print(f"[{n}/{len(items)}] {time.time()-t0:.0f}s", flush=True)

    elapsed = time.time() - t0
    stop.set()
    watcher.join(timeout=5)
    health_after = _probe_health()
    min_avail = min((v for _, v in samples), default=float("nan"))
    print(f"[readout] serve health after: {health_after} | "
          f"MemAvailable min {min_avail:.2f} GiB | {elapsed:.0f}s", flush=True)

    (args.out / "memwatch.log").write_text(
        f"# MemAvailable during decision-readout extraction ({args.name})\n"
        f"# min {min_avail:.2f} GiB | serve before={health_before['status']} "
        f"after={health_after['status']}\n"
        + "".join(f"{t:8.1f}s  MemAvailable={v:.2f} GiB\n" for t, v in samples)
    )

    fracs = [f for f, _ in depth_specs]
    np.savez_compressed(
        args.out / f"readout-states-{args.name}.npz",
        item_ids=np.array([i["item_id"] for i in items]),
        labels=np.array([i["label"] for i in items]),
        splits=np.array([i["split"] for i in items]),
        buckets=np.array([i["bucket"] for i in items]),
        origins=np.array([i["origin"] for i in items]),
        sources=np.array([i.get("source", "bench") for i in items]),
        task_ids=np.array([i.get("task_id", "") for i in items]),
        p_raw=np.array([raw[i["item_id"]] for i in items]),
        input_tokens=np.array([metas[i["item_id"]]["input_tokens"] for i in items]),
        ms=np.array([metas[i["item_id"]]["ms"] for i in items]),
        **{f"L{f}": np.stack([states[i["item_id"]][f] for i in items]) for f in fracs},
    )
    (args.out / f"extract-{args.name}.json").write_text(json.dumps({
        "name": args.name,
        "model": args.model,
        "adapter": str(args.adapter) if args.adapter else None,
        "device": args.device,
        "dtype": args.dtype,
        "temperature": temperature,
        "n_layers": n_layers,
        "depths": [{"depth": f, "layer": i} for f, i in depth_specs],
        "n_items": len(items),
        "elapsed_s": round(elapsed, 1),
        "serve": {"before": health_before, "after": health_after},
        "mem_available_min_gib": round(float(min_avail), 2),
        "median_ms": round(statistics.median(m["ms"] for m in metas.values()), 1),
    }, indent=2) + "\n")
    print(f"[readout] wrote {args.out / f'readout-states-{args.name}.npz'}", flush=True)


# --------------------------------------------------------------------------
# fit + score
# --------------------------------------------------------------------------

def _metrics(p: np.ndarray, y: np.ndarray) -> dict:
    p_, y_ = list(map(float, p)), list(map(int, y))
    ba, thr = best_ba(p_, y_)
    return {
        "n": len(y_),
        "auc": auc(p_, y_),
        "best_ba": {"bal_acc": ba, "threshold": thr},
        "ece": ece(p_, y_),
        "no_leak": no_leak_threshold(p_, y_),
        "at_0.5": threshold_stats(p_, y_, 0.5),
    }


def _source_line(src: str, m: dict) -> str:
    at = m["at_0.5"]
    auc_s = f"{m['auc']:.3f}" if m["auc"] else "  n/a"
    return (f"  └ {src:9s} n {m['n']:3d}  AUC {auc_s:>5s}  "
            f"acc {at.get('correct', '?')}/{m['n']}  "
            f"FA {at.get('false_accepts', '?')}  FR {at.get('false_rejects', '?')}  "
            f"BA {m['best_ba']['bal_acc']:.3f}  ECE {m['ece']:.3f}")


def _apply_saved_heads(path: Path, X: np.ndarray, name: str) -> dict[str, np.ndarray]:
    """Apply ``ridge_concat`` + ``mlp`` heads saved by an earlier fit run."""
    h = np.load(path, allow_pickle=False)
    out: dict[str, np.ndarray] = {}
    if "ridge_concat_w" in h.files:
        s = ((X - h["ridge_concat_mu"]) / h["ridge_concat_sd"]) @ h["ridge_concat_w"] \
            + float(h["ridge_concat_bias"])
        a, b = h["ridge_concat_platt"]
        out[f"{name}:ridge-concat"] = 1.0 / (1.0 + np.exp(-(a * s + b)))
    if "mlp_mu" in h.files:
        hid = np.maximum(((X - h["mlp_mu"]) / h["mlp_sd"]) @ h["mlp_0.weight"].T
                         + h["mlp_0.bias"], 0.0)
        logit = (hid @ h["mlp_2.weight"].T + h["mlp_2.bias"]).ravel()
        out[f"{name}:mlp-concat"] = 1.0 / (1.0 + np.exp(-logit))
    return out


def _bucket_breakdown(p: np.ndarray, mask: np.ndarray, labels: np.ndarray,
                      sources: np.ndarray, buckets: np.ndarray) -> dict:
    out: dict = {}
    for src in sorted(set(sources[mask].tolist())):
        for bucket in sorted(set(buckets[mask & (sources == src)].tolist())):
            m = mask & (sources == src) & (buckets == bucket)
            out[f"{src}/{bucket}"] = _metrics(p[m], labels[m])
    return out


def _origin_breakdown(p: np.ndarray, mask: np.ndarray, labels: np.ndarray,
                      origins: np.ndarray) -> dict:
    """Acceptance @0.5 per origin — the stage-1 summary table, machine-read."""
    out: dict = {}
    for origin in sorted(set(origins[mask].tolist())):
        m = mask & (origins == origin)
        accepted = p[m] >= 0.5
        y = labels[m]
        out[origin] = {
            "n": int(m.sum()),
            "wrong": int((y == 0).sum()),
            "wrong_accepted": int(((y == 0) & accepted).sum()),
            "correct": int((y == 1).sum()),
            "correct_accepted": int(((y == 1) & accepted).sum()),
        }
    return out


def _mlp_fit(X: np.ndarray, y: np.ndarray, *, seed: int, epochs: int,
             hidden: int = 64, lr: float = 1e-3, wd: float = 1e-4,
             sample_weight: np.ndarray | None = None):
    import torch

    torch.manual_seed(seed)
    Xt = torch.tensor(X, dtype=torch.float32)
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-6
    Xt = (Xt - mu) / sd
    yt = torch.tensor(y, dtype=torch.float32)
    net = torch.nn.Sequential(
        torch.nn.Linear(Xt.shape[1], hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, 1),
    )
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
    wt = (torch.tensor(np.asarray(sample_weight), dtype=torch.float32)
          if sample_weight is not None else None)
    for _ in range(epochs):
        opt.zero_grad()
        per = loss_fn(net(Xt).squeeze(-1), yt)
        loss = (per * wt).mean() if wt is not None else per.mean()
        loss.backward()
        opt.step()
    return net, (mu, sd)


def _mlp_predict(X: np.ndarray, net, stats) -> np.ndarray:
    import torch

    mu, sd = stats
    with torch.no_grad():
        Xt = (torch.tensor(X, dtype=torch.float32) - mu) / sd
        return torch.sigmoid(net(Xt).squeeze(-1)).numpy()


def fit(args: argparse.Namespace) -> None:
    z = np.load(args.out / f"readout-states-{args.states_name or args.name}.npz",
                allow_pickle=False)
    labels = z["labels"].astype(int)
    splits = z["splits"]
    fracs = sorted((k for k in z.files if k.startswith("L")),
                   key=lambda k: float(k[1:]))
    train, test = splits == "train", splits == "test"
    if train.sum() < 10 or test.sum() < 5:
        raise SystemExit(f"train/test split too small ({train.sum()}/{test.sum()}) "
                         "— run without --limit for the real fit")
    y_tr, y_te = labels[train], labels[test]

    sources = (z["sources"].astype(str) if "sources" in z.files
               else np.array(["bench"] * len(labels)))
    test_sources = sorted(set(sources[test].tolist()))
    buckets = z["buckets"].astype(str)
    origins = z["origins"].astype(str)
    w_tr = np.ones(len(labels))
    n_org_wrong = 0
    if args.organic_boost != 1.0:
        organic_wrong = (train & (sources == "harvest") & (labels == 0)
                         & np.char.startswith(origins, "harvested:"))
        n_org_wrong = int(organic_wrong.sum())
        w_tr[organic_wrong] = args.organic_boost

    def _by_source(p: np.ndarray) -> dict:
        if len(test_sources) < 2:
            return {}
        return {src: _metrics(p[test & (sources == src)], labels[test & (sources == src)])
                for src in test_sources}

    protocol = ("ridge fitted on the train split only; Platt on train scores; "
                "frozen test split; MLP seed fixed, no test peeking")
    if len(set(sources.tolist())) > 1:
        protocol = ("combined corpus (J-Bench + harvested 176B answers); " + protocol)
    results: dict = {
        "name": args.name,
        "protocol": protocol,
        "corpus": {
            "items": len(labels),
            "sources": {src: int((sources == src).sum())
                        for src in sorted(set(sources.tolist()))},
            "train": int(train.sum()),
            "test": int(test.sum()),
        },
        "training_weights": {"organic_wrong_boost": args.organic_boost,
                             "organic_wrong_items": n_org_wrong},
        "readouts": {},
        "baseline": {
            "train": _metrics(z["p_raw"][train], y_tr),
            "test": _metrics(z["p_raw"][test], y_te),
        },
    }
    baseline_src = _by_source(z["p_raw"])
    if baseline_src:
        results["baseline"]["by_source"] = baseline_src
    heads: dict[str, np.ndarray] = {}
    ps: dict[str, np.ndarray] = {"raw(readout)": z["p_raw"]}

    def _ridge_readout(X: np.ndarray, key: str) -> None:
        model = ridge_fit(X[train].astype(np.float64), y_tr.astype(np.float64),
                          sample_weight=w_tr[train])
        s_tr = ridge_score(X[train].astype(np.float64), model)
        a, b = platt(s_tr, y_tr.astype(np.float64), sample_weight=w_tr[train])
        p = 1.0 / (1.0 + np.exp(-(a * ridge_score(X.astype(np.float64), model) + b)))
        entry = {"kind": "ridge+platt",
                 "train": _metrics(p[train], labels[train]),
                 "test": _metrics(p[test], labels[test])}
        entry["by_source"] = _by_source(p)
        if not entry["by_source"]:
            del entry["by_source"]
        results["readouts"][key] = entry
        ps[key] = p
        w, (mu, sd, bias) = model
        heads[f"ridge_{key}_w"] = w
        heads[f"ridge_{key}_mu"] = np.asarray(mu)
        heads[f"ridge_{key}_sd"] = np.asarray(sd)
        heads[f"ridge_{key}_bias"] = np.asarray(bias)
        heads[f"ridge_{key}_platt"] = np.asarray([a, b])

    for frac in fracs:
        _ridge_readout(z[frac], frac)
    Xc = np.concatenate([z[f] for f in fracs], axis=1)
    _ridge_readout(Xc, "concat")

    net, stats = _mlp_fit(Xc[train], y_tr, seed=args.seed, epochs=args.mlp_epochs,
                          sample_weight=w_tr[train])
    p_mlp = _mlp_predict(Xc, net, stats)
    ps["mlp-concat"] = p_mlp
    entry = {"kind": f"mlp(64,{args.mlp_epochs}ep,seed{args.seed})",
             "train": _metrics(p_mlp[train], labels[train]),
             "test": _metrics(p_mlp[test], labels[test])}
    entry["by_source"] = _by_source(p_mlp)
    if not entry["by_source"]:
        del entry["by_source"]
    results["readouts"]["mlp-concat"] = entry
    for k, v in net.state_dict().items():
        heads[f"mlp_{k}"] = v.detach().numpy()
    heads["mlp_mu"], heads["mlp_sd"] = stats
    np.savez_compressed(args.out / f"readout-head-{args.name}.npz", **heads)
    results["heads_file"] = f"readout-head-{args.name}.npz"

    # Optional cross-corpus comparison: heads saved by an earlier fit (e.g.
    # the stage-1 bench-only readout) applied to this corpus unchanged.
    if args.apply_head:
        applied_name = args.apply_head_name or args.apply_head.stem.replace("readout-head-", "")
        for key, p in _apply_saved_heads(args.apply_head, Xc, applied_name).items():
            entry = {"kind": f"applied:{args.apply_head.name}",
                     "train": _metrics(p[train], labels[train]),
                     "test": _metrics(p[test], labels[test])}
            entry["by_source"] = _by_source(p)
            if not entry["by_source"]:
                del entry["by_source"]
            results["readouts"][key] = entry
            ps[key] = p
        try:
            results["applied_head"] = str(args.apply_head.resolve().relative_to(REPO_ROOT))
        except ValueError:  # outside the repo — record as given
            results["applied_head"] = str(args.apply_head)

    results["breakdowns"] = {
        key: {
            "by_bucket_test": _bucket_breakdown(ps[key], test, labels, sources, buckets),
            "by_origin_test": _origin_breakdown(ps[key], test, labels, origins),
        }
        for key in ("raw(readout)", "concat", "mlp-concat",
                    *[k for k in results["readouts"] if ":" in k])
        if key in ps
    }

    # Integrity check against the decision-oracle record (same model, same
    # verdict path): our replica of the model's own readout should match.
    ref_path = JUDGE_REF / f"judge-{args.name}.json"
    if ref_path.exists():
        ref = {i["item_id"]: i["p_correct"] for i in json.loads(ref_path.read_text())["items"]}
        diffs = [abs(float(z["p_raw"][k]) - ref[iid])
                 for k, iid in enumerate(z["item_ids"]) if iid in ref]
        results["baseline"]["oracle_ref"] = {
            "path": str(ref_path.relative_to(REPO_ROOT)),
            "matched_items": len(diffs),
            "max_abs_p_diff": round(max(diffs), 6) if diffs else None,
        }

    (args.out / f"readout-{args.name}.json").write_text(json.dumps(results, indent=1) + "\n")

    def _line(key: str, entry: dict) -> str:
        m = entry["test"]
        at = m["at_0.5"]
        uc = "uc" if not m["auc"] else f"{m['auc']:.3f}"
        return (f"{key:14s} AUC {uc:>6s}  acc@0.5 {at.get('correct', '?')}/{m['n']}  "
                f"FA {at.get('false_accepts', '?')}  FR {at.get('false_rejects', '?')}  "
                f"BA {m['best_ba']['bal_acc']:.3f}  ECE {m['ece']:.3f}"
                if m["auc"] else f"{key:14s} AUC n/a (single class)")

    print(f"=== readout-{args.name} (test split, n={int(test.sum())}) ===")

    def _emit(key: str, entry: dict) -> None:
        print(_line(key, entry))
        for src, m in (entry.get("by_source") or {}).items():
            print(_source_line(src, m))

    _emit("raw(readout)", results["baseline"])
    for key, entry in results["readouts"].items():
        _emit(key, entry)
    if "oracle_ref" in results["baseline"]:
        ref = results["baseline"]["oracle_ref"]
        print(f"oracle_ref: {ref['matched_items']} items, max |Δp| "
              f"{ref['max_abs_p_diff']}")
    print(f"[readout] wrote {args.out / f'readout-{args.name}.json'}", flush=True)


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("action", choices=("run", "extract", "fit"))
    ap.add_argument("--model", default="internlm/Intern-Decision-0.8B")
    ap.add_argument("--name", required=True)
    ap.add_argument("--bench", type=Path, action="append", default=None,
                    help="bench jsonl (repeatable; default: frozen J-Bench)")
    ap.add_argument("--out", type=Path, default=RESULTS_DIR)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--adapter", type=Path, default=None,
                    help="PEFT adapter directory to load on top of --model (extract)")
    ap.add_argument("--depths", default="0.5,0.75,1.0",
                    help="comma-separated depth fractions")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--mlp-epochs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--apply-head", type=Path, default=None,
                    help="saved readout-head npz applied to this corpus for comparison")
    ap.add_argument("--apply-head-name", default=None,
                    help="label for applied-head readouts (default: file stem)")
    ap.add_argument("--states-name", default=None,
                    help="read readout-states-<states-name>.npz (default: --name)")
    ap.add_argument("--organic-boost", type=float, default=1.0,
                    help="sample weight for harvested organic wrongs during fitting")
    args = ap.parse_args()
    args.depths = [float(t) for t in args.depths.split(",")]
    args.bench = args.bench or [DEFAULT_BENCH]

    if args.action in ("run", "extract"):
        extract(args)
    if args.action in ("run", "fit"):
        fit(args)


if __name__ == "__main__":
    main()
