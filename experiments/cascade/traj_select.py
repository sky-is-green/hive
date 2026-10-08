#!/usr/bin/env python
"""Trajectory-based selection: pick among a task's samples with end-of-trace
monitor scores.

Trains the trajectory GRU on train tasks (same protocol as the context eval),
then for each held-out test task selects the sample with the best score.
Compares against single, majority, the frozen judge-prompt head, the raw
answer-slot margin, an ensemble, and the oracle.  Memory-safe (mmap rows).

Usage:
    traj_select.py --windows .../streams/scion-bench-monitor-ctx \
        --samples .../correction-samples.jsonl --pilot-dir .../logit-pilot-scion \
        --out results/.../traj-select-scion.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "cascade"))

from community_eval import canonical  # noqa: E402

torch.set_num_threads(4)
SEEDS = (0, 1, 2)


def load_windows(windows: Path):
    meta = json.loads((windows / "meta.json").read_text())
    order = meta["order"]
    splits = np.array([meta["tasks"][o.rsplit(":", 1)[0]] for o in order])
    files = sorted((windows / "dump").glob("win_*_h.bin"),
                   key=lambda p: int(p.stem.split("_")[1]))
    if len(files) != len(order):
        raise SystemExit(f"{len(files)} dumps for {len(order)} streams")
    seq = int(meta["seq"])
    dim = files[0].stat().st_size // (seq * 4)
    return (files, seq, dim, order, splits,
            np.asarray(meta["lengths"]), np.asarray(meta["labels"], float),
            np.asarray(meta["prompt_len"]),
            np.asarray(meta.get("truncated", [False] * len(order))))


def load_rows(files, seq, dim, idx, pos):
    X = np.zeros((len(idx), pos.shape[1], dim), dtype=np.float32)
    for j, i in enumerate(idx):
        mm = np.memmap(files[int(i)], dtype=np.float32, mode="r", shape=(seq, dim))
        X[j] = mm[pos[j]]
        del mm
    return X


def gru_scores(Xs, yy, splits_i, seed, *, hidden=32, pca=32, epochs=300):
    tr = np.array([j for j in range(len(Xs)) if splits_i[j] == "train"])
    flat = Xs[tr].reshape(-1, Xs.shape[2])
    mu = flat.mean(0, keepdims=True)
    sd = flat.std(0, keepdims=True) + 1e-6
    Z = (Xs - mu) / sd
    S = Z[tr].reshape(-1, Z.shape[2])
    S = S - S.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(S, full_matrices=False)
    V = Vt[:pca].T
    P = Z @ V
    fmu = P[tr].reshape(-1, pca).mean(0, keepdims=True)
    fsd = P[tr].reshape(-1, pca).std(0, keepdims=True) + 1e-6
    P = (P - fmu) / fsd
    Xt = torch.tensor(P, dtype=torch.float32)
    yt = torch.tensor(yy, dtype=torch.float32)

    torch.manual_seed(seed)
    perm = np.random.RandomState(seed).permutation(len(tr))
    nval = max(1, int(0.2 * len(tr)))
    vi, ti = tr[perm[:nval]], tr[perm[nval:]]
    gru = nn.GRU(pca, hidden, batch_first=True)
    head = nn.Linear(hidden, 1)
    opt = torch.optim.Adam(list(gru.parameters()) + list(head.parameters()),
                           lr=2e-3, weight_decay=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    from readout_judge import auc as auc_fn
    best, best_state, bad = -1.0, None, 0
    for _ in range(epochs):
        gru.train()
        head.train()
        opt.zero_grad()
        o, _ = gru(Xt[ti])
        loss = lossf(head(o[:, -1]).squeeze(1), yt[ti])
        loss.backward()
        opt.step()
        gru.eval()
        head.eval()
        with torch.no_grad():
            o, _ = gru(Xt[vi])
            sv = head(o[:, -1]).squeeze(1).numpy()
        a = auc_fn(list(sv), list(yy[vi]))
        if a is not None and a > best + 1e-4:
            best, bad = a, 0
            best_state = (gru.state_dict(), head.state_dict())
        else:
            bad += 1
            if bad >= 40:
                break
    if best_state:
        gru.load_state_dict(best_state[0])
        head.load_state_dict(best_state[1])
    gru.eval()
    head.eval()
    with torch.no_grad():
        o, _ = gru(Xt)
        return head(o[:, -1]).squeeze(1).numpy()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--windows", type=Path, required=True)
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--tasks", type=Path,
                    default=REPO / "experiments/cascade/tasks-bench.json")
    ap.add_argument("--pilot-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    files, seq, dim, order, splits, lengths, y, plen, trunc = load_windows(args.windows)
    n = len(order)
    tasks_by = {t["id"]: t for t in json.loads(args.tasks.read_text())["tasks"]}
    recs = {}
    for line in args.samples.read_text().splitlines():
        r = json.loads(line)
        recs[f"{r['task_id']}:{r['sample']}"] = r

    task_of = [o.rsplit(":", 1)[0] for o in order]
    test_tasks = sorted({task_of[i] for i in range(n) if splits[i] == "test"})
    by_task = defaultdict(list)
    for i in range(n):
        by_task[task_of[i]].append(i)

    scores = {}

    # frozen judge-prompt head (stored p_correct)
    frozen = np.full(n, np.nan)
    for i, o in enumerate(order):
        frozen[i] = float(recs[o].get("p_correct", float("nan")))
    scores["frozen"] = frozen

    # raw answer-slot margin from the slot-logit pilot (if present)
    if args.pilot_dir is not None:
        pidx = json.loads((args.pilot_dir / "prompts" / "index.json").read_text())
        prob = {}
        for line in (args.pilot_dir / "readout.tsv").read_text().splitlines():
            p = line.split("\t")
            if len(p) >= 2:
                prob[pidx[int(p[0])]["key"]] = dict(x.split("=") for x in p[2:])
        raw = np.full(n, np.nan)
        raw_idx = {}
        for i, o in enumerate(order):
            key = f"{task_of[i]}|sample|{o.rsplit(':', 1)[1]}"
            if key in prob:
                pa = max(float(prob[key].get("A", 0)), float(prob[key].get(" A", 0)))
                pb = max(float(prob[key].get("B", 0)), float(prob[key].get(" B", 0)))
                raw[i] = np.log(pa + 1e-12) - np.log(pb + 1e-12)
                raw_idx[i] = raw[i]
        scores["raw"] = raw

    # trajectory selectors
    rel = lengths - plen
    for name, horizon, mode in (("trj_last8", None, "last"),
                                ("trj_start256", 256, "start")):
        if mode == "last":
            idx = np.arange(n)
            pos = np.stack([np.arange(lengths[i] - 8, lengths[i]) for i in idx])
        else:
            idx = np.where((rel > horizon) & (~trunc))[0]
            t = np.linspace(0, horizon, 8).astype(int)
            pos = plen[idx][:, None] + t
        Xs = load_rows(files, seq, dim, idx, pos)
        y_i, splits_i = y[idx], splits[idx]
        seed_scores = []
        for seed in SEEDS:
            sc = gru_scores(Xs, y_i, splits_i, seed)
            sc = (sc - sc.mean()) / (sc.std() + 1e-12)
            seed_scores.append(sc)
            print(f"{name} seed {seed}: test-stream AUC "
                  f"{_auc_quick(sc[splits_i == 'test'], y_i[splits_i == 'test']):.3f}",
                  flush=True)
        arr = np.full(n, np.nan)
        arr[idx] = np.mean(seed_scores, axis=0)
        scores[name] = arr

    # ensemble: standardized trajectory + raw where both exist
    if "raw" in scores and not np.all(np.isnan(scores["raw"])):
        zt = np.full(n, np.nan)
        zr = np.full(n, np.nan)
        ok = ~np.isnan(scores["trj_last8"]) & ~np.isnan(scores["raw"])
        zt[ok] = (scores["trj_last8"][ok] - scores["trj_last8"][ok].mean()) / (
            scores["trj_last8"][ok].std() + 1e-12)
        zr[ok] = (scores["raw"][ok] - scores["raw"][ok].mean()) / (
            scores["raw"][ok].std() + 1e-12)
        scores["trj+raw"] = zt + zr

    def canon_of(i):
        return canonical(tasks_by[task_of[i]], recs[order[i]]["content"])

    def maj_pick(rows):
        reps = defaultdict(list)
        for i in rows:
            c = canon_of(i)
            if c is not None:
                reps[c].append(i)
        if reps:
            m = max(reps, key=lambda c: len(reps[c]))
            if len(reps[m]) > 1:
                return reps[m][0]
        return rows[0]

    def eval_arm(tid_list, pick):
        correct = n_used = 0
        for t in tid_list:
            i = pick(by_task[t])
            if i is None:
                continue
            n_used += 1
            correct += int(y[i] > 0.5)
        return {"tasks": n_used, "correct": correct,
                "acc": correct / max(1, n_used)}

    def best_pick(rows, s):
        vals = [(i, s[i]) for i in rows if not np.isnan(s[i])]
        return max(vals, key=lambda p: p[1])[0] if vals else None

    arms = {
        "single": eval_arm(test_tasks,
                           lambda rows: min(rows, key=lambda i: int(order[i].rsplit(":", 1)[1]))),
        "majority": eval_arm(test_tasks, lambda rows: maj_pick(rows)),
        "oracle": {"tasks": len(test_tasks),
                   "correct": sum(1 for t in test_tasks
                                  if any(y[i] > 0.5 for i in by_task[t])),
                   "acc": 0.0},
    }
    arms["oracle"]["acc"] = arms["oracle"]["correct"] / max(1, len(test_tasks))

    for name, s in scores.items():
        arms[name] = eval_arm(test_tasks, lambda rows, s=s: best_pick(rows, s))

    res = {"windows": str(args.windows), "n_streams": n,
           "n_test_tasks": len(test_tasks), "arms": arms}
    args.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0


def _auc_quick(s, yv):
    from readout_judge import auc
    return auc(list(s), list(yv)) or float("nan")


if __name__ == "__main__":
    raise SystemExit(main())
