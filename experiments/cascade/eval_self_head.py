#!/usr/bin/env python3
"""Window A: the 176B's own heads on the harvest organic-66 and qa-75.

Reads the judge states pulled through the decision-head server (the 140-item harvest transfer
subset), applies the 176B's two heads (J-Bench-frozen and community in-domain), and reports the
same slices the partner baseline used.  Also runs an exploratory leave-one-out fusion that adds
the own-head signals on the full held-out test-182 (bench-116 states come from the port's
`probe-states-main176b.npz`; organic-66 from this window).

    .venv/bin/python experiments/cascade/eval_self_head.py --out <path.json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WORK = REPO.parent
sys.path.insert(0, str(HERE))

import fusion_probe as fp  # noqa: E402

WINDOW = REPO / "experiments/cascade/results/scion176b-self-organic-20261018/probe-states-scion176b-transfer.npz"
PORT = WORK / "scion/decision-heads-work/overnight/probe-states-main176b.npz"
HEAD_JBENCH = WORK / "scion/decision-heads/scion-176b-a6b-decision-head.json"
HEAD_COMMUNITY = WORK / "scion/decision-heads-work/overnight/scion176b-community-indomain-head.json"


def load_states(path: Path) -> tuple[dict[str, np.ndarray], dict[str, int], dict[str, str]]:
    z = np.load(path, allow_pickle=True)
    key = "L1" if "L1" in z.files else "H"
    ids = [str(i) for i in z["item_ids"]]
    labels = {i: int(v) for i, v in zip(ids, z["labels"])}
    splits = {i: str(v) for i, v in zip(ids, z["splits"])}
    states = {i: z[key][n].astype(np.float64) for n, i in enumerate(ids)}
    return states, labels, splits


def head_probe(name: str, head: dict, states: dict[str, np.ndarray], ids: list[str]) -> dict[str, np.ndarray]:
    present = [i for i in ids if i in states]
    vecs = np.stack([states[i] for i in present])
    p = fp.apply_head(head, vecs)
    return dict(zip(present, p.tolist()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    meta, p08 = fp.load_r2()
    ids = [str(i) for i in meta["item_ids"]]
    labels = meta["labels"].astype(int)
    splits = meta["splits"]
    sources = meta["sources"]
    buckets = meta["buckets"]
    pos = {i: n for n, i in enumerate(ids)}

    wstates, wlabels, wsplits = load_states(WINDOW)
    organic = [i for i in ids if sources[pos[i]] == "harvest" and splits[pos[i]] == "test"]
    qa = [i for i in ids if sources[pos[i]] == "harvest" and buckets[pos[i]] == "qa"]
    window_extra = [i for i in wstates if i not in set(organic) | set(qa)]
    not_in_r2 = [i for i in wstates if i not in pos]
    print(f"window items {len(wstates)}; organic {len(organic)}; qa {len(qa)}; "
          f"extra {window_extra}; not in r2 {not_in_r2}")
    for i in organic + qa:
        if i not in wstates:
            raise SystemExit(f"window states missing {i}")
        if wlabels[i] != int(labels[pos[i]]):
            raise SystemExit(f"label mismatch for {i}")

    heads = {
        "176b-own-jbench": json.loads(HEAD_JBENCH.read_text()),
        "176b-own-community": json.loads(HEAD_COMMUNITY.read_text()),
    }
    probs = {name: head_probe(name, head, wstates, list(wstates)) for name, head in heads.items()}

    # cross-window sanity: the 4 shared bench train seeds were embedded in both windows
    pstates_all, _, _ = load_states(PORT)
    shared = sorted(set(wstates) & set(pstates_all))
    if shared:
        a = np.stack([wstates[i] for i in shared])
        b = np.stack([pstates_all[i] for i in shared])
        cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
        print(f"shared states {len(shared)}: mean cosine {cos.mean():.6f} min {cos.min():.6f}")

    slices = {"organic-66": organic, "qa-harvest-75": qa,
              "window-140": [i for i in wstates if i in pos]}
    out: dict = {"window": str(WINDOW), "slices": {}, "heads": {}}
    for sname, sl in slices.items():
        y = np.array([int(labels[pos[i]]) for i in sl])
        out["slices"][sname] = {"n": len(sl), "wrong": int((y == 0).sum())}
    for name, pr in probs.items():
        entry: dict = {}
        for sname, sl in slices.items():
            y = np.array([int(labels[pos[i]]) for i in sl])
            p = np.array([pr[i] for i in sl])
            e = {"auc": fp.auc(y, p),
                 "catch_at_0.5": int(((y == 0) & (p < 0.5)).sum()),
                 "correct_rejected_at_0.5": int(((y == 1) & (p < 0.5)).sum())}
            if sname == "organic-66":
                e["pareto_organic"] = fp.pareto(y, p, [0, 3, 5, 9, 14])
            entry[sname] = e
        out["heads"][name] = entry

    # references for the organic slice, from the committed fusion probe
    ref = json.loads((REPO / "experiments/cascade/results/fusion-probe-20261018/fusion-probe.json").read_text())
    out["references_organic"] = {
        k: {"auc": v["organic-66"]["auc"], "catch_at_0.5": v["organic-66"]["catch_at_0.5"],
            "correct_rejected_at_0.5": v["organic-66"]["correct_rejected_at_0.5"],
            "pareto_organic": v["organic-66"]["pareto_organic"]}
        for k, v in ref["signals"].items()
        if k in ("bonsai-frozen", "bonsai-community", "L0.5", "mlp-concat")
    }

    # exploratory fusion on the full held-out test-182, own-head signal included
    pstates = pstates_all
    test = splits == "test"
    f176 = {}
    for name, head in heads.items():
        p = np.full(len(ids), np.nan)
        have = [i for i in ids if i in wstates or i in pstates]
        vecs = np.stack([(wstates if i in wstates else pstates)[i] for i in have])
        p[[pos[i] for i in have]] = fp.apply_head(head, vecs)
        f176[name] = p
    covered = np.array([not np.isnan(f176["176b-own-community"][n]) for n in range(len(ids))])
    test_c = test & covered
    if (test & ~covered).any():
        raise SystemExit("test-182 not fully covered by own-head states")
    y = labels[test_c]
    org_mask = (sources[test_c] == "harvest")
    feats = {
        "self-community": fp.logit(f176["176b-own-community"][test_c]),
        "self-jbench": fp.logit(f176["176b-own-jbench"][test_c]),
        "L0.5": fp.logit(p08["L0.5"][test_c]),
        "MLP": fp.logit(p08["mlp-concat"][test_c]),
        "p_raw": fp.logit(p08["p_raw"][test_c]),
    }
    combos = {
        "self-community": ["self-community"],
        "self-jbench": ["self-jbench"],
        "self-community+self-jbench": ["self-community", "self-jbench"],
        "self-community+L0.5+MLP+p_raw": ["self-community", "L0.5", "MLP", "p_raw"],
        "self-both+L0.5+MLP+p_raw": ["self-community", "self-jbench", "L0.5", "MLP", "p_raw"],
    }
    fusion: dict = {}
    for name, fl in combos.items():
        F = np.stack([feats[f] for f in fl], axis=1)
        pv = fp.loo_fusion(F, y)
        fusion[name] = {
            "features": fl,
            "auc_test": fp.auc(y, pv),
            "auc_organic": fp.auc(y[org_mask], pv[org_mask]),
            "pareto_organic": fp.pareto(y[org_mask], pv[org_mask], [0, 3, 5, 9, 14]),
        }
    out["fusion_loo_test182"] = fusion

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}")
    for name, entry in out["heads"].items():
        o = entry["organic-66"]
        pa = o["pareto_organic"]
        seq = " ".join(f"{pa[str(c)]['caught']}/{pa[str(c)]['correct_rejected']}" for c in (0, 3, 5, 9, 14))
        print(f"{name:20s} organic AUC {o['auc']:.3f} catch@0.5 {o['catch_at_0.5']}/9 "
              f"CR@0.5 {o['correct_rejected_at_0.5']}/57 | catch/CR {seq}")
    for name, entry in fusion.items():
        pa = entry["pareto_organic"]
        seq = " ".join(f"{pa[str(c)]['caught']}/{pa[str(c)]['correct_rejected']}" for c in (0, 3, 5, 9, 14))
        print(f"fusion {name:32s} test AUC {entry['auc_test']:.3f} organic AUC {entry['auc_organic']:.3f} | {seq}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
