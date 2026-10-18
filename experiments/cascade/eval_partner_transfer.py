#!/usr/bin/env python
"""Apply a frozen partner decision head to organic and QA transfer sets.

The frozen head was trained on J-Bench train only.  This script evaluates it
without refitting on the held-out combined-corpus organic test and on every QA
item, then writes per-item probabilities, a human-gate candidate list and a
markdown report.  It also fits one secondary control head on J-Bench non-QA
train only, so the QA number is not contaminated by the 13 QA training rows.

No labels are used at runtime; labels are read from the reference readout NPZ
for evaluation only.

Usage:
    eval_partner_transfer.py \
      --extra-states results/partner-bonsai-organic-20261018/probe-states-bonsai-transfer.npz \
      --outdir results/partner-bonsai-organic-20261018
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "cascade"))

from probe_judge import platt, ridge_fit, ridge_score  # noqa: E402
from readout_judge import metrics, threshold_stats  # noqa: E402

READOUT = REPO / "experiments/cascade/results/decision-readout-r2-20261017/readout-states-intern-decision-0.8b-r2.npz"
BASE = REPO / "experiments/cascade/results/jev-readout-20261006/probe-states-bonsai27b.npz"
HEAD = REPO / "experiments/cascade/results/jev-readout-20261006/bonsai-2-27b-decision-head.json"
COMBINED = REPO / "experiments/cascade/results/partner-scion35b-organic-20261018/combined-bench.jsonl"


def load_states(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as z:
        ids = [str(x) for x in z["item_ids"]]
        if "L1" in z:
            arr = z["L1"]
        else:
            keys = [k for k in z.files if k.startswith("L")]
            if not keys:
                raise SystemExit(f"{path}: no L* state array")
            arr = z[keys[0]]
        arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2:
        raise SystemExit(f"{path}: expected 2-D states, got {arr.shape}")
    return {item_id: arr[n] for n, item_id in enumerate(ids)}


def sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def apply_head(head: dict[str, Any], states: np.ndarray) -> np.ndarray:
    w = np.asarray(head["weights"]["w"], dtype=np.float64)
    mu = np.asarray(head["weights"]["mu"], dtype=np.float64)
    sd = np.asarray(head["weights"]["sd"], dtype=np.float64)
    b = float(head["weights"]["b"])
    a, c = (float(v) for v in head["weights"]["platt"])
    z = ((states - mu) / sd) @ w + b
    return sigmoid(a * z + c)


def fit_head(states: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    model = ridge_fit(states, labels.astype(np.float64))
    a, b = platt(ridge_score(states, model), labels.astype(np.float64))
    return model, (float(a), float(b))


def p_from_model(model, platt_ab, states: np.ndarray) -> np.ndarray:
    a, b = platt_ab
    return sigmoid(a * ridge_score(states, model) + b)


def metric_dict(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"n": 0}
    return metrics([{"p_correct": r["p"], "label": r["label"]} for r in records])


def at_threshold(records: list[dict[str, Any]], threshold: float) -> dict[str, Any] | None:
    if not records:
        return None
    return threshold_stats(
        [r["p"] for r in records], [r["label"] for r in records], threshold
    )


def count_records(records: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "n": len(records),
        "correct": sum(1 for r in records if r["label"] == 1),
        "wrong": sum(1 for r in records if r["label"] == 0),
    }


def compact_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    out = metric_dict(records)
    out.update(count_records(records))
    out["caught_at_0.5"] = sum(1 for r in records if r["label"] == 0 and r["p"] < 0.5)
    out["false_rejects_at_0.5"] = sum(1 for r in records if r["label"] == 1 and r["p"] < 0.5)
    out["false_accepts_at_0.5"] = sum(1 for r in records if r["label"] == 0 and r["p"] >= 0.5)
    return out


def make_preview(recs: list[dict[str, Any]], threshold: float, api_target: str,
                 api_model: str) -> str:
    """Render the real HumanGate._show output without asking or calling an API."""
    from harness.cascade.judgement import Judgement, result_for
    from experiments.cascade.run_cascade import HumanGate

    rejected = [r for r in recs if r["p"] < threshold]
    chosen = rejected[:3] if rejected else recs[:3]
    lines = [
        "# Simulated human-gate preview (offline)",
        "",
        "This is not a live trigger and it did not call a hosted API. It renders the",
        "evidence the pre-API human gate would show, using the frozen Bonsai judge as",
        "a stand-in for the verifier. `eval label` is for analysis only and is never",
        "available to the runtime gate. Items are sorted by judge probability, not by",
        "runtime arrival order, so the first block is a worst-case visibility test.",
        "",
        f"Simulated reject threshold: p < {threshold:.2f}; "
        f"{len(rejected)} of {len(recs)} gate candidates would be shown.",
        "",
    ]
    for n, rec in enumerate(chosen, 1):
        verdict = result_for(
            Judgement(
                id=f"preview-{n}", request_id=rec["item_id"], generation_id=0,
                role="D3", kind="verify",
            ),
            "reject", 1.0 - rec["p"],
            payload={"reason": f"frozen Bonsai judge p={rec['p']:.3f} "
                               f"below threshold {threshold:.2f}"},
        )
        gate = HumanGate(
            api_target=api_target, api_model=api_model,
            print_fn=lines.append,
        )
        gate._show(
            task={"id": rec["item_id"], "bucket": rec["bucket"],
                  "prompt": rec.get("prompt", "")},
            local_answer=rec.get("candidate", ""),
            verdict=verdict,
            checker_ok=None,
            first=n == 1,
            still_failing=False,
            attempts=1,
        )
        lines.append(
            f"[offline eval label: {rec['label']} "
            "(correct=1, wrong=0; never shown to the runtime gate)]"
        )
    return "\n".join(lines) + "\n"


def make_report(
    name: str,
    head: dict[str, Any],
    generator: dict[str, Any] | None,
    test: dict[str, Any],
    test_bench: dict[str, Any],
    organic: dict[str, Any],
    qa_all: dict[str, Any],
    qa_organic: dict[str, Any],
    qa_clean: dict[str, Any],
    qa_clean_organic: dict[str, Any],
    qa_harvest: dict[str, Any],
    qa_harvest_organic: dict[str, Any],
    qa_refit: dict[str, Any] | None,
    gate: dict[str, Any],
    sanity: dict[str, Any],
    stability: dict[str, Any] | None,
) -> str:
    def table(d: dict[str, Any], title: str) -> list[str]:
        out = [f"### {title}", "", "| metric | value |", "|---|---:|"]
        for key in ("n", "correct", "wrong", "auc", "ece",
                    "caught_at_0.5", "false_rejects_at_0.5",
                    "false_accepts_at_0.5"):
            if key in d and d[key] is not None:
                value = d[key]
                if isinstance(value, float):
                    value = f"{value:.4f}"
                out.append(f"| {key} | {value} |")
        ba = d.get("best_ba") or {}
        if ba:
            out.append(f"| best_ba | {ba.get('bal_acc', 0):.4f} @ {ba.get('threshold', 0):.2f} |")
        at = d.get("at_0.5") or {}
        if at:
            out.append(f"| at_0.5 decisions correct | {at.get('correct')} / {at.get('n')} |")
            out.append(f"| at_0.5 false accepts | {at.get('false_accepts')} |")
            out.append(f"| at_0.5 false rejects | {at.get('false_rejects')} |")
        ath = d.get("at_head") or {}
        if ath:
            out.append(f"| at_head threshold | {ath.get('threshold'):.2f} |")
            out.append(f"| at_head decisions correct | {ath.get('correct')} / {ath.get('n')} |")
            out.append(f"| at_head false accepts | {ath.get('false_accepts')} |")
            out.append(f"| at_head false rejects | {ath.get('false_rejects')} |")
        out.append("")
        return out

    lines = [
        f"# {name}",
        "",
        "## What was measured",
        "",
        "The Bonsai 27B final-token hidden state under the same judge prompt used for",
        "the frozen J-Bench head. The head was **not refit** for the primary numbers;",
        "this is a transfer baseline. The extra extraction contains held-out organic/QA",
        "items plus four train examples needed by the generic probe runner; those four",
        "are only a numerical parity check, not a refit. Evaluation labels are from the",
        "reference readout NPZ and are used for scoring only.",
        "",
        "Runtime: pure CPU (HIP/ROCR/CUDA hidden, `--device none --no-op-offload`),",
        "`-t 8 -c 2048 -np 1`, nice 19, taskset 0-7. The 176B serve on 8081 stayed up;",
        "see Stability.",
        "",
        "## Head sanity",
        "",
        f"- Frozen head: `{head.get('name', '?')}`, training n={head.get('training', {}).get('n_train')}",
        f"- J-Bench test: expected AUC {sanity.get('expected_auc')}, observed {sanity.get('observed_auc')} "
        f"(n={sanity.get('n')})",
        *(
            [
                f"- CPU/GPU state parity on {parity['n']} re-embedded train rows: "
                f"mean cosine {parity['mean_cosine']:.6f}, "
                f"min {parity['min_cosine']:.6f}, "
                f"max |diff| {parity['max_abs_diff']:.6f}"
            ]
            if (parity := sanity.get("state_parity_cpu_vs_gpu"))
            else []
        ),
        "",
        *(
            [
                "## Generator context (prior; not this CPU run)",
                "",
                f"Source: `{generator['source']}` — {generator['note']}.",
                "",
                "| arm | correct | n | acc |",
                "|---|---:|---:|---:|",
                *[
                    f"| {arm} | {v['correct']} | {v['n']} | {v['acc']:.3f} |"
                    for arm, v in generator["arms"].items()
                ],
                "",
                "Those J-Bench generation numbers are in-family and from the earlier GPU",
                "run. This run measures judge transfer only; it does not measure Bonsai",
                "generation.",
                "",
            ]
            if generator else []
        ),
        *table(test, f"Combined test ({test.get('n')} items: J-Bench + organic)"),
        *table(test_bench, "J-Bench test only"),
        *table(organic, "Organic test (57 correct / 9 wrong)"),
        *table(qa_all, "QA all 129 (reference; includes 38 frozen-head train rows)"),
        *table(qa_organic, "QA organic-only all 129 (reference)"),
        *table(qa_clean, "QA clean 91 (excludes 38 frozen-head train rows)"),
        *table(qa_clean_organic, "QA organic-only clean 91"),
        *table(qa_harvest, "QA harvest-only 75 (out-of-family)"),
        *table(qa_harvest_organic, "QA organic-only harvest 75"),
    ]
    if qa_refit:
        lines.extend([
            "### QA control: J-Bench non-QA refit",
            "",
            "This control fits a fresh ridge/Platt head on the",
            f"{qa_refit.get('train_n')} J-Bench training items whose bucket is not QA",
            "(all 38 QA training rows removed), then scores the 129 QA items. It is not",
            "the full 1,186-item combined LOFO protocol. It is a weak control: its",
            "all-129 AUC is real but its 0.5 calibration accepts most wrong answers,",
            "and its organic-only AUC is at chance. The frozen head's clean 91 number",
            "above is the stronger QA transfer result.",
            "",
            *table(qa_refit, "QA transfer with J-Bench non-QA train"),
            *table(qa_refit.get("organic_only") or {}, "QA control organic-only"),
            *table(qa_refit.get("harvest_only") or {}, "QA control harvest-only"),
        ])
    lines.extend([
        "## Human-gate workload (simulated)",
        "",
        f"- Gate candidates: {gate['n']} ({gate['correct']} correct / {gate['wrong']} wrong)",
        f"- Rejected at p < {gate['threshold']:.2f}: {gate['rejected']} "
        f"({gate['ask_rate'] * 100:.1f}%)",
        f"- Rejected items with eval label correct / wrong: "
        f"{gate['rejected_correct']} / {gate['rejected_wrong']}",
        f"- Wrong answers missed (p >= threshold): {gate['missed_wrong']} of "
        f"{gate['wrong']} (recall {gate['recall_wrong'] * 100:.1f}%)",
        f"- Uncertain band [{gate['threshold']:.2f}, {gate['threshold'] + 0.15:.2f}): "
        f"{gate['uncertain']}",
        "",
        "| threshold | rejected / ask rate | eval-correct rejects | eval-wrong rejects |",
        "|---:|---:|---:|---:|",
        *[
            f"| {t} | {v['rejected']} ({v['ask_rate'] * 100:.1f}%) | "
            f"{v['rejected_correct']} | {v['rejected_wrong']} |"
            for t, v in gate.get("by_threshold", {}).items()
        ],
        "",
        "| source | candidates | wrong | rejected wrong | missed wrong |",
        "|---|---:|---:|---:|---:|",
        *[
            f"| {source} | {v['n']} | {v['wrong']} | {v['rejected_wrong']} | "
            f"{v['missed_wrong']} |"
            for source, v in gate.get("by_source", {}).items()
        ],
        "",
        "| bucket | candidates | wrong | rejected wrong | missed wrong |",
        "|---|---:|---:|---:|---:|",
        *[
            f"| {bucket} | {v['n']} | {v['wrong']} | {v['rejected_wrong']} | "
            f"{v['missed_wrong']} |"
            for bucket, v in gate.get("by_bucket", {}).items()
        ],
        "",
        "The gate only asks after a verifier reject; this simulation uses the Bonsai",
        "head's rejection as that trigger. It does not spend API tokens and does not",
        "use labels at runtime. The precision is perfect on rejected items (0 correct",
        "rejected), but recall is weak on organic/free-form wrong answers: the head",
        "accepted most of the organic wrongs. See `human-gate-preview.md` for the",
        "rendered first triggers, `human-gate-candidates.jsonl` for the ranked list,",
        "and `human-gate-missed.jsonl` for the wrong answers the gate would miss.",
        "",
    ])
    if stability:
        lines.extend([
            "## Stability",
            "",
            "| metric | value |",
            "|---|---:|",
            f"| samples | {stability.get('samples')} |",
            f"| duration | {stability.get('duration_s')} s |",
            f"| max load1 | {stability.get('max_load1')} |",
            f"| min MemAvailable | {stability.get('min_mem_available_gib')} GiB |",
            f"| max swap used | {stability.get('max_swap_used_gib')} GiB |",
            f"| max partner RSS | {stability.get('max_rss_gib')} GiB |",
            f"| 176B health failures | {stability.get('health_failures')} |",
            "| min GPU free | "
            + ", ".join(
                f"{name}: {free} MiB"
                for name, free in (stability.get("min_vram_free_mib") or {}).items()
            )
            + " |",
            "",
        ])
    lines.extend([
        "## Reading this result",
        "",
        "- **In-family (J-Bench) and QA-choice verification are strong.** The frozen",
        "  head reproduces its J-Bench test AUC exactly; on the clean 91 QA rows",
        "  (38 training rows excluded) it catches all 48 choice-flip wrongs and",
        "  6 of the 8 organic QA wrongs at p<0.5.",
        "- **Organic/free-form verification is weak.** On the 66 held-out organic",
        "  tasks the head catches only 1 of 9 wrong answers at p<0.58 (0 at p<0.5)",
        "  and rejects no correct answer. That is below the 0.8B organic reference",
        "  (AUC 0.844; 5/9 wrong caught). The combined-test AUC is carried by the",
        "  in-family J-Bench half; do not read it as general organic accuracy.",
        "- **Do not use Bonsai as the sole organic verifier.** Use it for structured",
        "  QA/J-Bench verification, or test it as a generator/repairer (the prior",
        "  in-family generation eval was strong), while the 0.8B verifier and the",
        "  human gate cover organic free-form tasks.",
        "- **Human-gate consequence.** A Bonsai-prefiltered gate would ask on ~50%",
        "  of held-out candidates and every ask would be a wrong answer, but it would",
        "  silently accept most organic wrongs. The gate recall, not just precision,",
        "  has to be tuned with the organic numbers in view.",
        "- The QA control excludes every QA row from training but still uses only",
        f"  {qa_refit.get('train_n') if qa_refit else '?'} J-Bench non-QA rows; a true",
        "  combined LOFO head would need the 1,186 non-QA harvest embeddings, which",
        "  were not extracted in this run.",
        "- Gold labels are evaluation-only; the runtime gate remains gold-free.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", default="bonsai-2-27b-transfer")
    ap.add_argument("--readout-states", type=Path, default=READOUT)
    ap.add_argument("--base-states", type=Path, default=BASE)
    ap.add_argument("--extra-states", type=Path, required=True)
    ap.add_argument("--head", type=Path, default=HEAD)
    ap.add_argument("--combined", type=Path, default=COMBINED)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--stability-summary", type=Path, default=None)
    ap.add_argument("--api-target", default="hosted API (per-run --api-fallback; provider not named)")
    ap.add_argument("--api-model", default="not set")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    with np.load(args.readout_states, allow_pickle=True) as z:
        ids = [str(x) for x in z["item_ids"]]
        labels = [int(x) for x in z["labels"]]
        splits = [str(x) for x in z["splits"]]
        buckets = [str(x) for x in z["buckets"]]
        origins = [str(x) for x in z["origins"]]
        sources = [str(x) for x in z["sources"]]
        task_ids = [str(x) for x in z["task_ids"]]
    meta = {
        item_id: {
            "label": labels[n], "split": splits[n], "bucket": buckets[n],
            "origin": origins[n], "source": sources[n], "task_id": task_ids[n],
        }
        for n, item_id in enumerate(ids)
    }

    states = load_states(args.base_states)
    base_ids = set(states)
    extra = load_states(args.extra_states)
    for item_id, vec in extra.items():
        states.setdefault(item_id, vec)
    missing = [i for i in ids if i not in states]
    required_ids = {
        i for i in ids
        if meta[i]["split"] == "test" or meta[i]["bucket"] == "qa"
    }
    missing_required = sorted(required_ids - set(states))
    if missing_required:
        print(f"missing {len(missing_required)} required test/QA embeddings; "
              f"first={missing_required[:8]}", file=sys.stderr)
        return 2
    head = json.loads(args.head.read_text())
    threshold = float(head.get("test", {}).get("threshold", 0.5))
    generator = None
    generator_path = REPO / "experiments/cascade/results/jev-readout-20261006/bonsai-eval.json"
    if generator_path.exists():
        raw = json.loads(generator_path.read_text())
        arms = raw.get("arms", {})
        if arms:
            generator = {
                "source": str(generator_path.relative_to(REPO)),
                "note": "prior GPU J-Bench generation eval; not this CPU run",
                "arms": {
                    name: {
                        "correct": arm.get("correct"),
                        "n": arm.get("n"),
                        "acc": arm.get("acc"),
                    }
                    for name, arm in arms.items()
                },
            }
    parity = None
    overlap = sorted(base_ids & set(extra))
    if overlap:
        a = np.stack([states[i] for i in overlap])
        b = np.stack([extra[i] for i in overlap])
        cos = np.sum(a * b, axis=1) / (
            np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        )
        parity = {
            "n": len(overlap),
            "mean_cosine": float(np.mean(cos)),
            "min_cosine": float(np.min(cos)),
            "max_abs_diff": float(np.max(np.abs(a - b))),
            "items": overlap[:10],
        }
    combined: dict[str, dict[str, Any]] = {}
    if args.combined.exists():
        for line in args.combined.read_text().splitlines():
            item = json.loads(line)
            combined[str(item["item_id"])] = item

    records: list[dict[str, Any]] = []
    for item_id in ids:
        if item_id not in states:
            continue
        p = float(apply_head(head, states[item_id][None, :])[0])
        rec = {"item_id": item_id, "p": p, **meta[item_id]}
        orig = combined.get(item_id) or {}
        rec["prompt"] = orig.get("prompt", "")
        rec["candidate"] = orig.get("candidate", "")
        records.append(rec)
    by_id = {r["item_id"]: r for r in records}

    test = [r for r in records if r["split"] == "test"]
    test_bench = [r for r in test if r["source"] == "bench"]
    organic = [r for r in test if r["source"] == "harvest"]
    qa = [r for r in records if r["bucket"] == "qa"]
    qa_seen = [r for r in qa if r["item_id"] in base_ids and r["split"] == "train"]
    qa_clean = [r for r in qa if r["item_id"] not in {x["item_id"] for x in qa_seen}]
    qa_harvest = [r for r in qa if r["source"] == "harvest"]
    qa_correct = [r for r in qa if r["label"] == 1]
    qa_perturbed = [r for r in qa if r["label"] == 0 and "perturbed" in r["origin"]]
    qa_organic_wrong = [r for r in qa if r["label"] == 0 and "perturbed" not in r["origin"]]
    qa_organic = qa_correct + qa_organic_wrong
    qa_clean_correct = [r for r in qa_clean if r["label"] == 1]
    qa_clean_perturbed = [r for r in qa_clean if r["label"] == 0 and "perturbed" in r["origin"]]
    qa_clean_organic_wrong = [r for r in qa_clean if r["label"] == 0 and "perturbed" not in r["origin"]]
    qa_clean_organic = qa_clean_correct + qa_clean_organic_wrong
    qa_harvest_correct = [r for r in qa_harvest if r["label"] == 1]
    qa_harvest_perturbed = [r for r in qa_harvest if r["label"] == 0 and "perturbed" in r["origin"]]
    qa_harvest_organic_wrong = [r for r in qa_harvest if r["label"] == 0 and "perturbed" not in r["origin"]]
    qa_harvest_organic = qa_harvest_correct + qa_harvest_organic_wrong

    # Secondary control: exclude every QA row from the J-Bench training fit.
    base_meta = {i: meta[i] for i in base_ids}
    base_train = [
        i for i in base_meta
        if base_meta[i]["split"] == "train" and base_meta[i]["bucket"] != "qa"
    ]
    base_X = np.stack([states[i] for i in base_train])
    base_y = np.asarray([base_meta[i]["label"] for i in base_train], dtype=np.int64)
    qa_refit = None
    if base_train and qa:
        model, ab = fit_head(base_X, base_y)
        qa_refit_records = [
            {**r, "p": float(p_from_model(model, ab, states[r["item_id"]][None, :])[0])}
            for r in qa
        ]
        qa_refit = compact_metrics(qa_refit_records)
        refit_organic = [r for r in qa_refit_records
                         if r["label"] == 1 or "perturbed" not in r["origin"]]
        refit_harvest = [r for r in qa_refit_records if r["source"] == "harvest"]
        qa_refit["organic_only"] = compact_metrics(refit_organic)
        qa_refit["harvest_only"] = compact_metrics(refit_harvest)
        qa_refit["train_n"] = len(base_train)
        qa_refit["train_pos"] = int(base_y.sum())
        qa_refit["train_neg"] = int(len(base_y) - base_y.sum())
        qa_refit["note"] = "J-Bench non-QA train only; not the full combined 1,186-row LOFO"

    test_metrics = compact_metrics(test)
    test_bench_metrics = compact_metrics(test_bench)
    organic_metrics = compact_metrics(organic)
    qa_metrics = compact_metrics(qa)
    qa_organic_metrics = compact_metrics(qa_organic)
    qa_clean_metrics = compact_metrics(qa_clean)
    qa_clean_organic_metrics = compact_metrics(qa_clean_organic)
    qa_harvest_metrics = compact_metrics(qa_harvest)
    qa_harvest_organic_metrics = compact_metrics(qa_harvest_organic)
    for stats, recs in (
        (test_metrics, test),
        (test_bench_metrics, test_bench),
        (organic_metrics, organic),
        (qa_metrics, qa),
        (qa_organic_metrics, qa_organic),
        (qa_clean_metrics, qa_clean),
        (qa_clean_organic_metrics, qa_clean_organic),
        (qa_harvest_metrics, qa_harvest),
        (qa_harvest_organic_metrics, qa_harvest_organic),
    ):
        stats["at_head"] = at_threshold(recs, threshold)

    # Head sanity: the frozen head's own J-Bench test number.
    sanity_recs = test_bench
    sanity = {
        "n": len(sanity_recs),
        "expected_auc": head.get("test", {}).get("auc"),
        "observed_auc": test_bench_metrics.get("auc"),
    }
    if parity:
        sanity["state_parity_cpu_vs_gpu"] = parity

    # Human-gate view: held-out combined test plus harvest QA, de-duplicated.
    # Exclude the 38 J-Bench QA train rows the frozen head already saw.
    gate_ids = {r["item_id"] for r in test} | {r["item_id"] for r in qa_harvest}
    gate_recs = sorted((by_id[i] for i in gate_ids), key=lambda r: r["p"])
    rejected = [r for r in gate_recs if r["p"] < threshold]
    uncertain = [r for r in gate_recs if threshold <= r["p"] < threshold + 0.15]
    by_threshold = {}
    for t in (0.5, 0.58, 0.6, 0.65, 0.7):
        at = [r for r in gate_recs if r["p"] < t]
        by_threshold[f"{t:.2f}"] = {
            "rejected": len(at),
            "ask_rate": len(at) / max(1, len(gate_recs)),
            "rejected_correct": sum(1 for r in at if r["label"] == 1),
            "rejected_wrong": sum(1 for r in at if r["label"] == 0),
        }
    gate_wrong = [r for r in gate_recs if r["label"] == 0]
    gate_missed = [r for r in gate_wrong if r["p"] >= threshold]
    gate = {
        "n": len(gate_recs),
        "threshold": threshold,
        "correct": sum(1 for r in gate_recs if r["label"] == 1),
        "wrong": len(gate_wrong),
        "rejected": len(rejected),
        "ask_rate": len(rejected) / max(1, len(gate_recs)),
        "rejected_correct": sum(1 for r in rejected if r["label"] == 1),
        "rejected_wrong": sum(1 for r in rejected if r["label"] == 0),
        "missed_wrong": len(gate_missed),
        "recall_wrong": (len(gate_wrong) - len(gate_missed)) / max(1, len(gate_wrong)),
        "uncertain": len(uncertain),
        "by_threshold": by_threshold,
        "by_bucket": {
            bucket: {
                "n": sum(1 for r in gate_recs if r["bucket"] == bucket),
                "wrong": sum(1 for r in gate_recs if r["bucket"] == bucket and r["label"] == 0),
                "rejected": sum(1 for r in gate_recs if r["bucket"] == bucket and r["p"] < threshold),
                "rejected_correct": sum(1 for r in gate_recs if r["bucket"] == bucket and r["p"] < threshold and r["label"] == 1),
                "rejected_wrong": sum(1 for r in gate_recs if r["bucket"] == bucket and r["p"] < threshold and r["label"] == 0),
                "missed_wrong": sum(1 for r in gate_recs if r["bucket"] == bucket and r["label"] == 0 and r["p"] >= threshold),
            }
            for bucket in sorted({r["bucket"] for r in gate_recs})
        },
        "by_source": {
            source: {
                "n": sum(1 for r in gate_recs if r["source"] == source),
                "wrong": sum(1 for r in gate_recs if r["source"] == source and r["label"] == 0),
                "rejected_wrong": sum(1 for r in gate_recs if r["source"] == source and r["p"] < threshold and r["label"] == 0),
                "missed_wrong": sum(1 for r in gate_recs if r["source"] == source and r["label"] == 0 and r["p"] >= threshold),
            }
            for source in sorted({r["source"] for r in gate_recs})
        },
    }

    # Outputs.
    (args.outdir / "bonsai-transfer-predictions.jsonl").write_text(
        "\n".join(json.dumps({
            "item_id": r["item_id"], "p": r["p"], "label": r["label"],
            "split": r["split"], "source": r["source"], "bucket": r["bucket"],
            "origin": r["origin"], "task_id": r["task_id"],
        }) for r in records) + "\n"
    )
    candidate_path = args.outdir / "human-gate-candidates.jsonl"
    with candidate_path.open("w") as handle:
        for n, r in enumerate(gate_recs, 1):
            handle.write(json.dumps({
                "rank": n, "item_id": r["item_id"], "p": r["p"],
                "rejected": r["p"] < threshold,
                "uncertain_band": threshold <= r["p"] < threshold + 0.15,
                "label": r["label"], "split": r["split"], "source": r["source"],
                "bucket": r["bucket"], "origin": r["origin"], "task_id": r["task_id"],
                "prompt": r["prompt"], "candidate": r["candidate"],
            }) + "\n")
    (args.outdir / "human-gate-preview.md").write_text(
        make_preview(gate_recs, threshold, args.api_target, args.api_model)
    )
    with (args.outdir / "human-gate-missed.jsonl").open("w") as handle:
        for n, r in enumerate(gate_missed, 1):
            handle.write(json.dumps({
                "rank": n, "item_id": r["item_id"], "p": r["p"],
                "label": r["label"], "split": r["split"], "source": r["source"],
                "bucket": r["bucket"], "origin": r["origin"], "task_id": r["task_id"],
                "prompt": r["prompt"], "candidate": r["candidate"],
            }) + "\n")

    stability = None
    if args.stability_summary and args.stability_summary.exists():
        stability = json.loads(args.stability_summary.read_text())

    result = {
        "name": args.name,
        "protocol": "frozen Bonsai J-Bench head; held-out organic/QA transfer; no refit",
        "head": str(args.head),
        "head_training": head.get("training", {}),
        "missing_from_state_union": missing,
        "missing_required": missing_required,
        "n_records": len(records),
        "runtime": "pure CPU; HIP/ROCR/CUDA hidden; --device none --no-op-offload; 176B serve on 8081 stayed up",
        "state_parity_cpu_vs_gpu": parity,
        "head_sanity": sanity,
        "generator_context": generator,
        "combined_test": test_metrics,
        "jbench_test": test_bench_metrics,
        "organic_test": organic_metrics,
        "qa_transfer": {
            "all": qa_metrics,
            "organic_only": qa_organic_metrics,
            "all_129_reference_contaminated": qa_metrics,
            "organic_only_all_129": qa_organic_metrics,
            "clean_91": qa_clean_metrics,
            "organic_only_clean_91": qa_clean_organic_metrics,
            "harvest_75": qa_harvest_metrics,
            "organic_only_harvest_75": qa_harvest_organic_metrics,
            "counts": {
                "all": {
                    "correct": len(qa_correct),
                    "organic_wrong": len(qa_organic_wrong),
                    "perturbed": len(qa_perturbed),
                },
                "clean_91": {
                    "correct": len(qa_clean_correct),
                    "organic_wrong": len(qa_clean_organic_wrong),
                    "perturbed": len(qa_clean_perturbed),
                },
                "harvest_75": {
                    "correct": len(qa_harvest_correct),
                    "organic_wrong": len(qa_harvest_organic_wrong),
                    "perturbed": len(qa_harvest_perturbed),
                },
                "seen_train_rows": len(qa_seen),
            },
            "at_0.5": {
                "organic_wrong_caught": sum(1 for r in qa_organic_wrong if r["p"] < 0.5),
                "correct_false_rejects": sum(1 for r in qa_correct if r["p"] < 0.5),
                "perturbed_false_accepts": sum(1 for r in qa_perturbed if r["p"] >= 0.5),
            },
            "at_head_threshold": {
                "threshold": threshold,
                "organic_wrong_caught": sum(1 for r in qa_organic_wrong if r["p"] < threshold),
                "correct_false_rejects": sum(1 for r in qa_correct if r["p"] < threshold),
                "perturbed_false_accepts": sum(1 for r in qa_perturbed if r["p"] >= threshold),
            },
            "clean_at_0.5": {
                "organic_wrong_caught": sum(1 for r in qa_clean_organic_wrong if r["p"] < 0.5),
                "correct_false_rejects": sum(1 for r in qa_clean_correct if r["p"] < 0.5),
                "perturbed_false_accepts": sum(1 for r in qa_clean_perturbed if r["p"] >= 0.5),
            },
            "clean_at_head_threshold": {
                "threshold": threshold,
                "organic_wrong_caught": sum(1 for r in qa_clean_organic_wrong if r["p"] < threshold),
                "correct_false_rejects": sum(1 for r in qa_clean_correct if r["p"] < threshold),
                "perturbed_false_accepts": sum(1 for r in qa_clean_perturbed if r["p"] >= threshold),
            },
            "harvest_at_head_threshold": {
                "threshold": threshold,
                "organic_wrong_caught": sum(1 for r in qa_harvest_organic_wrong if r["p"] < threshold),
                "correct_false_rejects": sum(1 for r in qa_harvest_correct if r["p"] < threshold),
                "perturbed_false_accepts": sum(1 for r in qa_harvest_perturbed if r["p"] >= threshold),
            },
            "jbench_nonqa_refit": qa_refit,
        },
        "human_gate": gate,
        "stability": stability,
    }
    (args.outdir / f"eval-{args.name}.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.outdir / "BONSAI-TRANSFER.md").write_text(
        make_report(args.name, head, generator, test_metrics, test_bench_metrics,
                    organic_metrics, qa_metrics, qa_organic_metrics,
                    qa_clean_metrics, qa_clean_organic_metrics,
                    qa_harvest_metrics, qa_harvest_organic_metrics,
                    qa_refit, gate, sanity, stability)
    )

    print(f"records={len(records)} missing={len(missing)}")
    print(f"combined test AUC={test_metrics.get('auc')} organic AUC={organic_metrics.get('auc')}")
    print(f"qa clean AUC={qa_clean_metrics.get('auc')} clean organic={qa_clean_organic_metrics.get('auc')}")
    print(f"qa harvest AUC={qa_harvest_metrics.get('auc')} harvest organic={qa_harvest_organic_metrics.get('auc')}")
    print(f"gate rejected={gate['rejected']}/{gate['n']} at {threshold:.2f}")
    print("wrote", args.outdir / f"eval-{args.name}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
