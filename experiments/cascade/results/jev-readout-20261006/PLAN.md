# JEV-readout test plan (2026-10-06)

Question: for the cascade verifier role, what is the **smallest footprint** that
answers "is this candidate correct?" well enough — and does it have to be an
independent model at all?

Frozen set: `judge-bench.jsonl` (378 items: 97 pos / 281 neg; split 68/164
train, 29/87 test; sha256 `3289dcb8...`; all items checker-verified).  The test
split is frozen; fitting uses train only.  Prior art: `PRIOR-ART.md`.

## Experiments

| # | test | readout | status |
|---|---|---|---|
| T1a | generator judges J-Bench (self) | prompted (yes/no logits, both orders) | queued (Scion-35B session) |
| T1b | generator judges J-Bench (self) | final-layer linear probe | queued (bridge hidden dump) |
| T1c | independent model judges J-Bench | prompted + probe | running (Qwen3.5-4B / 0.8B; Intern/JEV later) |
| T1d | self-preference | Scion's own vs Strata-answer accept rates | queued (needs Scion answers on the test split) |
| T2 | size/label curve | probe across 0.8B / 1.7B / 4B / 9B, labels 20/100/300 | 0.8B smoke: AUC 0.973 (final), 0.898 (mid) on 120 items |
| T3 | ternary survival (novel) | probe fitted on bf16 states -> ternary states; refit | blocked on mid-layer extraction for ternary GGUF |
| T4 | calibration + cascade end-to-end | threshold/no-leak point, then `run_cascade.py` | after a candidate passes |

## Gates

- G1: if T1a/T1b (generator self) match T1c within 3 points BA and no worse
  false-accepts -> no independent model needed.
- G2: else the smallest independent base + probe that clears the bar wins
  (literature predicts ~4B-class).
- G3: if a sub-1B model must be quantised for residency, T3 decides whether the
  readout survives.

## Results (2026-10-06, frozen test split unless noted)

Headline table — see `COMPARISON.md` for the live version:

| readout | kind | test AUC | best BA | note |
|---|---|---|---|---|
| Intern-4B | trained judge | 0.979 | 0.937 | the quality bar (purpose-trained) |
| **Scion-35B final-layer probe** | probe | **0.999** | **0.994** | the generator itself; no-release threshold 0.95 -> acc 0.974 |
| Qwen3.5-4B final-layer probe | probe | 0.960 | 0.937 | base model + fitted head matches the trained 4B |
| Scion-35B prompted self-judge | prompted | 0.903 (test) | 0.925 | instruct model; order-flip 7.4%; better than trained 0.8B/1.7B |
| Qwen3.5-0.8B mid-layer probe | probe | 0.924 | 0.856 | beats every small trained judge |
| Intern-2B | trained judge | 0.932 | 0.856 | |
| Tiny-Jev-1.7B | trained judge | 0.890 | 0.862 | |
| Intern-0.8B | trained judge | 0.774 | 0.759 | |
| Qwen3.5-4B prompted | prompted | 0.494 | 0.546 | base model: 82.8% option-order flips (label bias) |

Other measurements:

- **Cross-type transfer** (4B probe, depth 0.75, fit on number-flips only):
  boxed 0.998, code 0.874 — a general correctness signal, not perturbation
  detection.
- **Natural negatives**: Scion generated its own answers for all 100 bench
  tasks (`scion-answers.json`): **72/100 correct -> 28 natural errors**, five
  times the natural negatives in the recorded runs (3).
- **Prompted self-preference (paired 100 tasks)**: Scion's prompted judgment of
  **its own answers is near-chance as a discriminator (AUC 0.559, 21/28 wrong
  accepted)**, while the same prompt on Strata's answers scores AUC 0.371 on
  only 3 available negatives (uninformative).  The probe-vs-prompted gap on
  the same model is the story: 0.999 vs 0.559.
- **Frozen-head source-swap audit (done, `source-audit.json`)**: the head
  fitted on Strata-answer train items, applied frozen to:
  - Scion's **own** answers: AUC **0.971** over the same 100 tasks including
    **28 natural errors** (mean score 1.04 correct vs 0.22 wrong) — the
    correctness signal generalizes beyond perturbations;
  - Strata's answers: AUC 0.990 (97 correct / 3 wrong).
  Self-preference deltas: **correct +0.078** (own above other), **wrong +0.018**
  — a small familiarity shift on correct answers, **no inflation of own
  errors**.  The controlled attribution test (same text, "your answer" vs
  "another model's") remains as the next audit.

### T2 size / label map (probe at depth 0.75, test AUC)

| base | 20 labels | 100 labels | full train (262) |
|---|---|---|---|
| Qwen3.5-0.8B | 0.784 | 0.831 | 0.924 |
| Qwen3-1.7B | — | — | **0.988** |
| Qwen3.5-4B | 0.959 | 0.962 | 0.946 |

The 1.7B is the surprise: its mid-layer probe beats the 4B and matches the
purpose-trained Intern-4B (0.979), with a usable no-false-accept point
(thr 0.94 -> acc 0.948).  The 4B is label-saturated at ~20 (0.959 vs Tiny-Jev
0.890 / Intern-2B 0.932).  The 0.8B keeps scaling with labels.



## Status and next steps

- Done: J-Bench; the full readout/judge/probe matrix; Scion-35B session
  (prompted, probe, 100 generated answers); source-swap audit incl. natural
  errors; T2 size/label map.
- Running: T3 ternary survival (Clef f16 / h128 / QAT embeddings, cross-body
  transfer).
- Next: controlled attribution test (same text, own vs other), calibration
  (threshold/no-false-accept points already reported), one cascade end-to-end
  run, whitepaper outline.



