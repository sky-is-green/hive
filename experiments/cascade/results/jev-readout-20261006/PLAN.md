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

## Results so far (2026-10-06)

| readout | model | test AUC | note |
|---|---|---|---|
| prompted yes/no | Qwen3.5-0.8B | 0.375 (n=6) | smoke only |
| prompted yes/no | Qwen3.5-4B | 0.494 (n=116) | **82.8% option-order flips** — dominated by a label/position bias |
| probe, depth 0.75 | Qwen3.5-0.8B | 0.924 | best layer for the 0.8B |
| probe, final | Qwen3.5-0.8B | 0.795 | signal degrades at the last layer here |
| probe, depth 0.75 | Qwen3.5-4B | 0.946 | nearly coincident with final |
| probe, final | Qwen3.5-4B | 0.960 | best overall |
| probe (4B, 0.75), cross-type transfer | fit on num-flips only | boxed 0.998, code 0.874 | a general correctness signal, not perturbation detection |
| probe (4B, 0.75), per negative type | entity 1.00, boxed 0.99, code 0.91, num 0.98, choice 0.81-0.86 (n=4) | | choice items underpowered |

Caveats: `Qwen/Qwen3.5-0.8B` and `Qwen/Qwen3.5-4B` are base
(`Qwen3_5ForConditionalGeneration`) checkpoints without an instruct tag, so the
prompted leg is **not** a fair instruction-tuned comparison — the trained
judges (Intern family, Tiny-Jev, JEV-9B) and the Scion-35B generator session
supply that leg.  All negatives are still perturbations; natural generator
errors are added in the Scion session.

## Status

- `jev-small` chain: done (probe 0.8B/4B, prompted 4B).
- `jev-judges` chain: running (Intern 4B/2B/0.8B on GPU, Tiny-Jev on CPU).
- queued: Scion-35B generator session (prompted self-judge, final-layer probe
  via the bridge, natural wrong answers on the test split); ternary survival
  probe once a mid-layer hidden path exists for the V2/QAT GGUFs.

