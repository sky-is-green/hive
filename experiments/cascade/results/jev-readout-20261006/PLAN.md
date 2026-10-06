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

## Smoke results so far

- `readout-smoke-qwen35-0.8b.json`: prompted readout, 6 test items, AUC 0.375
  (sanity generation answers "A. Yes" — prompt format is understood).
- `probe-smoke0.8b.json`: ridge probe on the first 120 items, test AUC 0.973
  (final layer) / 0.898 (mid layer) — the model's hidden state carries the
  verdict even when its prompted answer does not.
