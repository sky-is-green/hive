# Gate replay (M3): the production scorer over committed artifacts

2026-10-18, offline (no serve load, no runner re-run — the drafts are the
committed ones).  The question this answers: does the *wired* gate
(`harness/cascade/learned_gate.py` behind the runner's `--gate`, frozen head
L0.75 at threshold 0.5) behave as the plan promised, on data it has never
seen, and what would it have done inside the two committed arm runs?

Machine: `.venv/bin/python experiments/cascade/gate_replay.py --out
experiments/cascade/results/gate-replay-20261018` — one model load, then
three scorings; per-item results in `replay.json`.

## Part 1 — the harvest-66 reproduction (the study body)

All 66 stored organic drafts re-extracted live (intern-decision-0.8B, CPU,
the same row builder the heads were fitted on) and scored through the
production heads:

- **max |Δp| vs the stored fitted p: 0.0 for both L0.75 and L0.5 across all
  66 items.**  The extraction path + head math in the harness reproduce the
  committed artifact bit-for-bit — the analysis numbers are the production
  numbers, not a re-implementation.
- Deterministic detectors re-run on the same drafts: **6/9 wrongs caught,
  0 false flags** — same as committed.
- The frozen cut reproduces the operating table: **L0.75: +1 catch,
  0 flags** (`gsm8k-0015:hs2`, p 0.073751); the recall dial **L0.5: +3
  catches, 9 flags** (`gsm8k-0009:hs2` included at p 0.4255, `math-0020:hs2`
  at 0.00271).  Both AUCs land in the committed band (0.8187 / 0.8265).
- Scoring cost: 66 items in ≈ a few seconds each, matching the 0.44 s/item
  extraction median (cold-start overhead aside).

## Part 2 — the two committed arm runs, replayed

Blind-spot-only, exactly the runner's rule: `detect()` on every served
draft; truncated drafts excluded (the runner rejects them by ceiling and
never gates them); only silent drafts go through the gate.  Labels are the
committed row truth (`scion_ok` — eval-only).

| arm | rows | detector-flagged | truncated | silent | L0.75 fires | catches | extra flags |
|---|---|---|---|---|---|---|---|
| A sampled (thinking) | 66 | 4 | 2 (no score) | 60 | 2 | **0** | 2 |
| B nothink (comparable regime) | 66 | 4 | 0 | 62 | 5 | **2** | 3 |

Arm B detail (the M4 regime):

- catches (silent wrongs, p < 0.5): `gsm8k-0015:hs2` (p 0.0028 — the catch
  that recurs across the stored slice and this fresh draft) and
  `gsm8k-0036:hs1` (p 0.0013 — a fresh catch: the stored corpus's draft of
  this item was *correct* (label 1, stored p 1.0), so the head read the
  fresh draft and found it wrong; this is distribution-fresh signal, not
  item memorization).
- extra flags (silent corrects that would fire): `humaneval-0003:hg`
  (p 0.279 — its own stored draft was correct at p 0.7364, so this is a
  fresh-distribution false flag), `math-0020:hg` (p 0.024; stored p 0.665),
  `math-0020:hs2` (p 0.108) — three repairs per 66 correct runs that were
  not needed, i.e. ~0.7 s CPU + one greedy generation each; under the strict
  re-verify these clear only if the re-score clears.  The `math-0020`
  family recurs as the flag risk in both arms.
- missed (silent wrongs that stay silent at 0.5 under L0.75):
  `gsm8k-0033:hs1` (p 0.096 under the L0.5 dial — the fresh draft moves this
  item from ≈1.0 stored into the dial's range), `math-0023:hg`,
  `truthfulqa-0015:hg`.

Arm A detail: 0 catches; fires on 2 silent **corrects** (`humaneval-0002:hg`
p 0.407, `math-0020:hs2` p 0.035); of the 6 silent wrongs in that run, 2
keep the gate silent at L0.75 that fired under L0.5 (`gsm8k-0038:hs1`
p 0.336, `math-0020:hg` p 0.240).

## The overhead line

Per 66-item run the gate scores the silent drafts only: 60–62 scores ×
~0.45–0.5 s (warm, measured in the live smoke: p50 0.46 s) ≈ **+28–30 s per
run**.  Against arm B's ≈5.7 min of generation (66 × 5.2 s + repairs) that
is ≈ +8–9 % of wall clock — inside the plan's ~10 % budget, and it is
CPU-side throughput, not GPU contention with the serve.

## Standing numbers (for M4's pre-registered acceptance)

- Go/no-go prior: on fresh nothink drafts the default head catches 2 of 5
  silent wrongs at 3 extra flags.  The fresh catch of `gsm8k-0036` is real
  (it did not exist in the stored slice); the recurring `gsm8k-0015` catch
  is stable across two distributions.
- False-flag damage risk concentrates on `math-0020` (flagged in *both*
  arms even when the draft was correct in one) — the M4 paired run is what
  measures whether the repair repairs a correct answer into a wrong one
  (the measured repair-pass control: 0/8 content harm).
- False-flag risk is distribution-sensitive, not item-fixed: `humaneval-
  0003:hg` sat at p 0.736 on the stored correct draft and fires at 0.279 on
  the fresh (also correct) draft; `math-0020` fires in *both* arms on
  regimes whose drafts were correct there.  Measuring that damage — and the
  strict re-verify's steadiness — is what the live paired run (M4) exists
  for; `gsm8k-0015` remains the only catch present in both distributions.
