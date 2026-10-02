# Router threshold re-fit — E4=Strata (2026-10-03)

Source labels: `strata-cascade-20261003-hard/report.json` +
`strata-cascade-20261003-easy/report.json` (27 tasks, measure mode, one sample
each; Strata served by `stacks/strata-125b.json`).
Evaluator: `experiments/cascade/router_eval.py` (Tiny-Jev-1.7B, GPU, 46
ms/decision across the 27 tasks). Machine artifact: `strata-pilot-router-eval.json`.

## Result

With Strata as the local E4 generator the hard set no longer discriminates:
**local 27/27, API 27/27, cascade 27/27, zero escalations**. Consequently
always-local, the oracle and every Tiny-Jev threshold up to 0.55 all give
policy quality **1.000 at api share 0.000**. The Scion-era operating point
(0.90) would have escalated **55.6%** of tasks for no quality gain.

| router | quality | api share | routing accuracy |
|---|---:|---:|---:|
| always local | 1.000 | 0.000 | — |
| always API | 1.000 | 1.000 | — |
| oracle | 1.000 | 0.000 | — |
| API router (pilot records) | 1.000 | 0.000 | 1.000 |
| Tiny-Jev, best threshold (0.05) | 1.000 | 0.000 | 1.000 |

Tiny-Jev sweep (quality / api share / accuracy): 0.05–0.55 → 1.000 / 0.000 /
1.000; 0.60 → 1.000 / 0.037 / 0.963; 0.90 → 1.000 / 0.556 / 0.444; 0.95 →
1.000 / 0.630 / 0.370.

## New operating point

**Threshold 0.55** — the top of the flat region. Any value ≤ 0.55 routes every
task locally on this set; 0.55 keeps the most headroom for escalating a task
the router is genuinely unsure about (e.g. a future task where
P(local correct) < 0.55).

## Caveats

- 27 hand-written tasks, single sample, no held-out split; thresholds are
  fitted and reported on the same labels.
- With E4=Strata the set no longer discriminates the generator, so the flat
  region is a lower bound on what a harder benchmark would need: the next
  calibration asset is the held-out benchmark slice (priority plan item 1),
  not more threshold fitting on these 27 tasks.
- The verifier was the frontier API prompt in this pilot; with local accuracy
  saturated the judge saw only correct answers (0 false accepts, 0 false
  rejects), so this run does not re-calibrate the judge threshold.
