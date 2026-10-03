# Held-out benchmark refit — E4=Strata (2026-10-03)

Task set `cascade-bench-v1` (`experiments/cascade/tasks-bench.json`): seeded,
stratified GSM8K/MATH/HumanEval/TruthfulQA, 70 train / 30 test, checkers only
(no model in the loop).  Fitted on the **train** labels and reported on the
held-out **test** labels; no test task was used to pick the threshold.

| split | tasks | local (Strata) | API (flash) | Tiny-Jev fitted threshold | quality | API share | routing acc |
|---|---:|---:|---:|---:|---:|---:|---:|
| train (fit) | 70 | 0.971 | 0.900 | **0.35** | 0.971 | 0.000 | 0.986* |
| test (report) | 30 | **0.967** | 0.933 | 0.35 (from train) | **0.967** | **0.000** | 0.967 |

\* at the chosen 0.35 the train routing accuracy is 0.971; the 0.986 belongs to
0.40, which buys 1.4% API share for the same quality (the rule prefers the
free point).

Rule (`bench_refit.py`): maximise train quality; tie → minimise train API
share; tie → highest threshold (escalation headroom).  On this data every
threshold ≤ 0.50 routes all tasks locally with the same quality; the rule
therefore picks the top of the flat region (0.35 on train).

## Multi-sample benchmark (model card)

`strata-bench-test-capped-20261003`: held-out test split, 3 samples/task
(T=0.7, thinking capped at 1,024 tokens, max_tokens 2,048, high effort):

| benchmark | tasks | samples | sample acc | task macro | mean ms | mean tokens |
|---|---:|---:|---:|---:|---:|---:|
| gsm8k | 12 | 36 | 0.972 | 0.972 | 9,422 | 333 |
| humaneval | 6 | 18 | 0.833 | 0.833 | 24,877 | 557 |
| math | 8 | 24 | 0.833 | 0.833 | 22,468 | 498 |
| truthfulqa | 4 | 12 | 0.833 | 0.833 | 20,050 | 290 |
| **overall** | 30 | 90 | **0.889** | **0.889** | — | — |

`MODEL-CARD.md` in that run dir.  The uncapped first capture
(`strata-bench-test-20261003`, 0.833) is superseded: its misses were engine
crashes and thinking truncation, recorded in that dir's `INCIDENT.md`.

## Caveats

- 30 test tasks (12/8/6/4 per benchmark): a slice, not the full benchmarks;
  per-benchmark numbers have wide intervals.
- The label runs use temperature 0 and a 1,024-token thinking cap (the capture
  uses T=0.7); local answers are single-sample per task.
- Checker-based grading undercounts unusual formats (LaTeX in MATH, code
  formatting).
- The engine wedged twice with Strata's #267 during label runs and recovered
  via `/unload` + retry (see the incident note); no failed label survived into
  the reports.
