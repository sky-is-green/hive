# M4 — the live paired run: arm-B config + the learned gate (2026-10-18)

The pre-registered test of the learned gate on fresh serve data: the exact
arm-B nothink configuration (`--judge deterministic --repair auto`,
T=0.7, `--local-max-tokens 1024`, `--local-reasoning-effort none`, no
`--api-fallback`) **plus `--gate intern-0.8b`** (frozen head L0.75,
threshold 0.5), 66 tasks, serve 8081 untouched (CPU-only gate, ~520 ms per
silent-draft score).

```sh
.venv/bin/python experiments/cascade/run_cascade.py \
    --tasks experiments/cascade/results/runner-cascade-sampled-nothink-20261018/tasks-organic66.json \
    --scion-base http://127.0.0.1:8081/v1 \
    --judge deterministic --repair auto \
    --local-temperature 0.7 --local-max-tokens 1024 \
    --local-reasoning-effort none \
    --gate intern-0.8b \
    --out experiments/cascade/results/runner-cascade-gate-nothink-20261018 \
    --run-name run
```

## Headline

| run | served correct | drafts correct | false accepts | false rejects | flagged rows |
|---|---|---|---|---|---|
| baseline arm B (committed) | **60/66** | 57/66 | 6 | 0 | 1 |
| gated run | **59/66** | 54/66 | 7 | 0 | 6 |

Gate telemetry: **57 drafts scored** (the detector-silent, non-truncated
ones), **4 fires — 1 catch of a fresh silent wrong, 3 false flags on
correct drafts**; `gate_cleared` 2, `repairs_attempted` 13,
`repairs_cleared` 7, `repairs_still_flagged` 6.

## The four gate events (the only gate-attributable deltas)

| item | draft | p | what happened | served |
|---|---|---|---|---|
| `gsm8k-0006:hg` | **wrong** | 0.306 | **fresh catch** — the only silent wrong of this run's 3; the greedy repair passed the detectors *and* the strict re-score, but the repaired math was still wrong | wrong (the wrong draft would have been served anyway; in the baseline sample this task happened to be right — sampling drift, not gate damage) |
| `gsm8k-0014:hs1` | correct | 0.061 | false flag → repair self-cleared (detectors + re-score ≥ 0.5) | correct |
| `math-0020:hg` | correct | 0.210 | false flag → repair cleared the detectors but the re-score still rejected → **persistent flag**, draft served | correct, flagged |
| `gsm8k-0035:hs1` | correct | 0.378 | false flag → same, **persistent flag**, draft served | correct, flagged |

The two remaining detector-silent wrongs of this run (`gsm8k-0033:hg`,
`math-0002:hs1`) were scored and stayed above the line (p 0.9999 / 0.963)
— the universal-miss class, dead honest.

## The pre-registered acceptance, scored

1. **≥1 marginal catch of a fresh silent wrong — PASS** (1: `gsm8k-0006:hg`,
   p 0.306; it caught none of the 1 catch it *carried over* — the recurring
   `gsm8k-0015:hs2` drew a correct fresh draft and never fired, sampling).
2. **≤2 extra false flags — FAIL: 3** (`gsm8k-0014:hs1`, `math-0020:hg`,
   `gsm8k-0035:hs1` — all on correct drafts; two persisted as flags after
   the strict re-verify did its job).
3. **No false-flag damage — PASS** (all three false-flagged items were
   served *correct*: one via a cleared repair, two via the persistent-flag
   ladder serving the untouched draft; zero cascade_ok flips attributable
   to the gate. The strict re-verify worked exactly as designed: a fired
   gate on a wrong-looking-but-right draft never produced a wrong served
   answer).
4. **≤ ~10 % overhead — PASS** (+519.7 ms per scored draft × 57 = **+29.6 s
   per run ≈ +8.7 %** of the ≈ 342 s generation wall; gated D3 p50 448 ms,
   right at the 0.44 s r2 extraction median. The scion-side mean drifted
   +0.8 s/answer in this run — sampling/serialization noise at T=0.7, not
   gate cost.)

**Verdict per the plan's own rule: the gate stays OFF as the default**
(clause 2 misses 3 > 2; not renegotiated post hoc).  `--gate intern-0.8b`
remains a wired, tested opt-in flag with honest telemetry.

## Context that belongs next to the verdict

- **Sampling drift dwarfs the gate.**  Baseline→gated served flips:
  −5 baseline rights turned wrong and +4 baseline wrongs turned right
  (mean 0.8 s/answer slower, drift, not the gate) — ±4-5 items of
  run-to-run noise against a ~1-item gate effect.  A single paired run
  cannot statistically separate them; the frozen thresholds were the
  discipline that made a decision possible, and they decided.
- **The false flags are exactly the repair-pass's own risk class**
  (`math-0020` recurs as the flag-magnet; the fresh-distribution rates run
  ~2× the stored slice's).  A miss by one flag is a miss: the default stays
  off until the flag-risk ("/2 per 66") is improved — a better head, a
  tighter threshold *chosen on data, not per-run*, or path A (M5).
- What the gate bought when it fired: 1 fresh silent wrong *caught* (the
  repair cured nothing — detectors and the re-score both cleared a still
  wrong answer, i.e. one self-consistency recheck cannot see a wrong-but-
  consistent final step; that is the repair arm's limit, not the gate's
  bug), 1 false-flag repair that self-cleared, and 2 persistent flags that
  kept their correct drafts served.  The ladder showed its work with zero
  silent damage anywhere.

Artifacts: `run/report.json` (per-row `gate_scored`/`gate_fired`/`gate_p`/
`gate_cleared`), `run.log`, `tasks-organic66.json` (copy).  Preliminary
fresh-draft statistics from the committed runs:
`experiments/cascade/results/gate-replay-20261018/REPLAY.md`.
