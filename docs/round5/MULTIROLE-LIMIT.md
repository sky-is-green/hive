# Multi-role limit at 1.7B — two roles hold, three degrade

**Date:** 2026-10-31
**Question:** how many constraint roles can one 1.7B adapter hold before the
earlier roles degrade?
**Answer:** **two**. Controller+planner holds cleanly. Adding placement — or
placement+critic — degrades controller and planner on their original suites,
even though the new roles themselves learn well.

## Results

All numbers are HF merged bf16 on held-out suites. The controller suite is 60
tasks (9 original); planner is 77 (17 original); placement is 46; critic is 38.

| model | controller | planner | placement | critic |
|---|---:|---:|---:|---:|
| **2-role: controller+planner** | **59/60 (0.983)**, 9/9 orig | **69/77 (0.896)**, 16/17 orig | — | — |
| 3-role: + placement | 54/60 (0.900), 9/9 orig | 68/77 (0.883), **14/17 orig** | 45/46 (0.978) | — |
| 4-role v1: + placement + critic | 56/60 (0.933), 9/9 orig | 67/77 (0.870), **14/17 orig** | 44/46 (0.957) | 37/38 (0.974) |
| 4-role v2: more controller replay | 55/60 (0.917), 9/9 orig | 69/77 (0.896), **13/17 orig** | 44/46 (0.957) | 37/38 (0.974) |

Gate rules: controller original 9/9, controller wider ≥57/60; planner original
≥16/17 HF / ≥15/17 GGUF, planner wider ≥67/77; new roles ≥0.90.

- The 2-role model passes.
- The 3-role model fails: controller 54/60 (−5) and planner original 14/17.
- Both 4-role models fail: controller 55–56/60 and planner original 13–14/17.
- Increasing controller replay from 1,200 to 2,200 rows did **not** fix it; the
  3-role placement test with 2,200 controller rows still scored 54/60.

## What this means

The new roles are learnable — placement reaches 0.978 and critic 0.974. The
failure is **interference/capacity**, not task difficulty. Adding a third
constraint domain pulls the shared 1.7B away from the controller/planner
behaviour, and more replay only moves the loss around.

So the practical limit for one 1.7B LoRA adapter is **two constraint roles**.

## Recommendation

1. **Keep `multirole-cp-v1` as the live controller+planner checkpoint.**
2. **Serve placement and critic as separate adapters/GGUFs.** Each is 1.1 GB
   Q4_K_M; the display card can hold several. They score 0.96–0.98 on their own
   suites and do not need to share the controller’s weights.
3. **Do not merge a third role into the live adapter** unless a larger base is
   used or a per-role adapter router is built.
4. If one model must cover 3+ constraint roles, move to a larger base
   (Qwen3.8-27B class) or a mixture-of-adapters design.

The queue gate correctly stopped batch 1. No candidate was promoted; the live
`:8156` endpoint remains the 2-role model.

## Artifacts

```text
work/train-runs/multirole-cp-v1/    two-role baseline (live)
work/train-runs/multirole-pc-v1/    four-role v1 (failed gate)
work/train-runs/multirole-pc-v2/    four-role v2, more controller replay (failed)
work/train-runs/multirole-cpp-v1/   three-role controller+planner+placement (failed)
```

Suites/corpora:

```text
experiments/roles/tasks-placement.json
experiments/roles/tasks-critic.json
experiments/data/placement-sft-v1/
experiments/data/critic-sft-v1/
experiments/data/multirole-pc-v1/
experiments/data/multirole-pc-v2/
experiments/data/multirole-cpp-v1/
```

## Next

Train **placement-only** and **critic-only** specialists as separate small
models, using the same base and evaluation gate. They are independent of the
controller/planner adapter, so the two-role model stays intact. Add them to the
queue as separate-adapter batches rather than merged-role batches.
