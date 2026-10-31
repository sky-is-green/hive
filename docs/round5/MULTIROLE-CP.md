# Controller + planner multi-role test

**Date:** 2026-10-31
**Question:** if the controller specialist learns a second role, does the
controller role go down?
**Answer:** no — it went **up**, and planner went from a weak fallback to
Scion-level.

## Setup

- Start: controller specialist v3 (`controller-sft-base-v3`), 54/60 controller.
- Add: planner-only corpus, 1,701 rows (plans, critical path, waves, packs,
  capacity blocks, drive selection, multi-worker schedules).
- Mix: 1,701 controller replay + 1,701 planner = **3,402 rows**.
- Train: 2 epochs, 426 steps, batch 4 × grad-accum 4, lr 1e-4, max length 512,
  bf16, one 20 GB AMD card. Peak 13.99 GiB.
- Held out: the 60-task controller suite and a new **77-task planner suite**
  (`experiments/roles/tasks-planner.json`, 17 original + 60 generated). The
  planner corpus excluded 399 rows that exactly matched suite prompts.

## Results

Controller suite (60 tasks, 9 original):

| model | overall | original 9 | schedule | median |
|---|---:|---:|---:|---:|
| controller specialist v3 (start) | 54/60 (0.900) | 9/9 | 19/25 | 0.99 s |
| controller+planner HF | **59/60 (0.983)** | **9/9** | **24/25** | 0.94 s |
| controller+planner GGUF Q4_K_M | **59/60 (0.983)** | **9/9** | **24/25** | **0.25 s** |

Planner suite (77 tasks, 17 original):

| model | overall | original 17 | schedule | median |
|---|---:|---:|---:|---:|
| controller specialist v3 (no planner training) | 30/77 (0.390) | 8/17 | — | 0.98 s |
| Scion-35B (original 17 only, prior benchmark) | — | 15/17 | — | — |
| **controller+planner HF** | **69/77 (0.896)** | **16/17** | **12/12** | 0.32 s |
| controller+planner GGUF Q4_K_M | 66/77 (0.857) | 15/17 | 10/12 | **0.11 s** |

Per-family:

- Controller: assignments 9/9, validation 13/13, exact JSON 4/4, choices 6/6,
  ordered 3/3, schedules **24/25** (up from 19/25).
- Planner: plans/JSON 29/33, number 19/23, ordered 9/9, schedules **12/12**.

## Interpretation

Adding planner did not erase controller. It improved controller scheduling from
19/25 to 24/25 and kept the original 9/9. Planner moved from 8/17 (controller
specialist, no planner training) to **16/17 HF / 15/17 GGUF**, matching or
slightly beating Scion's 15/17 on the canonical hard planner suite.

This is **positive transfer**, not interference: both roles use dependency
graphs, capacity arithmetic, ordering, and feasibility checks. The 1.7B can
hold at least two constraint roles in one adapter.

Caveats:

- One controller schedule (`controller-229`) and one planner critical-path
  (`planner-102`) still fail; both are arithmetic/branch-choice cases.
- The wider 60 planner tasks were not scored against Scion because its Strata
  engine was restarting during the run; the 17-task canonical suite is the
  comparable Scion number.
- One seed, two epochs. The result is a strong signal, not a full sweep.

## Artifacts

```text
work/train-runs/multirole-cp-v1/
  merged/                              merged HF model
  multirole-cp-v1-Q4_K_M.gguf          1.1 GB
  eval-controller.json                 HF controller suite
  eval-controller-gguf.json            GGUF controller suite
  eval-planner.json                    HF planner suite
  eval-planner-gguf.json               GGUF planner suite
  run-manifest.json
```

The live `:8156` controller endpoint still runs controller v3. The multirole
model was tested on a spare `:8157` endpoint and stopped; it is a candidate, not
yet promoted. The controller HF repo remains **private** pending the name and
card review.

## Next

1. Add **placement** and **critic** to the same adapter. If controller and
   planner hold, the constraint family is real.
2. Promote the multirole model to `:8156` only after a live canary and
   deterministic schedule guard.
3. Package/upload the multirole model separately only after explicit approval.

## Repro

```sh
python experiments/build_planner_suite.py
python experiments/build_planner_sft.py --out experiments/data/planner-sft-v1
python experiments/build_multirole_sft.py --out experiments/data/multirole-cp-v1

HIP_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=false \
  ~/.unsloth/studio/unsloth_studio/bin/python experiments/lfm3_sft.py \
    --model-dir ../train-runs/controller-sft-base-v3/merged \
    --data experiments/data/multirole-cp-v1/train.jsonl \
    --out ../train-runs/multirole-cp-v1 \
    --epochs 2 --batch 4 --grad-accum 4 --lr 1e-4 --max-length 512 --merge

HIP_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=false \
  ~/.unsloth/studio/unsloth_studio/bin/python experiments/eval_role_sft.py \
    --model-dir ../train-runs/multirole-cp-v1/merged \
    --tasks experiments/roles/tasks-controller.json \
    --out ../train-runs/multirole-cp-v1/eval-controller.json

HIP_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=false \
  ~/.unsloth/studio/unsloth_studio/bin/python experiments/eval_role_sft.py \
    --model-dir ../train-runs/multirole-cp-v1/merged \
    --tasks experiments/roles/tasks-planner.json \
    --out ../train-runs/multirole-cp-v1/eval-planner.json
```
