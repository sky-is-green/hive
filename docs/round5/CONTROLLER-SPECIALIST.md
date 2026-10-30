# Controller specialist — 1.7B hyper-specialisation

**Date:** 2026-10-30
**Target:** `LFM3-1.7B-Base` + controller-only LoRA (one role, one output contract)
**Deployed:** `controller-1.7b-sft` on `:8156`, role `controller` → `controller17`

The generalist SFT run took controller from 2/9 to 8/9 on the nine-task role
suite. That was promising but too narrow to hand over a merge seat. This run
builds a wider held-out controller suite, trains a controller-only specialist,
and compares it with Scion-35B and with a specialist warm-started from the v3
generalist.

## Wider held-out suite

`experiments/roles/tasks-controller.json` — **60 tasks**:

- the nine original controller tasks, unchanged;
- 9 assignment tasks, 25 dependency/skill/release-window schedules, 8
  assignment-validation questions, 6 priority/merge choices, 3 ordered merges,
  4 capacity yes/no questions, plus policy cases.

The generated assignment tasks use a new **feasibility checker** (`assign`) that
accepts any assignment within worker capacities. The old exact-set checker
rejects a valid alternative assignment; that is fine for the original nine
curated tasks but unfair on generated instances with many feasible solutions.
The nine original tasks still use their exact checkers.

`experiments/build_controller_suite.py` validates every gold answer before
freezing the suite. The suite is excluded from training (one corpus row in v2,
five in v1, happened to match a suite prompt and was dropped).

## Corpus

`experiments/build_controller_sft.py` — **2,399 rows**, controller only:

| family | rows |
|---|---:|
| exact-fill assignment (first-fit) | 700 |
| general assignment | 420 |
| dependency schedules | 420 |
| skill schedules | 420 |
| release/deadline windows | 280 |
| assignment validation yes/no | 320 |
| capacity yes/no | 140 |
| feasibility | 80 |
| priority / merge policy | 39 |

v1 of the corpus was 73% "yes" on validation questions, and the model learned
to over-accept invalid assignments. v2 balances the labels and adds harder
invalid cases; that is the shipped corpus.

## Result

Held-out 60-task controller suite, greedy decoding:

| model | overall | original 9 | new 51 | median |
|---|---:|---:|---:|---:|
| Scion-35B | 50/60 (0.833) | **9/9** | 41/51 | 3.96 s |
| v3 generalist + controller adapter | 51/60 (0.850) | 8/9 | 43/51 | 0.94 s |
| **base + controller adapter v2** | **52/60 (0.867)** | **9/9** | **43/51** | **0.94 s** |
| base + controller adapter v1 | 50/60 (0.833) | 9/9 | 41/51 | 0.95 s |

Per checker type (Scion / v3-spec / base-v2):

| type | Scion | v3-spec | base-v2 |
|---|---:|---:|---:|
| assignment | 7/9 | 8/9 | 6/9 |
| schedule | 19/25 | 19/25 | **22/25** |
| contains / validation | 11/13 | 12/13 | 11/13 |
| exact JSON | **4/4** | 3/4 | **4/4** |
| choice | 6/6 | 6/6 | 6/6 |
| ordered | 3/3 | 3/3 | 3/3 |

The specialist beats Scion by two tasks overall, ties the original nine at 9/9,
and is **4× faster** (0.94 s vs 3.96 s median in HF; 0.25 s through the GGUF
server). The warm-start-from-v3 variant is one task behind and loses one of the
original nine, so it was not deployed.

Q4_K_M GGUF parity: **52/60 (0.867)**, median **0.254 s** through
`:8156` with `--reasoning off`.

## What still fails

- **Assignment capacity arithmetic.** `controller-212/215/217` return a
  `loads` map that visibly exceeds a worker's capacity, but the assignment is
  left unchanged. The model can state the load and still not enforce it.
- **Validation false positives/negatives.** `controller-308` (W1 overloaded)
  still answers yes; `controller-307` is a false no in some runs. This is the
  main reason to keep a deterministic post-check in the controller path.
- **Schedule edge cases.** Three of 25 schedules still miss a deadline or
  dependency; Scion misses six.

The next functionality win is a **deterministic guard**: after the model emits
assignments, validate loads in code; on failure, retry once or fall back to
Scion. That converts the remaining arithmetic misses into retries rather than
bad merges.

## Deployment

- Model: `work/train-runs/controller-sft-base-v2/controller-sft-base-v2-Q4_K_M.gguf`
- Server: llama.cpp b11960 Vulkan, `:8156`, alias `controller-1.7b-sft`,
  8k context, 2 slots, `--reasoning off`
- Role config: `hivebench/harness/webui/roles.json` → `controller:
  controller17`, placement `display`
- UI endpoint map: `harness/webui/dev_server.py` → `controller17:
  http://127.0.0.1:8156`
- Provider: `controller17` (`http://127.0.0.1:8156`, model
  `controller-1.7b-sft`); default provider remains `scion35`
- Scion stays live on `:8082` as the fallback for planner/router/face/agency and
  for controller retries. The fallback is currently manual: flip `roles.json`
  controller back to `scion35`, placement `non_display`, and stop `:8156`.

The controller specialist runs on the display card because the non-display card
is fully resident with Scion + DS4.1. If a game or display stall hits, the
controller can fall back to Scion; there is no automatic timeout failover yet.

## Repro

```sh
python experiments/build_controller_suite.py
python experiments/build_controller_sft.py --out experiments/data/controller-sft-v2

HIP_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=false \
  ~/.unsloth/studio/unsloth_studio/bin/python experiments/lfm3_sft.py \
    --model-dir ../models-hf/LFM3-1.7B-Base \
    --data experiments/data/controller-sft-v2/train.jsonl \
    --out ../train-runs/controller-sft-base-v2 \
    --epochs 3 --batch 4 --grad-accum 4 --lr 2e-4 --max-length 512 --merge

HIP_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=false \
  ~/.unsloth/studio/unsloth_studio/bin/python experiments/eval_role_sft.py \
    --model-dir ../train-runs/controller-sft-base-v2/merged \
    --tasks experiments/roles/tasks-controller.json \
    --out ../train-runs/controller-sft-base-v2/eval-specialist.json

python experiments/eval_role_endpoint.py \
  --base http://127.0.0.1:8156 --model controller-1.7b-sft \
  --tasks experiments/roles/tasks-controller.json \
  --max-tokens 512 --no-response-format \
  --out ../train-runs/controller-sft-base-v2/eval-gguf-v2.json
```

## Next

1. Add the deterministic assignment guard + Scion retry to the controller path.
2. Run a live canary on a sample of controller traffic and compare with Scion
   traces; revert if false positives appear.
3. Keep planner as the next specialist candidate only if the controller canary
   is clean. Router and face stay Scion.
