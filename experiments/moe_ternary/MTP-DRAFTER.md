# MTP drafter in HiveBench (Scion drafter lane)

The Scion release ships a frozen-body **MTP drafter** (see
`~/Desktop/work/scion/docs/SCION-RECIPE.md` §7). This is what was wired into
HiveBench and how to bench it.

## What was integrated

1. **Stack schema** (`harness/stack/schema.py`): `ENGINE_ARGS_ALLOWED` now
   accepts the draft-model flags `--model-draft`/`-md` and
   `--spec-draft-n-max`/`--draft-max`, so a tier can carry a drafter in
   `engine_args`. The existing `spec: {type, n_max}` seam already emits
   `--spec-type <type>`.
2. **Stacks** (`stacks/`):
   - `scion-35b-v2.json` — face = `qwen35-release-v2-cur05-soup.gguf`, drafter =
     `mtp-sidecar-v2-cur05.gguf` (k=1, `n_max 1`).
   - `scion-35b-v2-dspark.json` — same face, drafter =
     `mtp-dspark-v2-cur05.gguf` (mtp2, `n_max 3`; see the gap below).
   Model + drafters are symlinked into `models/gguf/`.
3. **Bench runner** (`experiments/moe_ternary/mtp_drafter_bench.sh`): launches
   `llama-server` baseline vs drafter on the same prompt and reports
   ms/token / speedup / acceptance; for an mtp2 sidecar it routes to the fork's
   `test-mtp-sidecar dspark` harness instead (the head is not server-wired yet).

Prereqs: the fork's ROCm build (`FORK=~/llama.cpp`,
`build/bin/llama-server` + `test-mtp-sidecar`), `HIP_VISIBLE_DEVICES`.

## Run

```sh
cd ~/Desktop/work/hivebench
HIP_VISIBLE_DEVICES=1 bash experiments/moe_ternary/mtp_drafter_bench.sh \
    models/gguf/qwen35-release-v2-cur05-soup.gguf \
    models/gguf/mtp-sidecar-v2-cur05.gguf 8138 128
```

## Measured (7900 XT, greedy, 128–160 tok)

| run | ms/token | speedup | acceptance |
|---|---|---|---|
| v2 baseline | 15.86 | — | — |
| v2 + k=1 sidecar | 13.15 | **1.206×** | 0.416 |
| v2 + DSpark (mtp2) | — | — | harness: 1.875 tok/fwd (ideal) |

CLI cross-check (`llama-cli`, 128 tok): 49.0 → 57.2 t/s = **1.17×**.

## The gap (blocks a release)

The mtp2/DSpark head is validated only in the harness. The **batched-verify
forward is not timed** and the multi-token propose is **not in
`common/speculative.cpp`**, so its real server speedup is unproven. Next session:
`~/Desktop/HANDOFF-DSPARK-NUMBERS.md`.
