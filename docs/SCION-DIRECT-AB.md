# Scion 176B: direct Q6_K vs the Q2 bridge

Purpose: decide whether the direct Scion Q6_K/Q4_0 release can retire the legacy Q2 bridge.
Same 30 tasks, same sampling, one engine at a time. The controller is upstream Strata 0.2.10:
`answer_room` (on) and the request's `thinking_budget_tokens`; no custom budget logic.

## Environment

```sh
export STRATA_REPO=/path/to/strata          # the scion-0.2.10 checkout
export SCION_MODELS=/path/to/models-p310/scion-strata
export STRATA_DATA=/path/to/strata-data
export PATH="$PWD/.venv/bin:$PATH"
```

The direct config expects `$SCION_MODELS/Scion-FlashNext-176B-A6B-Q6_K.gguf` and `pack-q6k`.
The bridge config expects `Scion-FlashNext-176B-A6B-Q2_0.gguf` and `pack-q2-ab`.

## Protocol

Both arms:

- 30-task test split from `experiments/model_matrix.py`
- `--max-tokens 4096`
- `--thinking-budget 3072` (request-level `thinking_budget_tokens`)
- `temperature 0.1`, one sample
- `--max-context 8192` (the stack forces it; the engine configs keep their own defaults)
- `answer_room: true` in both server configs (0.2.10's default, written down)

## Direct arm

```sh
hive serve scion-flashnext-strata --detach --timeout 900
hive status scion-flashnext-strata

curl -s http://127.0.0.1:8081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"scion-flashnext-176b-a6b-q6k",
       "messages":[{"role":"user","content":"What is 84 * 3 / 2? Answer briefly."}],
       "max_tokens":1024,"thinking_budget_tokens":512,"temperature":0}'

.venv/bin/python experiments/ab_qwen27_flashnext.py \
  --models scion-q6k --split test --max-tokens 4096 --thinking-budget 3072 \
  --out experiments/model-matrix/ab-scion-q6k-v0210-4096-tb3072.json

hive stop scion-flashnext-strata
```

## Bridge arm

```sh
hive serve scion-flashnext-strata-q2 --detach --timeout 900
hive status scion-flashnext-strata-q2

.venv/bin/python experiments/ab_qwen27_flashnext.py \
  --models scion-q2 --split test --max-tokens 4096 --thinking-budget 3072 \
  --out experiments/model-matrix/ab-scion-q2-v0210-4096-tb3072.json

hive stop scion-flashnext-strata-q2
```

The Q2 arm is the **legacy bridge**, a Strata-compatible repack of the Q6_K release. It is not an
official Scion Q2 and its number must not be used as one in public material.

## Compare

```sh
.venv/bin/python experiments/compare_scion_direct_q2.py \
  experiments/model-matrix/ab-scion-q6k-v0210-4096-tb3072.json \
  experiments/model-matrix/ab-scion-q2-v0210-4096-tb3072.json
```

Read accuracy first, then `finish_reason`/truncation. If the direct arm is equal or better, retire
the Q2 bridge from the primary path; if it is worse, inspect the pack and the correction
application before making any claim. A Q2 result produced under a smaller context (`-c 512`) is
not comparable and must be discarded.
