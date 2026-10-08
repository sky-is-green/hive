# Decision-head sidecar prototype (engine-loaded head)

Open question (HANDOFF §6): sidecar packaging (head tensors + format hash
loaded by the engine) vs the current client-side JSON.

## Artifact

`experiments/cascade/export_sidecar.py` converts a decision-head JSON to a
GGUF (arch `decision-head`):

- tensors `dh.w`, `dh.mu`, `dh.sd` (F32, `[dim]`);
- metadata `decision_head.version=1`, `dim`, `b`, `platt_a`, `platt_c`,
  `format_sha256`, `model`, `candidate`, `name`.

The `format_sha256` is the hash of the exact serving prompt string the head was
fitted on; the engine refuses a head whose hash does not match the prompt it is
asked to score (the format-lock result from the attribution audit).
Size: 61 KB for the 5120-d Bonsai head vs 164 KB as JSON.
Round-trip verified (`--verify`): sidecar vs JSON p delta < 2e-9.

## Engine API (moe-corr fork, local commits)

- `src/llama-decision-head.{h,cpp}` — `llama_decision_head::load()/score()`;
  score is `p = sigmoid(a * (((h - mu) / sd) . w + b) + c)` on the raw
  last-prompt-token state (the same vector `/embedding` returns for
  `--embd-normalize -1`).
- `src/llama-ext.h` — `llama_decision_head_load/free/score/dim/format_sha`.
- `tests/test-decision-head.cpp` — loads a model + head, applies a templated
  prompt through the normal context path (pooling last, no normalisation) and
  prints `p_correct`; the engine-loaded counterpart of the client JSON.

Build:

```sh
cd /home/penis/llama.cpp
cmake -B build && cmake --build build --target test-decision-head -j16
build/bin/test-decision-head -m model.gguf --head head.gguf \
    --prompt-file prompt.txt --format-sha <sha>
```

## Status

- [x] exporter + round-trip verification (JSON vs sidecar p < 2e-9)
- [x] loader / ext API / harness compile
- [x] loader/math self-test without a model (`--head-only`): C++ vs Python
      `p_correct=0.000026326` identical on `h[i]=sin(i*0.001)`
- [x] GPU parity vs the client JSON on real prompts (`sidecar_parity.py`,
      2026-10-08): p deltas 8.4e-9 / 2.4e-8 / 6.2e-9 on gsm8k/math/humaneval
      sample 0 (tolerance 2e-3).
- [x] batch readout mode (hidden states + answer-slot token probabilities for
      a directory of prompts, one process) — added to `test-decision-head` and
      used by the slot-logit and community readouts.
- [ ] server endpoint (`/head-score`) wiring.

## Deployment note: embedded correction adapters

During the GPU parity work the harness initially failed to reproduce the
stored scores: models like `Scion-35B-A3B-PQ2_0-corr.gguf` **embed a
correction LoRA adapter** (240 tensors), which `llama-server` applies per
batch via `llama_set_adapters_lora`; a raw context does not.  Any engine-side
scoring of an adapter-embedded model must call
`llama_set_adapters_lora(ctx, nullptr, 0, nullptr)` after context creation
(the harnesses now do; without it the embeddings come from the uncorrected
quant).  Details: `MONITOR-FIX.md` §2.
