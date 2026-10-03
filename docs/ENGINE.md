# Engine lane — prism ROCm expert residency (hive_cascade)

Our own work on serving MoE models whose expert working set does not fit VRAM
on consumer AMD (2× RX 7900 XT 20 GB, ROCm), by keeping a hot-expert slot band
in VRAM and fetching the routed experts on demand. The slot cache is an
**independent implementation** in the prism ROCm engine and the Hivebench
harness; it is *inspired by* [FlashML
FreeToken](https://github.com/FlashML-org/FreeToken)'s published design (LRU
expert caching, bandwidth-adaptive co-execution) but shares no code with it and
targets a different runtime (llama.cpp/ggml, not their Python/Triton engine).
Formerly tracked as "AMD FreeToken" / project Ember.

> **Consolidated 2026-10-02:** the engine lane is part of the **hive_cascade**
> program (formerly project Ember). The standalone `ember` branch is retired
> (tip `4d07d81`, fully contained in `hive_cascade`). The prism engine branch
> is renamed `engine/ember` → `engine/prism`; stacks are `prism-*`.

## Status (2026-09-28; unshelved 2026-10-02 — baseline below)

- **Engine end-to-end through Hivebench** — the 125B Qwen3.8-Flash-Next MoE
  (Q2_0, 512 experts top-10, 48 layers) applies via the stack API on the prism
  ROCm binary; `backend: "rocm"` per tier; chat completions round-trip.
- **Engine A/B vs a stock build**: prism `engine/amd-rig` vs `origin/master` on
  the 35B MoE — +33–37% prefill, +11% decode, byte-identical greedy output.
- **Slot cache** (`--moe-slot-cache N` / `-msc auto`): host-resident
  expert banks keep a hot-expert slot band in VRAM; decode fetches misses over
  PCIe and runs every routed expert on the GPU; fully-covered banks are
  promoted once at load (identity slots, no per-step overhead, graph capture
  stays on). Steady-state 125B measurements (256-token decode, 185-token
  prompt, ctx 4096, headless-first split):

  | config | experts off-VRAM | slot cache | decode | vs no-cache | hit rate |
  |---|---:|---:|---:|---:|---:|
  | all-resident | 0 | – | 26.9 t/s | – | – |
  | `-ncmoe 12` | 8.1 GiB | – | 19.3 | – | – |
  | `-ncmoe 12 -msc 256` | 8.1 GiB | 4.1 GiB | **24.5** | **+27%** | 87.9% |
  | `-ncmoe 24` | 16.2 GiB | – | 14.5 | – | – |
  | `-ncmoe 24 -msc 256` | 16.2 GiB | 8.1 GiB | **23.0** | **+59%** | 88.6% |
  | `-ncmoe 36` | 24.3 GiB | – | 3.9 | – | – |
  | `-ncmoe 36 -msc 336/448` | 24.3 GiB | 15.9–21.2 GiB | **11.5–15.4** | **+200–300%** | 89.7% |

  35B-A3B one card: 37.8 → **60.0 t/s** from 2.5 GiB of slots (75.9% hits);
  full coverage = all-GPU **108.9 t/s** (identity promotion, token-identical).
  Harness stack `stacks/prism-125b-lowvram.json` (`-ncmoe 12 -msc auto`) picks
  its own size (449 slots / 7.1 GiB at apply time) and serves warm
  268–307 t/s prefill / 20.6–27.4 t/s decode.
- **Shelved with these open items** (see "Next steps" below): kernel-count
  reduction, telemetry, prefill streaming for cached layers, device-side
  ensure.

The 2026-10-02 baseline re-measured the 125B arms through
`experiments/cascade/bench_e4_baseline.py`; see "E4 baseline results" below.

## Why it works (measured diagnosis)

- Steady-state hit rate is **~88–90%** (cold-start runs understate it); at
  ≤16 GiB of offload the misses are page-cache served (~50–70 MiB disk per
  256-token generation) and decode is within **9–15%** of all-resident.
- That residual gap is the **per-layer host round trip + lost CUDA-graph
  replay** (~4–6 ms/step at 24 cached layers): the map op syncs the stream, so
  the graph containing it cannot be captured.
- At 24 GiB of offload the **page cache thrashes** (1.6–7.4 GiB disk reads per
  generation) and decode becomes disk-sensitive (11.5–15.4 t/s), still 3–4×
  the no-cache arm.
- A GPU kernel **cannot read unpinned mmap memory** on this box ("Page not
  present"), so FreeToken's device-side gather needs pinned banks:
  `hipHostMalloc(8 GiB)` works and reads at ~12.5 GB/s (half the ~25 GB/s DMA
  copy rate). 30 GB RAM caps how much of the expert set can ever be pinned.
- All-resident is **dispatch-bound** (~5,000 kernels/step × ~3.5 µs per
  `tools/engine-rig/notes/BUDGET-20260928.md`) — the biggest untouched tok/s
  lever for every configuration.

## The engine

Repo: `sky-is-green/prism-ml-llama.cpp`, **branch `engine/prism`** (renamed
from `engine/ember`; based on `engine/amd-rig`; local checkout
`~/Desktop/work/prism-ml-llama.cpp`, built in `build-hip`).

- upstream llama.cpp master (has `qwen4exp` / Qwen3.8-Flash-Next and
  `--lazy-mode` PLE streaming) + the HSA upload fix (`set_tensor` staging;
  prevents the ROCm dual-GPU load hang) + `LLAMA_GRAPH_DUMP` + the slot cache.
- Slot-cache pieces: `GGML_OP_MOE_CACHE_MAP` (+ `ggml/include/ggml-moe-cache.h`),
  `src/llama-moe-slot-cache.{h,cpp}`, the `build_moe_ffn` hook,
  `--moe-slot-cache N|auto`, `LLAMA_MOE_SLOT_STATS=1` hit/fetch counters.
- Research record: `tools/engine-rig/notes/SLOT-CACHE-20260928.md` (design,
  measurements, FreeToken alignment); budget notes and rig under
  `tools/engine-rig/`.

Hivebench integrates it through the per-backend binary convention
(`harness/models.py::_binary_for_backend`):

```
tools/backends/rocm/llama-server      # the prism build (scripts/install_amd_engine.sh)
stacks/prism*.json                    # "backend": "rocm" tiers
```

## Install / use

```sh
# build/symlink the engine into tools/backends/rocm/  (~2 s if the engine is built)
scripts/install_amd_engine.sh            # or: --copy, SKIP_BUILD=1, PRISM_DIR=...
tools/backends/rocm/llama-server --version

# serve the 125B FreeToken-shape stack
.venv/bin/python -m harness --no-open --no-auto-start &      # sidecar on :8765
curl -sX POST localhost:8765/v1/stacks/prism-125b/validate
curl -sX POST localhost:8765/v1/stacks/prism-125b/apply
curl -s localhost:8765/v1/stacks/status

# the offload+cache variant (12 layers' experts host-side, cache auto-sized)
curl -sX POST localhost:8765/v1/stacks/prism-125b-lowvram/apply
```

Stacks in this branch:

| stack | what | status |
|---|---|---|
| `prism-125b` | 125B MoE all-resident (headless-first split) | works |
| `prism-125b-lowvram` | same + `-ncmoe 12 -msc auto` | works |
| `prism-35b` | 35B-A3B single-card smoke / A/B model | works |
| Bonsai-2 27B/1.7B (PQ2_0) | fork-private tensor type 142 | stack deleted (cannot load on `engine/amd-rig`; recover from git history for a future forward-port) |

Model library: the GGUFs are symlinked into `~/.lmstudio/models` (the harness's
default `models_dir` on this box): the 125B in
`~/Desktop/work/models/qwen38-q2_0/Q2_0/`, the 35B in
`~/Desktop/work/hivebench/artifacts/ternary/refs/qwen35/`.

## Toolchain fixes that made this possible (hivebench)

- `harness/models.py` resolved the repo root one level too high (left over from
  the flat-package move): `tools/backends/rocm/llama-server` was never found and
  backend selection always refused.
- `harness/stack/manager.py` emitted `--ts`, which no llama.cpp understands
  (`-ts` / `--tensor-split` is the real flag); every stack with a `ts` died at
  startup.
- `engine_args` seam: a validated whitelist (`ENGINE_ARGS_ALLOWED`) so a stack
  can carry `-ncmoe`, `-msc`, `--lazy-mode`, `-np`, etc. without owning the
  model/host/port flags.

## Strata — external reference point (2026-10-02)

[Strata](https://github.com/Niko1221/Strata) (MIT, ~5.7k stars, trending
September 2026) runs the same Qwen3.8-Flash-Next Q2_0 pack on consumer GPUs
with an overlapping design: an adaptive hot-expert cache in VRAM and all
experts in RAM, but **off-VRAM experts are computed by the CPU in place**
(AVX-512, concurrent with the GPU) instead of fetched over PCIe — plus the
model's own MTP speculation (2.4-3.2 accepted tokens/pass), 8K prefill chunks
with layer-ahead expert prefetch, KV streaming and k8v4, mapped/resident
low-RAM modes, and a full OpenAI + Anthropic serving surface with per-tier
`/metrics` and per-box calibration.

Published Q2_0 numbers on a 12 GB RTX 5070 / 64 GB PC: 87.3 t/s at 1K context,
73.7 at 128K, 60.3 at 262K. Our all-resident 125B measurement is 26.9 t/s; the
gap is speculation (1.6-1.8x alone), CPU co-execution, prefill chunking, and
the dispatch floor noted below. Their design validates the slot-cache thesis; the
unshelve decision is open. Full comparison and borrow list:
`docs/CASCADE-PILOT.md` ("Strata — the local Flash-Next engine").

### Strata on this box — feasibility measured (2026-10-02)

Built Strata `db4f91a1…` from source for gfx1100 (system ROCm 7.2.4) with the
Q2_0 pack made from our canonical GGUF (`--gguf-dir`), and its MTP runtime from
the pinned BF16 `mtp.*` tensors. It runs on 2× 7900 XT + 30 GB RAM in
`--mmap-experts` mode **with the desktop apps running**: the first attempt
aborted on VRAM exhaustion on the display card
(`experiments/cascade/results/strata-feasibility-20261002/INCIDENT.md`), fixed
by the coexistence profile `strata-q2_0-coexist.json`
(`--vram-reserve-mib 3072 --prefill 512`; load in 30 s, stable). Same prompt
sequence as the E4 baseline (Strata generated 170 tokens per decode — EOS, no
`ignore_eos`):

| metric | prism best (baseline) | Strata |
|---|---:|---:|
| cold pp 4K | 312 | 407 |
| warm pp 1K | 434 | 523 |
| decode | 25.9 | **56.8–57.3** |
| MTP draft acceptance | – | **~77%** (114–115 of 148–150) |
| RAM available min | 24.8 GB | 26.9 GB |
| VRAM peak c0/c1 | 20.5/19.8 GB | 18.5/19.2 GB |

Memory is released on close (VRAM back to desktop baseline, MemAvailable back
above the pre-run level).

### Integration in hivebench — endpoint tier + managed launch (2026-10-02/03)

**Decision: the hybrid** — a stack tier may carry `endpoint` (the contract)
and an optional `launch` block (the managed lifecycle). `apply` adopts a
running server that answers `/health`; when the endpoint is down and `launch`
is present, the harness starts Strata from a generated copy of its config and
owns that process until `unload`. An adopted endpoint is never killed.

| option | what it means | verdict |
|---|---|---|
| (a) endpoint-backed tier | tier carries `endpoint`; apply verifies and registers role/port/ctx/caps | shipped (2026-10-02): the base layer, and still the contract for external servers |
| (b) full launcher | harness spawns `serve/server.py` with a generated config, own port/log management | shipped as the optional `launch` block (`StrataLauncher`), deliberately *not* through `models.py::load`: Strata's argv/config/state do not fit the llama-server seam |
| (c) hybrid | both; adopt when up, start when down | **chosen** — the endpoint stays the contract, the launcher is opt-in per tier and never touches a process it did not start |

Why the launcher lives beside `LlamaServerManager` rather than inside it:
`_binary_for_backend` maps a backend to `tools/backends/<backend>/llama-server`
and `load()` hardcodes the llama-server argv (and adopts a listener only when
the process name contains "llama"). A Strata launch is `python
serve/server.py --engine strata --config <generated> --port N` from the Strata
checkout; it belongs in `harness/stack/strata.py`, which generates the config,
writes logs where llama-server logs live, and tracks only the processes it
started.

Why the endpoint layer stays:
- Strata's serving surface already covers every cascade need: `/health`,
  `/v1/models`, `/v1/chat/completions`, `/props`, `/metrics`, `/unload`,
  `/load`, and the idle-unload / min-free-VRAM / before-load triad. The dsh
  runtime seam (`stack_runtime.tier_routes`) already prefers an explicit
  `base_url` on a status row, so a registered endpoint flows to subagent
  routing with no new code.
- One heavy track at a time: an adopted Strata is the human's process; closing
  it releases VRAM/RAM (measured). A managed launch is stopped by `unload`, so
  the harness never holds 27 GB of expert pages after an experiment.
- Residency: the endpoint tier's footprint is external, so it is skipped by
  `plan_residency` (a warning, not a charge); a stack made only of endpoint
  tiers validates without a local GGUF or even a GPU probe.
- The port track is unaffected: when our models are ported to Strata, the same
  endpoint + launch pair serves them.

Implemented (hivebench, branch `hive_cascade`):

- `harness/stack/schema.py`: optional `endpoint` (http(s) URL), `caps`
  (free-form measured serving facts), and `launch` (`kind: strata`, `repo`,
  `config`, optional `python`, `engine_args`, `options`, `timeout_s`) per
  tier, all round-tripped through the file model. A `launch` block requires an
  endpoint; launcher-owned flags (`--port`/`--config`/...) and options
  (`args`/`exe`/...) are refused.
- `harness/stack/endpoint.py`: `probe_endpoint()` — `GET <root>/health` first
  (refuses a reachable-but-`loaded:false` model with an actionable message),
  `GET <root>/v1/models` as a fallback — plus `endpoint_base`/`endpoint_port`.
- `harness/stack/manager.py`: endpoint tiers become `TierRuntime` rows with
  `base_url`/`caps`/`managed`/`pid` and no `ServerInstance`; a `launch` tier
  is adopted when healthy and started through `StrataLauncher` when not;
  managed llama-server ports skip the endpoint's port; unload and rollback
  stop exactly the processes the harness started.
- `harness/stack/strata.py`: `StrataLauncher` — generated config (base + port
  + `--max-context <tier.ctx>` + `engine_args` merged in place + `options` +
  engine log under `logs/`), command `python serve/server.py --engine strata
  --config <generated> --port N` from the checkout, startup wait
  (`timeout_s`, default 600 s), graceful stop. Generated configs under
  `harness_state/strata/`, server output under `logs/strata_<port>.log`.
- `harness/stack/residency.py`: endpoint tiers are external (warning, not
  charged); an all-endpoint stack short-circuits the hardware probe.
- `stacks/strata-125b.json`: endpoint `http://127.0.0.1:8080/v1`, ctx 4096,
  backend `strata`, the measured caps (56.8–57.3 t/s decode, 407–523 t/s
  prefill, ~77% MTP acceptance, RAM available min 26.9 GB, VRAM peaks
  18.5/19.2 GB, coexistence `--vram-reserve-mib 3072`, artifact
  `experiments/cascade/results/strata-feasibility-20261002/`), and the
  `launch` block (`~/Desktop/work/strata`, `strata-q2_0-coexist.json`).
- `harness/cascade/registry.py`: the `flash-next-strata` E4 candidate carrying
  the measured serving facts (177B, Q2_0, 66 GB, Strata endpoint).

```sh
# apply adopts a running Strata, or starts one when it is down:
curl -s localhost:8765/v1/stacks/strata-125b/validate
curl -sX POST localhost:8765/v1/stacks/strata-125b/apply
curl -s localhost:8765/v1/stacks/status | jq '.tiers[0] | {base_url, caps, managed, pid}'
# unload stops only a process the harness started; an adopted one keeps running
curl -sX POST localhost:8765/v1/stacks/strata-125b/unload
```

Tests: `pytest tests/unit/test_stack_schema.py tests/unit/test_stack_strata_launcher.py
tests/unit/test_stack_residency.py tests/integration/test_stack_manager.py
tests/integration/test_stacks_api.py tests/unit/test_cascade_roles.py
tests/unit/test_cascade_api.py -q` — schema round-trip (including the
`stacks/strata-125b.json` caps and launch), argv merge and generated config,
resolve safety, start health wait/timeout/exit, adopt-vs-start apply,
unload/rollback ownership, health-probe semantics, and port avoidance.

**Pilot result (2026-10-03):** with E4=Strata the cascade scored **27/27**
(14/14 hard, 13/13 easy; Scion was 22/27, 9/14 hard) with **zero
escalations**; hard-set mean 7.4 s / 167 tokens and $0.0033 total API spend
for both sets. RAM available min 22.8 GB, VRAM peaks 17.7/19.0 GB with the
coexist profile; the endpoint was unloaded via the stack API without stopping
Strata, which released memory on close. The router threshold re-fit moved
0.90 → 0.55 (quality 1.000 at api share 0 for every threshold ≤ 0.55).
Artifacts: `experiments/cascade/results/strata-cascade-20261003-{hard,easy}/`
and `strata-pilot-router-eval.{json,md}`.

**Held-out benchmark slice (2026-10-03):** the seeded 100-task
`cascade-bench-v1` (GSM8K/MATH/HumanEval/TruthfulQA, 70/30 train/test) is the
held-out floor. Test split, 3 samples/task (T=0.7, thinking capped at 1,024):
Strata **0.889** sample accuracy (GSM8K 0.972, MATH 0.833, HumanEval 0.833,
TruthfulQA 0.833). Cascade labels (T=0): local 0.971 train / 0.967 test, API
0.900/0.933; a router threshold fitted on train (0.35: quality 0.971, API
share 0) held out at **0.967 quality with API share 0**. The first uncapped
capture (0.833) is superseded — its misses were engine restarts (#267 wedged
verify windows under sustained mmap paging) and thinking truncation, now
recovered in-client (`/unload` + retry); see
`experiments/cascade/results/strata-bench-test-20261003/INCIDENT.md`.
Artifacts: `experiments/cascade/results/strata-bench-*`.

**Cascade default (2026-10-03):** local-first — no pre-generation router,
Tiny-Jev validation on **CPU** (a torch ROCm context beside the resident HIP
engine hangs this host), and the API **off** unless the user configures
`--api-fallback URL`; a rejected answer then escalates there, otherwise it is
returned and flagged.  The benchmark's measurement configuration is unchanged.

### Dual-engine direction — what we gain, what we lose, the porting path (2026-10-02)

Strata is **deliberately single-architecture**. The guard in
`include/strata/artifact/gguf_reader.hpp` (`check_architecture`) refuses
anything but `qwen4exp`: *"The engine is specialised to ONE model; anything
else must be refused with a precise error rather than silently mis-run"*, and
36 source references bind the loader, layer, kernels (PLE streaming, QSA
indexer, hyper-connections) and MTP to that architecture. That focus is the
speed.

What actually blocks another model, from the source:

- **Front door (fail-fast, trivial to relax):** the `qwen4exp` guard plus
  direct reads of `qwen4exp.expert_count`, `qwen4exp.rope.*` and friends in
  `program/generate.cpp` (generic GGUF keys are never consulted). Empirical
  (2026-10-02, engine 0.1.37): pointing the engine at the Scion-35B GGUF fails
  with `no shard of the model holds per_layer_token_embd.weight`, and with
  `--ple-gguf` supplied it fails with `architecture is 'qwen35moe', this
  engine requires 'qwen4exp'`. GGUF is a container it can read; there are no
  interpreters for other architectures in the binary (CMake compiles only
  `third_party/llama.cpp/ggml`, not llama.cpp's `src/models/*`).
- **Forward pass (the real work):** `core/layer.cpp` implements Flash-Next-only
  machinery — QSA sparse attention with its indexer (`qsa.hpp`,
  `native_qsa_indexer`, `qsa_select`, `qsa_decode_attn`), hyper-connection
  mixers, PLE per-layer embeddings (`per_layer_token_embd.weight`), shared
  expert gating; `ModelGeometry` carries indexer dimensions that only this
  architecture has. A different model needs its own layer + graph, behind a
  dispatch.
- **Artifacts and quants:** the pack layout (`experts.bin`/`dense.bin`/
  `embd.bin` + `manifest.json`/`index.txt`/`native_experts.txt`, built for this
  exact tensor set) and the kernel quant families (Q2_0, IQ2_XS, IQ3_*,
  bf16/f32/q8_0 — `s2_gemv*`, `iq_kernels`); our PQ2_0 ternary type has no
  dequant/GEMV/MMQ implementation there.
- **What is generic:** `third_party/llama.cpp` is pinned (`GIT_TAG
  3cf03257…`), so ggml primitives exist; the expert cache, CPU co-execution,
  MTP speculation, prefill chunking and serving are architecture-agnostic. A
  port is "write the model + pack mapping + quant kernels", not a new engine.

**Direction confirmed (2026-10-02):** both engines live in the harness behind
the generator contract — Strata for the Flash-Next tier, prism/llama.cpp for
Scion-35B, the ternary pipeline and the decision models — and **our models are
ported to Strata as a planned track** (work order below). Strata is MIT
(`github.com/Niko1221/Strata`, Copyright (c) 2026 Niko1221 and contributors);
ggml/llama.cpp are MIT; model files keep their own licenses (Qwen,
ISTA-DASLab, Swift, Unsloth).

| | gain | lose / cost |
|---|---|---|
| Strata as the Flash-Next tier | 2.2x decode (57 vs 26 t/s), MTP (~77% acceptance), CPU co-execution of misses, 8K prefill chunks, KV streaming/k8v4, `/metrics`, per-box calibration, coexistence reserve | single-arch lock-in; ternary/35B need a fork port; a second engine to pin and upgrade; greedy output is not byte-identical (adaptive cache + speculation); the prism lane stalls unless kept alive for 35B |
| prism/llama.cpp as the general engine | every GGUF we own (Scion-35B PQ2_0, Bonsai, decision models), the ternary pipeline, upstream rebase, byte-identical greedy | slower on Flash-Next; its open list (CPU co-execution, MTP draft path, prefill chunking, telemetry) |

**Porting track (our models onto Strata)**, work order from the source:

1. **architecture seam**: add a `qwen35moe` implementation behind a dispatch.
   `check_architecture` is the single choke point; `layer.cpp` is
   Flash-Next-specific and needs a sibling, not a rewrite of the whole engine.
2. **packer**: `tools/strata_pack.py` already keys off standard
   `blk.N.ffn_{gate,up,down}_exps.weight` names, so a 35B pack is mostly
   parameterization (experts.bin layout + native_experts.txt).
3. **quant kernels**: add PQ2_0 dequant + GEMV/MMQ beside the existing
   `dequant_s2` / `s2_gemv*` and `iq_kernels` families (AVX-512 + HIP).
4. **MTP**: Scion's drafter is the fork's sidecar, not the Flash-Next MTP —
   skip or port separately.

That is a fork-scale project (weeks), which is exactly why the dual-engine
contract is the cheap path: the cascade only sees endpoint + caps + measured
latency, so either engine can sit behind a tier.

### MTP speculation: what it actually takes (2026-10-02)

The GSQ-RCO GGUF **ships no MTP head**. The 31 `mtp.*` tensors (fc_embedding,
hyper-connection mixers, one MTP layer with its own attention/indexer/experts,
pre-fc norms) live only in the BF16 checkpoint `Qwen/Qwen3.8-Flash-Next`
(~360 GB, 131 shards), spread over 28 shards. Strata's `tools/mtp_fetch.py`
reads the safetensors headers and range-fetches exactly those tensors
(SHA256-pinned revision `de4b8e4d…`), ~6 GB. So this gap-closer is not a flag:
it needs (a) an equivalent range-fetch/pack step for the MTP block and (b) an
MTP draft path in the engine — our fork's `llama-mtp-sidecar` is a different
(Scion) head architecture. Strata's MIT implementation is the reference;
porting it with attribution, or adopting Strata as the Flash-Next serving
engine, are the two paths. Decide before the baseline run.

Model count: HF GGUF metadata reports **~176.9B tensor elements** (so the
handoff's ~180B is right; Strata's "125B" is not the count to use). Corrected
in the cascade registry (`flash-next`: 177B, Q2_0, 66 GB).

## E4 baseline results (2026-10-02)

Runner: `experiments/cascade/bench_e4_baseline.py` (the protocol below, one
sample per prompt, 10 Hz RAM/VRAM sampling). Artifacts:
`experiments/cascade/results/e4-baseline-20261002/` (`SUMMARY.md`, per-arm
`manifest.json`/`samples.csv`, `index.json` rows). Provenance: model sha256
`69820c02ec7d0b45ef2ebb19d6620299db749fe2aded7f39f93c6b88b199b720` (matches
the HF `lfs_sha256`), engine `engine/ember @ 6bfd7640c7ae`, binary sha256
`62f422b8f4b97ec7…`, ROCm 7.2.4, kernel 6.18.52.

Common flags: `-ngl 99 -c 4096 -np 1 -t 8 -ts 0.522,0.478 --split-mode layer
-lm mmap --lazy-mode on -fa auto`, `HIP_VISIBLE_DEVICES=1,0`,
`LLAMA_MOE_SLOT_STATS=1`.

| arm | slots | hit rate | cold pp 1K | cold pp 4K | warm pp 1K | decode1 | decode2 | cached | disk/gen | RAM min | VRAM pk c0/c1 | load |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| `-ncmoe 0` | – | – | 214 | 312 | 361 | 25.5 | 25.9 | 28.4 | 11 MB | 24.8 GB | 20.5/19.8 GB | 50 s |
| `-ncmoe 12 -msc auto` (448 slots, 7.1 GiB) | 448 | 95.2% | 142 | 285 | 434 | 23.9 | 25.3 | 26.3 | 10 MB | 26.0 GB | 19.6/19.0 GB | 35 s |
| `-ncmoe 24 -msc 256` (8.1 GiB) | 256 | 91.3% | 115 | 236 | 350 | 22.9 | 22.7 | – | 13 MB | 25.6 GB | 12.4/19.7 GB | 29 s |

Prefill speeds are t/s (actual prompt 925 / 3,084 / 167 tokens for the 1K / 4K /
185-char slices of `ternary-serve/wiki.test.raw`); decode is 256 tokens with
`ignore_eos`. Against the 2026-09-28 record: all-resident 25.9 vs 26.9 t/s
(run-to-run spread), `-ncmoe 12` decode 25.3 vs 24.5, hit rate 95.2% vs 87.9%
(auto-sized 448 slots vs 256), disk/gen 10 MB vs 46 MiB. Page cache was cold
for each arm's first request; cold prefill reflects that.

Incidents (both recorded under the run dir):

- **First `-ncmoe 0` attempt aborted by a driver reset** (`ncmoe0/INCIDENT.md`):
  the unused Raphael iGPU's SMU hung during the all-resident load; the two
  7900 XTs logged no ring timeouts. The iGPU has been disabled; the retry
  completed normally.
- **Slot cache + prompt cache crash** (`ncmoe24-msc256/CRASH.md`): a request
  with `cache_prompt=true` under `-msc 256` aborts the server with `Memory
  access fault ... Page not present` (rc -6). The arm was re-run with the
  `cached` phase skipped and completed cleanly (91.3% hits, decode2 22.7 t/s);
  the 18.4 t/s decode2 seen in the crashed run was a one-off stall. Open
  engine bug in the slot-cache/prompt-cache/graph-replay interaction.

Still open from the protocol: the **Strata reference run** (item 3) — bounded
feasibility test on this box next.

<details>
<summary>Protocol (as run)</summary>

```sh
LLAMA_MOE_SLOT_STATS=1 tools/backends/rocm/llama-server \
  -m ~/Desktop/work/models/qwen38-q2_0/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
  -ngl 99 -ncmoe 12 -msc auto -c 4096 -np 1 -t 8 \
  -ts 0.522,0.478 --split-mode layer -lm mmap --lazy-mode on -fa auto
```

Per arm: cold prefill (1K/4K), warm prefill, 256-token decode (x2) + cached
decode; record hit rates, host-RAM peak, VRAM peak, disk MB per generation.
Abort criteria: free RAM < 4 GB during load, OOM, driver reset.

</details>

Then the gap-closing ladder, in value order: MTP layer (fetch/pack, then
`--model-draft`/`--spec-draft-max` through the engine-args seam), CPU
co-execution of misses, prefill chunking + layer-ahead prefetch, KV
streaming/k8v4, dispatch-floor reduction, telemetry/calibration.

## Next steps (in value order; baseline measured 2026-10-02, Strata reference next)

1. **Kernel-count reduction / fusion** (all configs, biggest headroom). The
   budget note's candidates are still unfused; targets the ~5,000
   kernels/step dispatch floor. Expected to lift the 27 t/s all-resident and
   the 23–24 t/s cached arms together.
2. **Telemetry + tuning seam**: expose the slot cache hit/fetch rate and per-tier
   `tok_s`/`vram_gb` through `/metrics` and Hivebench `/v1/stacks/status`
   (currently `null`), so `-msc auto` and offload choices are validated by the
   harness instead of by hand.
3. **Prefill for cached layers**: pp 125–186 first-request vs 224 all-resident
   (warm harness 268–307); needed for long agent contexts, and it is the same
   bytes-over-disk problem.
4. **Device-side ensure + pinned gather** (recovers the 9–15% gap and
   re-enables graph capture): needs pinned banks (8 GiB works; 16 GiB risky on
   30 GB RAM) and the gather reads at ~12.5 GB/s. Only worth it if cached must
   equal all-resident exactly.

## Verification

```sh
pytest tests/unit/test_stack_schema.py tests/integration/test_stack_manager.py -q
tools/backends/rocm/llama-server --version

# 125B steady-state smoke (256-token decode; expect ~19–25 t/s depending on offload)
LLAMA_MOE_SLOT_STATS=1 tools/backends/rocm/llama-server \
  -m ~/Desktop/work/models/qwen38-q2_0/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
  -ngl 99 -ncmoe 24 -msc 256 -c 4096 -np 1 -t 8 \
  -ts 0.522,0.478 --split-mode layer -lm mmap --lazy-mode on -fa auto
```

- Engine A/B setup: stock binary built from `origin/master` in
  `~/Desktop/work/prism-stock` (worktree), same cmake flags as the installer.
- `experiments/stack_ab.py --live` cannot A/B 125B-class arms (one applied
  stack at a time; two servers do not co-reside in 40 GiB) — the A/B is
  sequential: apply → bench → unload → swap binary → apply.
- The 125B has no stock arm: without the HSA bounce the dual-GPU load hangs
  (`LOAD-HANG-RESULTS.md`); not reproduced to protect driver state.
