# Serving GGUFs on AMD (RDNA3, one or two cards)

This is the shortest path from a clone to an OpenAI-compatible endpoint serving a GGUF on an AMD GPU.
It uses Hive's engine slots: the stack names a role and a backend, the harness owns the process.

Verified on 2× RX 7900 XT (gfx1100, ROCm 7.2.4, kernel 7.3): a one-tier stack with the `scion` build loaded a
10.5 GB GGUF in ~5 s, answered `/health`, printed the endpoint, and unloaded on `--once`. The same model with
`pin 0,1` and `ts 1,1` loaded **across both cards** — 5.58 GiB on card 0 and 6.98 GiB on card 1 — answered
`/v1/models`, and unloaded on SIGINT, returning both cards to their baseline.

## 1. Prerequisites

| | |
|---|---|
| GPU | any RDNA3 card (gfx1100 / gfx1101 / gfx1102). ROCm 7.x with `rocminfo` seeing the card |
| Python | 3.12+ |
| Disk | the GGUF plus a few GB |
| Build tools | not needed for the portable Vulkan build; the custom ROCm/Scion builds need them |

```bash
git clone https://github.com/sky-is-green/hive.git
cd hive
python3 -m venv .venv
.venv/bin/pip install -e .          # serve-only; no Studio/eval extras needed
```

The base install deliberately omits FastAPI, pytest, datasets and the splinter
memory system.  An installed wheel writes under `HIVE_HOME` (default
`~/.hive`); the paths below assume the source checkout.

## 2. Install a llama-server build

Hive launches one process per tier from `tools/backends/<backend>/llama-server`.

```bash
.venv/bin/hive setup                   # fetch the portable engine + create paths
.venv/bin/hive backends                # what is declared, what is installed
```

`hive setup` is the one-command onboarding; it fetches the portable Vulkan
build if no engine is installed. The same action on its own is
`hive backends fetch vulkan`.

That downloads the latest upstream `llama-server` build for this platform,
verifies it with `--version`, and installs it under `tools/backends/vulkan`.
No compiler and no ROCm version match are needed. The custom ROCm/Scion builds
remain the advanced path:

```bash
scripts/install_amd_engine.sh          # builds/links the prism ROCm engine as "rocm"
```

Expected shape after the fetch:

```
  vulkan   installed  amd,nvidia,intel     Vulkan
  rocm     installed  amd                  ROCm (HIP)
  cuda     -          nvidia               CUDA
  cpu      -          cpu                  CPU
  sycl     -          intel                SYCL (oneAPI)
  scion    installed  amd,nvidia           Scion (moe-corr-runtime)
```

`-` means the binary is not there. `hive serve` only picks an **installed** backend; if none suits the
accelerator it falls back to the default `tools/llama.cpp/llama-server` rather than selecting one that would
fail at load time.

## 3. Serve a model

```bash
# one card: the GGUF from HuggingFace, first card, backend auto-picked
.venv/bin/hive serve --model TheBloke/Some-Model-GGUF:model.Q4_K_M.gguf --cards 1

# two cards: split the model across both, write the stack for later
.venv/bin/hive serve --model TheBloke/Some-Model-GGUF:model.Q4_K_M.gguf --cards 2 --save two-card
.venv/bin/hive serve two-card

# a local GGUF you already downloaded
.venv/bin/hive serve --model /models/model.Q4_K_M.gguf --cards 1

# background: returns when the endpoint answers
.venv/bin/hive serve two-card --detach
.venv/bin/hive status
.venv/bin/hive logs two-card --tier face
.venv/bin/hive stop two-card
```

`--cards 2` writes:

```json
{"role": "face", "repo": "...", "file": "model.Q4_K_M.gguf", "ctx": 8192, "ngl": 99,
 "backend": "rocm", "pin": "0,1", "ts": "1,1"}
```

- `pin` is `HIP_VISIBLE_DEVICES`; the indices come from `harness.hardware`, which is BDF-ordered — the order HIP
  assigns device numbers.
- `ts` is the tensor split, proportional to VRAM (`1,1` for equal cards).
- `--devices 1` pins one specific card; `--ctx` sets the context.
- The display card is allowed but warned about: the engine and the desktop will fight for VRAM.
- A `repo:file` model is downloaded into the models library before the stack is applied; the first serve prints
  progress on stderr. `HF_TOKEN` is used for gated repos.

Or write the stack by hand (`stacks/my.json`) and run `hive serve stacks/my.json`; `--once` applies, prints the
endpoints, and unloads — a smoke test.

## 4. Check it

```bash
.venv/bin/hive status                  # endpoints, ctx, backend, up/down
.venv/bin/hive doctor                  # cards, backends, paths, ports
curl -s http://127.0.0.1:1234/v1/models | head -c 400
curl -s http://127.0.0.1:1234/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"x","messages":[{"role":"user","content":"Say hi."}],"max_tokens":16}'
```

## 5. Two cards: what to expect

Two cards help in one of two ways:

* **Capacity** — `--split-mode layer --tensor-split` (what `--cards 2` generates) lets a model larger than one
  card's VRAM load at all. This is the common reason to use two cards, and the default path here.
* **Throughput** — not automatic. Layers are sequential, so a layer split does not make one response faster; the
  cards take turns. Two independent streams (one per card) need either two stacks or a shared model loaded twice,
  and on a 31.7 GiB box two copies of a large model did not fit (the guard stopped them at ~14 GiB of GTT).

Two cards are also not symmetric on every machine: check the PCIe link of each card (`lspci -vv` on the bridge,
not just the endpoint). On this box the second card sits behind a PCIe 4.0 x2 bridge; that is fine for local
compute but rules out designs that stream weights across cards.

## 6. The big-model lane (Strata / DeepSeek-V4.1)

`stacks/ds41-strata.json` is an endpoint tier with a `launch.kind: strata`: it starts Strata's server, which in
turn drives this repo's DS4.1 engine. That lane exists for a 163 GB GGUF on a 31.7 GiB box (NVMe streaming,
4 GiB expert arena); it is not the default and it needs the model and the engine build. Treat it as the showcase,
not the starting point.

## 7. Troubleshooting

| symptom | cause / fix |
|---|---|
| `no llama-server binary for backend 'rocm'` | the build is not installed. Run `hive backends fetch rocm`, `hive backends fetch vulkan`, or use `--backend vulkan` |
| `invalid ggml type 142` | the model uses a quant the chosen build does not know (PQ2_0 needs the `scion` fork) — set `"backend": "scion"` |
| OOM at load | lower `--ctx` first (the KV cache), then `ngl` |
| desktop stutter | the tier is pinned to the display card; use `--devices 1` |
| endpoint answers from the wrong model | add `expect_model` to the tier; `apply` refuses a mismatch |
| `--cards 2 but only 1 GPU detected` | `harness.hardware` reads amdgpu sysfs; check `rocminfo` and `HIP_VISIBLE_DEVICES` |
| a detached serve will not stop | `hive stop <name> --force`; `hive status` shows whether the PID still matches |
| a load failed after the endpoint was up | `hive logs <name> --tier <role>` — the engine's startup error is in that file |

## 8. Notes from a sanitized tree

Local paths in saved stacks may appear as `<work>` placeholders (this repo replaces them before publishing).
Replace them with real paths, or generate a fresh stack with `--save`.
