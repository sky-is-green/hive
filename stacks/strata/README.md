# Strata deployment configs (Hive-owned)

These are Hive's copies of the Strata server configs. They are **not** engine source and are not
read from the Strata checkout. The stack that uses one sets `config_external: true`, so the
launcher resolves it against the Hive repo root.

| File | Use |
|---|---|
| `configs/scion-flashnext-176b-a6b-q6k.json` | direct 176B Q6K/Q4_0 release; `TQ1_0_g128` experts, Q4_0 PLE, embedded corrections |
| `configs/scion-flashnext-176b-a6b-q2_0.json` | legacy Q2 bridge (Q2_0 experts, Q5_0 PLE), kept as the comparison baseline |

## Environment

The configs use `$VAR` placeholders, expanded by `StrataLauncher.build_config` before the config is
written:

| Variable | Meaning |
|---|---|
| `STRATA_REPO` | the Strata checkout (engine, `serve/server.py`, `data/`) |
| `SCION_MODELS` | the model directory holding the GGUF files, `pack-q6k`, `pack-q2-ab`, `scion-embd-bf16.gguf` |
| `STRATA_DATA` | `strata-data` (MTP and expert-profile data) |

The engine `exe` is `build-hip-0.2.10/strata`, resolved against the config's `cwd`
(`$STRATA_REPO`). A packaged engine release can replace it with an absolute path.

## Stacks

- `stacks/scion-flashnext-strata.json` — the direct Q6K lane.
- `stacks/scion-flashnext-strata-q2.json` — the legacy Q2 bridge lane.

Only one of them can own port 8081 at a time. The stack launcher adopts an already-running server
at the endpoint instead of killing it; `hive stop` only stops a process Hive started.

## Chat template

The Q6_K GGUF carries no `tokenizer.chat_template`. Build the pack with the Qwen3.8 template:

```sh
python3 tools/iq_pack.py --gguf "$SCION_MODELS/Scion-FlashNext-176B-A6B-Q6_K.gguf" \
  --out "$SCION_MODELS/pack-q6k" --compat-bf16 \
  --chat-template "$SCION_MODELS/pack-q2-ab/tokenizer/chat_template.jinja"
```

A pack with the one-line placeholder (`{{ messages[-1].content }}`) makes thinking mode repeat the
prompt. The local `pack-q6k-ppl` was repaired in place with the Q2 template; `--chat-template`
keeps a rebuild from reintroducing the stub.

## VRAM reserve

The display is on DRM card1, which is HIP **CUDA0** (the first layer-split stage). Keep the split
reserve:

- `--vram-reserve-mib 5120` — first stage / display card
- `--vram-reserve-later-mib 1024` — later stage / non-display card

Measured with the Q2 pack loaded: display **2393 MiB free**, non-display **760 MiB free**. A single
`--vram-reserve-mib 3072` left the display card at 765 MiB, which is too close to an OOM.

## Model manifest

`stacks/manifests/scion-flashnext-176b-a6b.json` records the model, container types, pack paths and
the minimum engine release. The SHA-256 is intentionally blank until the direct Q6K file is
downloaded and hashed; do not fill it from the Q2 bridge.
