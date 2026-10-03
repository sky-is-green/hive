# clef-bridge — per-token hidden states from a GGUF backbone

The Cloudflare Clef releases are a Qwen3.5 backbone plus a **joint schema
head** that reads per-token hidden states, input ids/spans and the lm_head
weight.  A quantized GGUF backbone cannot run that head itself, so this bridge
decodes exact token ids in the fork (`/home/penis/llama.cpp`, the
`moe-corr-runtime` build with PQ2_0) and dumps `llama_get_embeddings_ith`
states for a Python head sidecar.

Verified exact on the f16 model: the dumped states match the HF backbone's
`last_hidden_state` at cosine 1.0000.

## Build

CPU-only (used for the ternary benchmark; no ROCm context):

```sh
P=/home/penis/llama.cpp
g++ -O2 -std=c++17 -I$P/include -I$P/ggml/include clef_embed.cpp \
  -L$P/build-cpu/bin -Wl,-rpath,$P/build-cpu/bin -lllama -lggml -lggml-base -o clef_embed
```

For a HIP build of the fork, use `-L$P/build/bin` instead.  CPU is mandatory
beside a resident serving engine (two ROCm contexts can hang this host).

## Use

```sh
# one record
./clef_embed model.gguf tokens.txt out.bin [n_ctx] [n_gpu_layers]

# batch: model loads once, fresh context per record (no state leaks)
./clef_embed model.gguf tokens_dir out_dir [n_ctx] [n_gpu_layers]

# serve: resident model for a harness (stdout protocol, see below)
./clef_embed model.gguf --serve [n_ctx] [n_gpu_layers]
```

- `tokens.txt`: space-separated ids (the Clef `encode_record` order).
- `out.bin`: `n_tokens * n_embd` float32, token-major.
- batch mode reads every `*.tok` in `tokens_dir` (sorted), writes `<stem>.bin`
  into `out_dir`, and prints `REC <stem> <n_tokens> <ms>` plus a `TOTAL` line
  with the one-time model load cost to stderr.
- serve mode keeps the model resident for a harness: stdout is the protocol,
  `READY <n_embd>` first, then one request per stdin line
  `<tokens.txt> <out.bin>` answered with `OK <n_tokens> <ms>` or
  `ERR <reason>` (one fresh context per request, like batch).  The
  `clef-ternary` harness backend (`harness/cascade/validator.py`) uses this.
- `n_gpu_layers`: `0` (default) keeps it CPU-only.

## Embedded adapters (single-file releases)

A body merged with `moe/merge_adapter_into_body.py` carries
`adapter.embedded=true`.  The fork attaches that adapter at model load, but a
raw-API context only picks it up when the caller registers the effective
adapter set.  This helper calls

```c
llama_set_adapters_lora(ctx, nullptr, 0, nullptr);
```

right after `llama_init_from_model`; that call adds the model's embedded
adapter at scale 1.0.  Without it the forward is bit-identical to the
uncorrected body (the adapter silently does nothing).

The head sidecar then runs `JointSchemaHead` from the Clef release with those
states, the same ids, attention mask ones, the record spans and the lm_head
weight.  See `scion/dense/clef_bridge_eval.py` for the full benchmark flow.
