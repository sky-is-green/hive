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

```sh
P=/home/penis/llama.cpp
g++ -O2 -std=c++17 -I$P/include -I$P/ggml/include clef_embed.cpp \
  -L$P/build/bin -lllama -lggml -lggml-base -o clef_embed
```

## Use

```sh
LD_LIBRARY_PATH=$P/build/bin ./clef_embed model.gguf tokens.txt out.bin [n_ctx] [n_gpu_layers]
```

- `tokens.txt`: space-separated ids (the Clef `encode_record` order).
- `out.bin`: `n_tokens * n_embd` float32, token-major.
- `n_gpu_layers`: `0` (default) keeps it CPU-only, which is mandatory beside a
  resident serving engine (two ROCm contexts can hang this host).

The head sidecar then runs `JointSchemaHead` from the Clef release with those
states, the same ids, attention mask ones, the record spans and the lm_head
weight.
