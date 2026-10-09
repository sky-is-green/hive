# strata-bridge — run Scion GGUF releases on the Strata engine

[Strata](https://github.com/Niko1221/Strata) serves Qwen3.8-Flash-Next on
consumer hardware (expert offload to RAM/CPU, the 28.8 GB PLE table streamed
from SSD, MTP drafting). Its native expert kernels and pack format take the
legacy **Q2_0** container (ggml type 42: 64-element blocks, fp16 scale, 2-bit
codes). Scion releases ship the same ternary values in **PTQ1_0** (type 143,
base-3 packed) or **PQ2_0** (type 142), which Strata cannot read.

`repack_experts_q2_0.py` rewrites **only** the expert tensors into Q2_0 and
streams every other tensor byte-for-byte (it never holds a whole tensor, or
the 28.8 GB PLE table, in memory). The transcode is **lossless at the
decoded-value level**: PTQ1_0/PQ2_0 and Q2_0 all store
`value = (code - 1) * fp16_scale`; a 128-block becomes two 64-blocks with the
scale duplicated. PTQ1_0 trit unpack reuses `scion/moe/ptq1_0.py`, the tested
1:1 transcription of the C++ `dequantize_row_ptq1_0`.

## Usage

```sh
PY=~/Desktop/work/.venv-rocm/bin/python     # needs torch (scion codec)

# codec self-test (synthetic trits through pack -> repack -> decode)
$PY repack_experts_q2_0.py --self-test

# census + plan only
$PY repack_experts_q2_0.py --in Scion-...-Q6K.gguf --dry-run

# rewrite + sampled bit-equality verification
$PY repack_experts_q2_0.py --in Scion-...-Q6K.gguf --out Scion-...-Q2_0.gguf
```

## Pipeline into hivebench

1. **Repack** the release: `repack_experts_q2_0.py --in Q6K --out Q2_0`
   (~+29% on expert bytes: 1.75 -> 2.25 bpw for the transcoded tensors).
2. **Pack** for the engine:
   `strata/tools/iq_pack.py --gguf <Q2_0 file> --out ~/Desktop/work/strata-data/packs/<name>`
   (add `--compat-bf16` when the small projections are not already in the
   float forms Strata's kernels read; the pack records every conversion in
   `conversions.json`). The pack exports its own tokenizer from the GGUF.
3. **Config**: `strata/strata-scion-176b.json` (modeled on the measured
   `strata-q2_0-coexist.json`; `--native` and `--ple-gguf` both point at the
   single repacked file; no `--mtp` until an MTP runtime is packed for it).
4. **Stack**: `hivebench/stacks/scion-flashnext-strata.json`; the managed
   launcher (`harness/stack/strata.py`) adopts or starts it, syncs the port,
   and points the engine log at `logs/`.
5. **Test**: `experiments/cascade/bench_strata.py` (engine timing, RAM/VRAM,
   MTP totals) and `experiments/cascade/community_eval.py` through the
   OpenAI-compatible endpoint.

## Caveat: corrections

Scion releases embed trained correction branches (rank-512 adapters + router
deltas) that are applied by the scion fork's runtime at the MoE output.
Strata does not apply them, so a Strata-served Scion model runs **without
corrections** until that hook is ported (the adapter tensors are ignored /
carried as inert extras). For a quantization-axis comparison against ISTA's
GSQ-RCO files (which ship no corrections) this is still apples-to-apples.
