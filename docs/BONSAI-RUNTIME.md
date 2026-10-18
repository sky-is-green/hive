# Bonsai Runtime (T29) — released Bonsai-2 27B `PQ2_0` on the Prism ROCm fork

Plan-B end-state: run Prism's released ternary artifact as-is and evaluate it with
hivebench. This is the pragmatic route to the user's goal — the quality lives in
their trained weights (Gate 3), so no local training is required.

## Artifact

| Item | Path |
|---|---|
| Model | `artifacts/ternary/oracle/bonsai27/Ternary-Bonsai-2-27B-PQ2_0.gguf` (7.2 GB, sha `3907dc16…`) |
| Fork binary | `artifacts/ternary/oracle/prism-fork/bin/llama-prism-b10709-9a9394a/llama-server` |
| Driver | `experiments/ternary_eval.py` (T11) |

## Serve / evaluate

```sh
cd ~/Desktop/work/hivebench
FORK=artifacts/ternary/oracle/prism-fork/bin/llama-prism-b10709-9a9394a
PIN=~/Desktop/work/worktrees/splinter-memory/hivebench-SPLINTER-PIN

HIP_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=$PWD/$FORK \
SPLINTER_HOME=$PIN PYTHONPATH=$PIN/splinter \
  ~/Desktop/work/splinter-memory/venv/bin/python -m experiments.ternary_eval \
    --gguf artifacts/ternary/oracle/bonsai27/Ternary-Bonsai-2-27B-PQ2_0.gguf \
    --fork-bin $FORK/llama-server \
    --no-thinking --max-convs 10 \
    --output artifacts/ternary/eval/bonsai27-hivebench.json
```

- `--no-thinking` passes `--chat-template-kwargs '{"enable_thinking": false}'` to
  `llama-server`; without it the reasoning template returns empty visible content.
- `HIP_VISIBLE_DEVICES=1` keeps the desktop GPU0 free; the 27B `PQ2_0` fits one
  RX 7900 XT at `-ngl 99` (the driver's default), so both cards are not needed.
- **T23 workaround:** the sibling `splinter-memory` main no longer defines
  `cortex.config.SplinterConfig`, so the harness import fails unless the F6 pin is
  on `PYTHONPATH` (`$PIN/splinter`) with `SPLINTER_HOME=$PIN`.

## Results (2026-09-20)

PPL under the Gate-2 protocol (tinyshakespeare, ctx 512, 8 chunks): **18.5851**.

Hivebench eval, `--max-convs 10`, 124 turns compared:

| Metric | hive | FIFO |
|---|---|---|
| answer recall | 72.6 | 77.4 |
| avg fact hit ratio | 0.731 | 0.789 |
| avg context fidelity | **0.306** | 0.205 |

- 5/5 smoke prompts PASS (coherent: `Paris`, `51`, `ternary.`); ~0.5–0.9 s/reply.
- `fidelity_hive_gt_fifo_ratio` 69.4%, `hive_ge_fifo_ratio` 89.0% — the memory
  layer improves context fidelity over FIFO while trailing slightly on raw answer
  recall. `hive_only` 4, `fifo_only` 10, `both_sufficient` 77.

## CPU-only partner probe (coexist with the 176B serve)

The same GGUF also runs under the generic CPU build. The 176B serve owns the
GPUs, so a partner run must hide every accelerator. Do **not** rely on
`-ngl 0` alone: the first Scion attempt still assigned a fused op to ROCm0.
Use `HIP_VISIBLE_DEVICES=""`, `--device none`, and `--no-op-offload`, plus the
preflight/guard in `experiments/cascade/run_partner_probe.sh`:

```sh
cd ~/Desktop/work/hivebench
./experiments/cascade/run_partner_probe.sh \
  artifacts/ternary/oracle/bonsai27/Ternary-Bonsai-2-27B-PQ2_0.gguf \
  experiments/cascade/results/partner-bonsai-organic-20261018/bonsai-transfer-bench.jsonl \
  bonsai-transfer experiments/cascade/results/partner-bonsai-organic-20261018 8899
```

The runner refuses to start below 12 GiB available RAM or if the 176B health
endpoint is down, samples RAM/VRAM/load/RSS/176B health into
`stability.csv`, stops the partner below 4 GiB available RAM or after 60 s of
176B health failure, and always writes a stability summary. The Bonsai 27B
dense model is much slower on CPU than on GPU: expect tens of seconds per
judge prompt at 8 threads. The transfer baseline and stability numbers for the
2026-10-18 run are in
`experiments/cascade/results/partner-bonsai-organic-20261018/BONSAI-TRANSFER.md`.

Observed on 2026-10-18 (140 prompts, `probe_rc=0`):

- 2,502 s total (~42 min), dim 5120, no GPU context.
- min MemAvailable 13.478 GiB, max partner RSS 13.576 GiB, max swap 3.801 GiB.
- 176B health failures 0; min free VRAM card0 2799 MiB / card1 1886 MiB.
- Frozen J-Bench head sanity exact (AUC 0.9881); CPU/GPU state parity on 4
  re-embedded rows: mean cosine 0.999957.
- Judge transfer: J-Bench 0.988, QA clean 0.995, organic 0.624. The head is a
  structured/QA verifier, not an organic one — see the report's
  "Reading this result" and `human-gate-missed.jsonl`.

## Status

T29 runtime + eval verified. CPU-only partner coexistence is guarded and
reproducible; the remaining optional work is LoRA adaptation around the
released weights and a 2-card throughput pass if serving latency matters.

Update 2026-10-18: the judge line is closed for organic tasks.  A second
Bonsai head — the community in-domain fit (natural answers, byte-identical
judge prompt) — lands at 0.630 organic AUC on the harvest states, the same
chance region as the frozen J-Bench head (0.624), so the failure is
cross-generator transfer, not an under-trained head; the deferred 7 h harvest
extraction (an in-domain refit on Bonsai-reads-176B errors) is not justified
as a judge.  The stronger on-disk instruments are the 0.8B MLP head (organic
0.844, bench 0.991) and the generator's own 176B J-Bench head (bench 0.9941);
see `experiments/cascade/results/fusion-probe-20261018/FUSION-PROBE.md`.  Any
future role for the dense body is repair/generation, not verification.

Update 2026-10-18 (Window A, 7 min 13 s serve-down): the generator's own readout was measured on
the same harvest slices through the decision-head build.  The in-domain own head is a real organic
instrument — 0.803 on organic-66, against 0.624/0.630 for the two Bonsai heads, and ~4 points under
the 0.8B MLP — and it is free at serve time (one engine pass, no second body).  The own J-Bench head
is the complementary QA instrument (0.979 on qa-75, 0.671 organic).  Fusion with the self signal
reproduces the previous best organic number (0.858) without the Bonsai body in the stack.  See
`experiments/cascade/results/scion176b-self-organic-20261018/SELF-HEAD.md`.
