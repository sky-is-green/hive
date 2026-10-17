# flashnext-176B serve tuning — 2026-10-17

Box: 2x RX 7900 XT (gfx1100, 20 GiB each), 7800X3D, 30 GiB RAM + zram swap.
Engine 0.1.44 (binary 2026-10-15 22:06), Q6K native + pack, expert-profile 24576 pairs,
MTP spec-4, reserve 3072, pcie-frac 0, pipeline-windows 2, max-context 4096.

## Baseline (morning session, warm)

| probe | prefill | decode |
|---|---|---|
| ~340 tok prompt | 115–120 t/s | 69–81 t/s |
| ~1000 tok prompt | 222–237 t/s | 94–101 t/s |
| ~2855 tok prompt | **365–369 t/s** | 92–100 t/s |
| 256-tok decode | — | 70–81 t/s |

Prefill scales with size (fixed ~2 s prompt-path startup + ~2.1 ms/token marginal).
Decode varies with draft acceptance (49 t/s reasoning-heavy → 100 t/s short answers).

## Window 1 — reserve 1024 (REVERTED, caused a system freeze)

- Load reached 13732+10752 slots (100% residency, 100.0% of routed mass).
- The bench then thrashed the box: first request read 150 tokens in 36.9 s (normal 2.2 s),
  decode decayed 26 → 5 → 0.6 tok/s; kwin_wayland hung; machine froze at 02:05:15
  (hard reset; no oomd kill line reached disk). Last engine line: "the OS read 7079.5 MB
  from storage" — page cache evicted, expert reads to SSD.
- Cause: the 3072→1024 change removed the ~2 GiB/card VRAM slack that absorbs prompt-path
  buffer churn; with ~8 GiB zram swap + desktop (VLC) in use, work fell into host RAM.
- The same bench ran clean at 3072 before and after. reserve stays **3072**.

## Window 2 — STRATA_DECODE_TIMING profile (findings)

Steady-state windows (profiling on, slightly pessimistic; K=26/27 split):

| request | ms/window | GPU-reach wait | CPU host | tokens/window |
|---|---|---|---|---|
| p1000b | 32.8 | 22.9 | 3.3 | 2.67 |
| p3000b | 34.9 | 22.5 | 4.6 | 2.91 |
| d256b | 33.7 | 21.2 | 4.8 | 2.85 |

- Engine's own load-time prediction: 38.1 ms/window. Actual: 33–35. **No pipeline leak.**
- The window is GPU-bound ("GPU-reach wait" ~22 ms = GPUs busy on the prior window's stage);
  host-side (pool jobs + CPU compute + swaps) is only 3–5 ms in steady state.
- VRAM hits ~99.6–100%; file tier ~zero in steady state. First boot after a reboot pays
  ~24 GB of expert reads (cache prefill + page-cache warm) — a one-time cost.

## Window 3 — spec 6 (REJECTED)

| | spec 4 | spec 6 |
|---|---|---|
| tokens/window | 2.6–2.9 | 3.2–3.4 (+18%) |
| ms/window | 33–35 | 42–47 (+30%) |
| decode | 81–85 t/s | 71–75 t/s (−12%) |

Longer verify windows cost disproportionately more GPU time. spec stays **4**, min-p **0.5**.

## Final state (restored byte-identical to the pre-tuning config)

Final bench (no profiling, fresh warm load): p3000 prefill 362.5 t/s, decode 69–90 t/s,
session drafts accepted 552/753 (73%), min available RAM 22.2 GiB.

Config left exactly as found: reserve 3072, spec 4, min-p 0.5, pcie-frac 0, pipeline-windows 2,
env STRATA_PREFILL_MMQ/RING/IO_THREADS only. Backup copy: `strata-q6k-176b.json.bak-20261017-reserve3072`.

## Window 4 — max context 262144 (KEPT, 15:32–15:40)

`--max-context` 4096 → 262144, config otherwise unchanged (backup:
`strata-q6k-176b.json.bak-20261017-maxctx-4096`). KV in this engine is fp16 (~28 KB/token,
~6.9 GiB at 256K measured by the slot delta; the docs' 13.7 KB/token is the 8-bit mode,
not the default here). `--kv-grow` is unavailable on a 2-GPU layer split, so the KV is static.

- Load: expert slots 22277 → 15343 (CUDA0 11745→8538, CUDA1 10532→6805, −31%); engine rates
  the caches 98.8% of the routed mass (was ~99.8%). Reserve untouched: ~3.0 GiB free per card
  with everything loaded.
- Short benches (two runs, warm): prefill identical (363–365 t/s @2855, 231 @1000, 115 @300);
  decode 71–87 t/s (was 69–90); 256-tok decode 73–83 (a cold first sample read 56–61;
  draft-acceptance noise). No measurable short-context regression.
- Long context (new capability):
  - 16K prompt, cold (first after restart): 505 t/s prefill in 30.9 s, 42.4 GB storage read
    (page-cache warm-up, one-time), decode 39.7 t/s, 94.4% decode hit.
  - 16K prompt, warm: 614 t/s prefill in 25.5 s, 0.76 GB read, decode 64.2 t/s, 98.9% hit.
  - Follow-up turn: 15623 tokens reused, wall 1.37 s (checkpoints work at 16K).
- RAM flat (~22 GiB available throughout, 5 s cadence); 1–2 checkpoints per 16K turn.

## Files

- `bench_flashnext.py` — repeatable bench (unique prompts per run, API timings + engine log
  lines per request; writes `results.json`).
- `bench_longctx.py` — 16K-prompt + follow-up probe for the 256K config (writes
  `results_longctx.json`).
- `results_baseline_v3.json` — morning baseline. `results_final_spec4.json` — final 4K-config bench.
- `results_longctx.json` — window 4 long-context run. `results_maxctx_short.json` — window 4
  short bench (run 2; run 1 numbers are in the engine log).
- `memwatch_spec6.log`, `memwatch_final.log` — RAM watch during benches (5 s cadence).
- `memwatch.log` — RAM watch during window 2. `memwatch_maxctx.log` — window 4.

## Left on the table (not config)

- Context bump 4096 → 16384 (queued, capability not speed; costs expert slots) — **superseded by
  window 4: now 262144.** Middle grounds if the ~42 GB first-touch read ever matters in daily use:
  65536 or 131072 (would free ~5.2/~3.4 GiB of the KV carve for expert slots; not measured).
- `--lookup-chain` (opt-in prompt-lookup drafts, up to 8-token windows) — untested.
- Engine-side kernel work — out of scope for serve config.
