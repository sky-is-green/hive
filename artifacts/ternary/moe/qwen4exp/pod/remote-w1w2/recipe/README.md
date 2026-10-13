# Flash-Next PLE pod recipe - cap_eval / serving a 65.8 GB GGUF

Proven 2026-10-04 after a painful session. Use this instead of rediscovering.
`$Q` = `hivebench/artifacts/ternary/moe/qwen4exp`. Scripts live next to this file.

## P5 shrink additions (2026-10-10)

- `router-stats.patch` - fork patch 4: env-gated per-expert routing counts,
  gate mass, full-softmax mass and the **REAP signal** (gate x expert-output
  norm, plus norm moments) in the MoE router (`SCION_MOE_STATS=<path>`,
  optional `SCION_MOE_STATS_EVERY`). Zero overhead when the env is unset
  (re-verified after the REAP change on the 4l proof model: identical output,
  no stats file, same wall time).
- `moe-stats-run.sh` + `router-stats-client.py` + `pod-moe-stats-run.sh` +
  `pod-moe-stats-book.sh` - collect the expert usage/importance map on the
  coder calibration corpus (`scion/moe/build_coder_corpus.py`: Evol-CodeAlpaca
  teacher-forced + on-policy generations, glaive tool-calling, fineweb/GSM8K
  ballast; ~3.3M tokens). Output `router-stats.json` (format v2). The driver
  takes `CORPUS=` to pick a different corpus file.
- `kld-gate.patch` - fork patch 5: `llama-kld`, a new tool that computes the
  full-vocabulary KLD of any GGUF against the parked teacher log-probs
  (`scion/moe/make_kld_teacher_park.py` converts `tcache-48l` and regenerates
  the exact token windows). Math verified against `moe/kld_eval.py`'s formula
  on perturbed parks.
- `ple-table.patch` - fork patch 6: the Q3X32/Q2X32/Q4X160 PLE table types
  (ids 144/145/146, CPU + CUDA `get_rows`, learned codebooks + per-block
  least-squares scale) and `tools/ple-table/ple-table-parity`. Design and
  local validation: `scion/docs/PLE-TABLE-TYPES.md`; healing plan:
  `scion/docs/PLE-HEALING-PLAN.md`. Applies after patch 5; the exporter side
  is `tools/strata-bridge/repack_experts_q2_0.py --table-type
  q3x32|q2x32|q4x160` (streamed, never materialises the table).
- `pod-build-pruned.sh` + `prune-gate-run.sh` + `pod-prune-gate-run.sh` +
  `pod-prune-gate-book.sh` - gate a pruned export: KLD on the release (same
  instrument, first run), prune export via
  `hivebench/tools/strata-bridge/repack_experts_q2_0.py --prune-plan ...
  --preserve-experts`, then KLD + PPL 580 + HS 400 + WG 400 on the pruned file.
- `verify-zero-overhead.sh` - the env-off equivalence/timing check.
- `community-run.sh` / `pod-community-run.sh` now apply `router-stats.patch`
  as the 4th patch, so every future window carries the instrumentation (inert
  when the env is unset).

## Shape (works; $0.82-1.09/hr)
- SECURE, gpuTypeIds `[RTX 6000 Ada, L40, L40S]`, pinned `dataCenterIds`
  `[US-KS-2, US-TX-4, US-GA-2]`, minRAMPerGPU 180, minVCPUPerGPU 16,
  containerDisk 100 GB + volume 100 GB at /workspace,
  image `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`, ports 22/tcp,
  SSH keys via `env.SSH_PUBLIC_KEY`/`PUBLIC_KEY`.
- Booking: `recipe/pod-one.sh` (writes /tmp/opencode/pod-id + pod-endpoint,
  runs the driver on success).

## Steps
1. **Endpoint wait (max 25 min)**: poll GraphQL
   `pod(input:{podId}) { runtime { ports { ip publicPort privatePort type } } }`
   for `privatePort == 22`; on timeout delete the pod and retry later.
2. **Build the fork CPU-only FIRST and verify the binary** (before any large
   upload). CPU-only has no nvcc dependency and is fully sufficient:
   with 188+ GB host RAM the whole GGUF is page-cached.
   `cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF &&`
   `cmake --build build -j$(nproc) --target llama-server`
3. **Upload in PARALLEL CHUNKS** (single stream caps at ~14 MB/s on this
   route; 2 streams = 25 MB/s; 10 streams ~ pipe limit):
   - `split -b 7G -d --suffix-length=1 file /tmp/opencode/chunk`
   - launch all `scp -q -o Compression=no` at once, then `wait`
   - on the pod: `for f in $(ls chunk* | sort); do cat "$f" >> final && rm -f "$f"; done`
     (append+delete keeps peak disk = final size; fits the 100 GB volume)
4. **sha256 verify** the concatenated file against the local expected sha
   before evaluating.
5. **cap_eval**: `--backend server --server-bin build/bin/llama-server
   --server-args "-c 512 -t 16" --server-log ... --out /root/cap-eval-48l-ple.json
   --timeout 1800`.
6. Fetch JSON + server log, print score, **delete the pod**, verify the API
   shows no pods.

## Traps (all paid for)
- **Bundle tar**: `--exclude='build*'` drops REAL source
  (`common/build-info.cpp.in`, `cmake/build-info.cmake`) -> cmake
  "configure_file Problem configuring file". Exclude only
  `llama-qwen4exp/build` and `llama-qwen4exp/build-*`.
- **rsync --partial after a kill**: the receiver RENAMES the hidden temp to
  the destination name when the transfer is interrupted -- so a partial DOES
  survive (usable with `--append-verify` or by appending the remainder), but
  it is under the destination name, not `.name.XXXXXX`.  Check both; do not
  conclude "no partial" from a hidden-file glob alone.
- **Local `/tmp` is a 16 GB tmpfs on this desktop**: never stage the 65 GB
  chunks there (`split` dies with ENOSPC after ~15 GB).  Stage on disk:
  `$WORK/tmp-export/` (root btrfs, ~150 GB free).
- **Single-connection upload cap ~14 MB/s** (consistent across 3 sessions);
  speedtest numbers are irrelevant. Parallelize.
- **Escaped-quote f-strings in watcher one-liners** (`f"{x[\"k\"]}"`) fail
  silently under `except: pass`; use `print(x["ip"], x["port"])`.
- **RunPod provisioning outage signature**: pod id + billing, but
  `publicIp` empty / `runtime: null` for 25+ min (status page: "elevated
  errors on GraphQL and REST v2 APIs"). Delete, wait, retry - do not keep
  re-booking.
- **Background work dies on harness restarts/reboots**: keep
  book -> endpoint -> build -> upload -> eval in ONE background script; keep
  the dead-man switch (`sleep 3.5h; DELETE /v1/pods/<id>`); never rely on
  /tmp files persisting.
- **Sequencing**: build before upload; on build failure KEEP the pod and
  print the build log (debug in place); on sha/eval failure delete it.
- **`wait` waits for ALL children** - including a
  `( sleep 3.5h; DELETE pod ) &` dead-man.  A finish script hung 75 min on
  exactly this.  `disown` the dead-man (or skip `wait`, or use `wait -n`
  with explicit PIDs).
- **`/tmp/opencode` is not durable**: harness restarts and reboots wipe it
  (and killed a mid-flight workflow twice today).  Keep the ONE workflow
  script self-contained; re-stage inputs under a durable path if needed.
- **Local serving on the 30 GB desktop: don't.** Measured kernel readahead
  amplification: 176 GB read for a 24-token completion (~7.3 GB/token), the
  28.8 GB table itself never touched; it stalls the desktop. Mitigation if
  unavoidable: `--numa distribute` (MADV_RANDOM on the mapping) +
  `ionice -c 3`. GPU offload is WORSE (host body pages are still touched for
  GPU-assigned layers).
- **Never materialise a whole GGUF locally** (past OOM-killed the desktop).

## Scripts
- `pod-one.sh` - book + endpoint wait + chain.
- `pod-cap-run2.sh` - driver: apt/source/build-first, single-stream rsync,
  cap_eval, fetch, delete (build failure keeps the pod).
- `pod-upload-parallel.sh` - CURRENT BEST: parallel chunked upload, concat,
  sha, cap_eval, fetch, delete.
- `pod-cap-all.sh` - self-contained book->eval in one script (older flow).
- `pod-parallel-finish.sh` - chunked *remainder* finish for an interrupted
  single-stream upload (only useful if the partial survived; usually it does
  not).
- `probe-cpu-moe.sh` - local serving timing probe (readahead trap).
