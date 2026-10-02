# Strata first-load abort — VRAM exhaustion + swap pressure (2026-10-02 ~23:04–23:07)

**Verdict: the first Strata Q2_0 load did not become ready.** The engine died
while filling the expert cache with `amdgpu: Not enough memory for command
submission` on the display card (`0000:03:00.0`), after the box had swapped.
No kernel OOM kill and no systemd-oomd kill was logged.

## Setup result (all successful)

- Engine compiled for gfx1100 from source with system ROCm 7.2.4:
  `engine/strata` (sha256 `9c91cbfb88b3ddc8…`), Strata commit
  `db4f91a1171d697928b0d2e1f50ef95e25559d4c`.
- Q2_0 pack built from our canonical GGUF (`--gguf-dir`): `experts.bin`
  31.64 GiB, `dense.bin` 5.44 GiB, `embd.bin` 0.44 GiB, tokenizer.
- MTP layer fetched from the pinned BF16 revision and packed:
  `strata-data/mtp/rt` (31 tensors, 5.21 GB raw, q8_0 runtime).
- Config: `strata-q2_0.json` (copy here): `--mmap-experts --expert-cache auto
  --prefill auto --spec 4 --spec-min-p 0.5 --mtp … --max-context 4096`,
  layer split auto across GPUs [0, 1].

## Failure timeline (engine log + server out + journal)

| time | event |
|---|---|
| ~23:04 | server starts; engine loads split K=26, PLE on, MTP 799 MiB VRAM |
| | `expert cache auto: 16.32 GiB free, 700 MiB reserved -> 12142 slots` (15.63 GiB) |
| | server: "filling the GPU's expert cache (12142 experts, 15.63 GiB) …" |
| 23:06:49 | kernel: `amdgpu 0000:03:00.0: [drm] *ERROR* Not enough memory for command submission!` |
| ~23:07 | engine exits before ready; server raises `RuntimeError: the engine exited before it was ready` |
| | Firefox/Discord/electron had reopened; swap in use ~3.4 GB; no OOM-kill lines |

The engine's auto cache sizing (16.32 GiB "free") did not leave the display
card enough headroom; card1 drives the desktop (Xwayland + browser GPU memory)
and was the device that failed. The mmap'd expert pages also pushed the box
into swap with the desktop apps back.

## State after cleanup

Engine + server stopped (pkill), port 8080 free, MemAvailable ~26.8 GB,
VRAM idle (card1 ~0.8 GB desktop). No processes were lost to a kernel OOM.

## Retry options (not yet done)

1. Free the box (close desktop apps) and add `--vram-reserve-mib 2048` (or
   more on the display card); keep `--max-context 4096`, `--prefill 512`.
2. Put the headless card first in the split (`--gpus 1,0`) and/or an explicit
   `--layer-split`, so the display card holds fewer weights.
3. Single-GPU engine (`--gpus 0`) in mapped mode: fewer VRAM bytes per card,
   more mmap traffic; the same RAM profile.
