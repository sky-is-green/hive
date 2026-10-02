# Arm abort — `-ncmoe 0` (all-resident), 2026-10-02 ~22:03–22:11

**Verdict: driver reset (EMBER.md abort criterion).** The arm was killed by a
system reboot triggered by an iGPU (Raphael/gfx1036) SMU hang during the
all-resident load. No result is recorded for this arm from this attempt.

## Timeline (previous boot, `journalctl -b -1`)

| time | event |
|---|---|
| 22:03:1x | `llama-server` starts: `-ncmoe 0`, both Navi 31 cards, `HIP_VISIBLE_DEVICES=1,0` |
| 22:03:5x | model loaded, listening (39.8 s load), first request (cold 1K) starts |
| 22:04:11 | `systemd-journald: Under memory pressure, flushing caches.` (twice) |
| 22:04:41 | `amdgpu 0000:7d:00.0: SMU: I'm not done with your previous command` / `Failed to export SMU metrics table!` (repeats every ~4 s) |
| 22:04:50+ | `timeout waiting for kiq fence` / `TLB flush failed for PASID 497` (repeats) |
| 22:04:19+ | first request crawls: 409 of 925 tokens at **1.95 t/s** (server.log, `t = 210.27 s`) |
| 22:11:20 | last journal line (System Monitor focused); box reset |
| 22:14:59 | new boot |

`0000:7d:00.0` is the **Raphael PCIe Dummy Function** (the iGPU's amdgpu
function group), not a Navi 31. The two 7900 XTs (`0000:03:00.0`,
`0000:07:00.0`) logged no ring timeouts. No OOM kill and no dGPU reset line in
the journal — the system became unusable through the iGPU SMU hang plus
memory pressure during the all-resident load.

## Evidence

- `ncmoe0/server.log` (this directory): load ok, first request at 1.95 t/s.
- `journalctl -k -b -1 --since "2026-10-02 22:03:30"`:
  memory-pressure notices at 22:04:11–13, iGPU SMU/kiq/TLB errors from
  22:04:41 to the end of the boot.
- Previous boot ended 22:11:20; next boot 22:14:59.

## Mitigation applied

After the reboot the iGPU is no longer present (`/sys/class/drm/card2` gone;
no `1002:164E` in `lspci`): the human disabled it. The iGPU was unused for
compute (ROCm torch 2.14+rocm7.2 carries no gfx1036 kernels), so nothing is
lost. The all-resident arm is to be retried on the iGPU-free boot.
