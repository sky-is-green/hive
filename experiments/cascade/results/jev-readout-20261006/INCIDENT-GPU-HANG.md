# Incident 2026-10-06: non-display GPU hang during the Scion-35B J-Bench session

**What:** during the Scion-35B generator session (fork `llama-server` on the
non-display card, `HIP_VISIBLE_DEVICES=1`, PCI `07:00.0`, card0), the server
logged `HW Exception by GPU node-2 ... reason: GPU Hang` after ~36 h of server
uptime? no — ~36 minutes of session traffic (prompted readout 756 requests,
embedding probe 378 requests, 100 answer generations, then the source-audit
embeddings).  `/health` stopped responding; the audit's embedding request
timed out.

**State:** the `llama-server` process (pid 501214) is unkillable (`kill -9`
leaves it in `Isl`), card0 `gpu_busy_percent` now reads *Operation not
supported*, VRAM stays at 12.3 GB, `power/control=on` (runtime PM was already
disabled per the 2026-10-04 prevention).  card1 (display) is healthy
(13% busy).  This matches the 2026-10-04 wedge; reboot is the documented
recovery.

**Impact:** none on collected data.  The session's main artifacts are on disk
(`readout-scion35b-prompted.json`, `probe-scion35b.json`, `scion-answers.json`);
only the source-swap audit (200 extra embedding requests) was interrupted.

**Recovery:** reboot, then
`source_audit.py --server http://127.0.0.1:8897` against a freshly started
server (or run the embedding leg on CPU with `HIP_VISIBLE_DEVICES=""`).

**Notes:** this was a single ROCm context (no torch beside the server).  The
new variable vs the earlier sessions is `/embedding` traffic mixed with
`/completion` on a 35B-A3B MoE (embeddings mode allocates another graph).
Consider separate servers for the embedding and completion legs, or the
CPU bridge, for the audit re-run.
