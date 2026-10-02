# Engine crash — slot cache + prompt cache (`cache_prompt=true`), 2026-10-02

**Finding: `-ncmoe 24 -msc 256` + a request with `cache_prompt=true` aborts
`llama-server` with a GPU page fault.** The arm otherwise ran normally
(decode1 23.6 t/s, matching the 2026-09-28 record of 23.0).

## Evidence

`server-crash.log`, task 529 (the `cached` request, `cache_prompt=true`,
`f_sim_best = 1.000` — full prompt reuse):

```
slot launch_slot_: id 0 | task 529 | processing task, is_child = 0
Memory access fault by GPU node-2 (Agent handle: 0x557178f29f60) on address
0x7fdb0e60c000. Reason: Page not present or supervisor privilege.
```

Process exit: `rc = -6` (SIGABRT). The fault kills the process before the
context destructor runs, so the `LLAMA_MOE_SLOT_STATS` hit/fetch line never
printed for that run.

## Context

- Same flags as the 2026-09-28 steady-state runs, plus this session's runner
  request order; `cached` is the only request with `cache_prompt=true`.
- The engine's documented constraint (`tools/engine-rig/notes/EMBER-20260928.md`):
  a GPU kernel cannot read unpinned mmap memory on this box ("Page not
  present"); the slot cache is supposed to only touch host banks through H2D
  copies. The fault address (`0x7fdb0e60c000`) is a host-range address.
- Decode2 in the same run showed a mid-generation dip to ~10 t/s for one 3 s
  window (`tg_3s = 9.92`) before settling at 18.4 t/s — recorded, cause not
  diagnosed.

## Status

Reproduction pending (the fix-up run skips `cached`). If it reproduces, this
is an Ember-lane engine bug in the prism fork (`engine/ember`), likely in the
slot-cache/prompt-cache/graph-replay interaction; the `graphs reused = 507`
line in the decode path is worth a look. Recorded for the engine workstream;
no fix attempted in this session.
