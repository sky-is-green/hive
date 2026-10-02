# Managed launcher smoke — Strata via the hybrid stack (2026-10-03)

First end-to-end run of the **managed lifecycle**: nothing was running on
:8080; `POST /v1/stacks/strata-125b/apply` started Strata through the harness
and `unload` stopped exactly that process.

| step | result |
|---|---|
| sidecar | `./start_harness.sh` on :8765 (lazy; no models preloaded) |
| `apply` | `{"ok": true, "tiers": [{"role": "face", "port": 8080, "key": "strata-125b-face"}]}` in **33.0 s** (cold load) |
| status | `managed=true`, `pid=36324`, `resident=true`, `base_url=http://127.0.0.1:8080/v1`, served model `qwen3.8-flash-next-q2_0`, ctx 4096, caps decode `[56.8, 57.3]` |
| generated config | `harness_state/strata/strata-strata-125b-face.json`: port 8080, `--max-context 4096`, `--vram-reserve-mib 3072` and `--prefill 512` overridden in place, engine log absolute under `logs/strata_8080.engine.log` |
| chat | `"managed launcher works"` (19 prompt + 4 completion tokens, `reasoning_effort=none`) |
| `unload` | `{"ok": true, "unloaded": ["face"]}`; pid 36324 gone, `/health` refused, MemAvailable back to ~24 GB |

Two defects the smoke caught and the fix commit carries:

1. **venv python was resolved through its symlink** — `.venv/bin/python` →
   `/usr/bin/python3.14`, which lost the venv's `regex`; the launcher now keeps
   the symlink path (`Path.absolute()`, never `.resolve()`).
2. **relative log dir** — the engine log path was written relative and the
   engine resolved it against the Strata checkout; launcher `log_dir`/
   `state_dir` are now absolute.

Provenance: Strata `db4f91a1171d` (unchanged), stack
`stacks/strata-125b.json`, harness `hive_cascade` before the launcher commit
`5eea3d9`; tests `tests/unit/test_stack_strata_launcher.py` (16) plus the
stack/cascade suites.
