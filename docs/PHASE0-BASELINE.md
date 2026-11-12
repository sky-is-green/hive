# Phase 0 baseline and frozen contracts

Recorded 2026-11-12 on the Phase 0 worktree (`phase0/console-gaps`). These are
the numbers the Rust phases are compared against, and the contracts a port
must preserve.

## Baseline

| metric | value |
|---|---|
| Python | 3.12.13 |
| `hive version` | 79–151 ms, 32,296 KB peak RSS |
| `import harness.app` | 327 ms, 65,312 KB peak RSS |
| `import torch` | 862 ms |
| unit + integration suite | 1,644 passed, 42 skipped, 222 s (unit-only: 1,506 / 39, 105 s) |
| cascade per item | 8.9–9.2 s |
| execution checker | 0.23 ms median |
| tracked Python | 131,539 LOC before Phase 0 |
| `harness/reports.py` | 4,377 LOC before; ~588 after removing `/server` |
| tracked repo | ~48 MB before; artifacts + old console removed in Phase 0 |

The runtime work is dominated by model loading and generation, not by the
harness. Rust buys distribution, supervision and a smaller install; it does not
make the models faster.

## Frozen contracts

| surface | the contract | where it is enforced |
|---|---|---|
| CLI | `hive help --json` command/option catalog, exit codes, `--json` output | `tests/unit/test_cli.py`, `tests/unit/test_cli_ui.py` |
| REST | paths, methods and descriptions | `hivebench/API.md`, `tests/unit/test_api_docs.py`, `/openapi.json` |
| Stack file | `stacks/<name>.json` version 1 | `harness/stack/schema.py`, `tests/unit/test_stack_schema.py` |
| State | atomic JSON, locks, schema versions, audit JSONL | `docs/STATE-STORE.md`, `harness/state_store.py`, state-store tests |
| Serve record | `harness_state/serve/<name>.json`, PID + cmdline check | `harness/serve_state.py`, `tests/unit/test_serve_state.py` |
| Sidecar | `/v1/splinter/*`, `/v1/openai/*`, `/v1/mcp` | `splinter/docs/INTEGRATE.md`, `splinter/splinter/server.py` |
| Shell assets | page sections, script assets, JS syntax | `tests/unit/test_webui_assets.py` |
| Deleted console | `/server` is gone; old page tests removed | `tests/unit/test_model_manager.py` regression |

## Phase 0 status

- R0.1 Linux-first Windows boundary: CI is Ubuntu-only unless the manual
  `windows` input is set; `/v1/setup/*` returns 410 on Linux.
- R0.2a–R0.2h console migration: shell pages, security/controller panels,
  release gate moved, `/server` deleted.
- R0.3 artifact inventory: `experiments/cascade/results/MANIFEST.md`; 38
  unreferenced run directories moved to
  `../hive-artifacts-archive-2026-11-12/results` (restorable without a revert).
- R0.4 compatibility shims: `--no-auto-start` and the `strata.server` fallback
  removed; legacy settings migration kept for user data.
- R0.5 contracts: this document plus the existing API map, schema and state
  tests.
- R0.7 baseline: this document.
