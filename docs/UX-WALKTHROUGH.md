# UX walkthrough — 2026-11-23

Pre-approval walkthrough of the Rust runtime and Studio on a clean prefix and
clean `HIVE_HOME`. The session fixed every hard failure it found; the full gate
was re-run afterwards.

## Environment

| | |
|---|---|
| install prefix | `/tmp/opencode/hive-ux/prefix` |
| `HIVE_HOME` | `/tmp/opencode/hive-ux/home` |
| Python sidecar | `/tmp/opencode/hive-ux/venv` (`splinter[serve]` + `.[studio]`) |
| engine | fetched CPU llama.cpp (`hive backends fetch cpu`) |
| model | `SmolLM2-135M-Instruct-Q3_K_M.gguf` (89 MiB) |
| live engine | `$PWD/tools/backends/cpu/llama-server` |
| evidence | `/tmp/opencode/hive-gate/20261123T165924Z/` |

Screenshots were not available because the Review pane was not visible to the
browser tool; evidence is DOM/JSON plus measured layout instead.

## Stops

| # | Stop | Expected | Actual | Verdict |
|---|---|---|---|---|
| 1 | Clean install | `install-rust.sh`, `hive setup`, `hive-py` coexist | Installer placed the manifest and console; `setup` fetched a Vulkan engine and native encoder; `hive-py` ran | pass after P2-33/P2-40 |
| 2 | First run | `hive`, `doctor`, `config`, `status` explain the box | Static dashboard suggested a real stack/model; doctor is green (sidecar + encoder checks); TUI renders Status/Models/Stacks/Runs and `q` exits | pass |
| 3 | Models | search, files, download, import, remove | All five worked; import hardlinked and remove stayed inside the library | pass |
| 4 | Stacks and serve | validate, serve, status, logs, stop | `ux-cpu` validated; `--once` and detached serve worked; status showed per-tier health; logs/stop cleaned up | pass after P1-53 |
| 5 | Daemon | health, ready, metrics, swap, sidecar | `/health` 200, `/ready` 503→200 during sidecar start, `/metrics` populated; swap moved 1234→1235 and closed the old port; clean home auto-started the native encoder | pass after P2-37 |
| 6 | Studio | Load, chat, memory inspector, lifecycle, sandbox, narrow viewport | Load hit the Rust runtime; direct chat and Splinter chat returned real model replies; Inspector showed the Splinter turn; Lifecycle showed loaded engines; Models listed the library; Sandbox showed Docker/pin policy; narrow layout had no horizontal overflow | pass after P2-34/P2-35/P2-39 |
| 7 | Failure states | clear, non-crashing errors | Missing model, missing sidecar, bad token, sandbox denial/timeout, killed engine, read-only model dir, and cancelled stream all returned clear errors and left the daemon usable | pass after P2-36/P2-38 |
| 8 | Upgrade and rollback | reinstall, uninstall, `hive-py` | Reinstall over the prefix worked; `--uninstall` removed binaries/console, preserved `HIVE_HOME`, and `hive-py` still ran; reinstall restored `hive` | pass |

## Fixed during the walkthrough

- P2-33 — manifest not installed; `$HIVE_HOME/harness/data` shadowed the Python
  package. The manifest now lives under `$HIVE_HOME/tools/backends` and
  `$PREFIX/share/hive`; the installer cleans the old shadow.
- P2-34 — Studio `/api/v1/server|stacks` calls went to Python; Load failed and
  Lifecycle was empty. Native routes now answer the `/api/v1` prefix too.
- P2-35 — `/v1/chat/completions` returned the mock body; it now forwards to the
  curated OpenAI passthrough. llama tiers default to a friendly `--alias`.
- P2-36 — `HARNESS_TOKEN` accepted only `x-splinter-token`; Bearer is now
  accepted by the daemon and sidecar.
- P2-37 — no native encoder model + no torch gave a 500 and could take the
  sidecar down. `hive setup` fetches the encoder, the daemon auto-detects it,
  and the sidecar returns a 503 with the install command when absent.
- P2-38 — the native sandbox defaulted to unwritable
  `/srv/hive-ai/workspaces`; it falls back to
  `$HIVE_HOME/harness_state/workspaces`.
- P2-39 — narrow viewport overflow and dead hash links; responsive CSS and a
  `hashchange` listener were added.
- P2-40 — the Python install docs omitted `splinter[serve]`; docs, installer
  output, and `hive doctor` now name it.
- P2-41 — the Overview hardcoded DS4.1/LFM3 role endpoints and showed them as
  down; it now shows the daemon plus only the engines it actually owns.
- P2-42 — `hive ui status` assumed `:8765`; the daemon writes a runtime
  descriptor and status reports the live URL/port, or off when absent.
- P1-53 — CPU-only display warnings and bare library names in `serve`.
- P1-54 — the first-run hint now names a real stack/model.

## Accepted observations (not merge blockers)

- The wide Studio layout was not visually re-checked because the browser pane
  stayed at 388 CSS px; the narrow layout is verified and the wide CSS is
  unchanged. A layout pass is planned separately from this functionality pass.

## Gate

`scripts/full-gate.sh` passed on commit `51f78e3`: **24/24** steps on the updated tree, including
Rust fmt/build/clippy/tests, Python unit/integration, non-live conformance,
every live daemon/llama test, install/uninstall layout, and release
manifest/signing. Evidence:
`/tmp/opencode/hive-gate/20261123T165924Z/`.
