# Rust migration and de-bloat plan

**Status:** proposed · **Base commit:** `008d758` (`hive_test`) · **Rewind tag:** `pre-rust-migration` · **Integration branch:** `rust-migration`

This is the plan for moving Hive's runtime/control plane to Rust without
rewriting the research code, and for removing the accumulated surface that a
1:1 port would otherwise carry forward. It is deliberately staged: each phase
ships something useful, has an exit gate, and can be abandoned without losing
the Python baseline.

---

## 0. Decision

**Do not do a big-bang rewrite.**

- The product runtime (CLI, serve/stack supervision, model library, Studio API,
  tool gateway, memory) is a good Rust candidate.
- `experiments/` is Python-native ML and stays Python.
- The Python implementation stays the reference for the papers and for
  `experiments/` even after a Rust binary becomes the default.
- Phase 0 (de-bloat and contract freeze) happens before any Rust port. It is
  worth doing even if the Rust work is cancelled.
- The first Rust deliverable is the **CLI + supervisor**, not the API.

The decision rule:

| goal | recommendation |
|---|---|
| more research, papers, measurements | stay Python; do Phase 0 only |
| a single-binary local product | Phase 0 + Phase 1, then decide on Phase 2 |
| replace Python everywhere | not recommended; experiments make it impossible |

---

## Caveat ledger

Known gaps and deliberate trade-offs are tracked in
[REWRITE-CAVEATS.md](REWRITE-CAVEATS.md), grouped by the phase that generated
them. A caveat does not block the next phase unless it invalidates that phase's
assumptions; the ledger is triaged before `main` is updated.

---

## 1. Baseline

Measured on the local checkout at `008d758`, 2026-11-11:

| metric | value |
|---|---|
| Python LOC | 131,539 |
| `harness/` | 38,228 |
| `splinter/` | 9,235 |
| `experiments/` | 45,815 |
| `tests/` | 30,243 (1,680 collected) |
| frontend JS/CSS/HTML | 8,934 |
| Markdown | 24,664 |
| CLI surface | 72 command nodes, 309 options |
| REST surface | 143 operations across 131 paths |
| unit gate | 1,523 passed, 18 skipped in 113.69 s |
| commits, last 12 months | 675 |
| `hive version` | 79–151 ms, 32,296 KB peak RSS |
| `import harness.app` | 327 ms, 65,312 KB peak RSS |
| `import torch` | 862 ms |
| cascade per item | 8.9–9.2 s |
| execution checker | 0.23 ms median |

ML dependencies are isolated: torch appears in **2** `harness/` files and **2**
`splinter/` files, all lazy; `experiments/` accounts for 29 torch files and 22
transformers files. The control plane is not ML-bound.

Bloat is localized, not pervasive:

- `harness/reports.py:796` `render_server_page()` was a **3,369-line** embedded
  HTML/JS page beside the **8,934-line** `harness/webui/` shell. At the audit
  neither was a superset: `/server` was the release-gated console and owned
  several operational workflows the shell did not (see Appendix C). R0.2h has
  since removed it.
- Windows/WSL2/VHDX: about **800 lines** of setup code in `harness/app.py`
  (`_setup_tier`, `_ensure_vhdx`, `_list_drives`, `_setup_health`, plus the
  `/v1/setup/*` routes), **69** platform branches, **81** VHDX mentions, three
  `.ps1` files (257 lines), and a nightly Windows CI leg. Windows stays
  supported long-term but is deferred behind a Linux-first boundary.
- `experiments/cascade/results` holds **400** tracked files (~23.5 MB), much of it
  one-off run output and run scripts.

---

## 2. Scope

### In scope

- `hive` CLI, `serve`, `status`, `stop`, `logs`, `models`, `stacks`,
  `backends`, `doctor`, `setup`, `tui`.
- Process supervision and serve records.
- Studio REST API and the tool gateway (Phase 2).
- The sandbox runtime and policy, preserved throughout: Docker backend, pinned
  image/digest, optional gVisor, workspace confinement, and audited fallback.
- Memory core and encoder, only if Phase 3 passes its gate.
- A native Landlock/seccomp sandbox backend, WASM plugins and observability
  (Phase 4, optional).

### Out of scope

- `experiments/`, training, model research, paper figures.
- A frontend rewrite. The existing `harness/webui/` stays and can be served by
  either backend.
- Replacing llama.cpp, Strata, DeepInfer or the dsh Node runtime. Rust manages
  them; it does not reimplement them.
- Changing the measured research contracts.

---

## 3. Principles

1. **One contract owner at a time.** A route, command, state file or stack
   schema is owned by exactly one implementation. During migration the other
   side proxies or delegates; it does not get a second copy of the logic.
2. **Strangler, not switch.** The Rust binary grows behind the same CLI/API
   contract. Unimplemented paths delegate to Python until they are ported.
3. **Conformance before cutover.** A component is done when the existing
   black-box suite and the new golden fixtures pass against it, not when it
   compiles.
4. **Python stays the reference.** The papers and `experiments/` keep the
   Python implementation. A Rust memory encoder must match the reference within
   a stated tolerance before it can become the default.
5. **Delete before porting.** Phase 0 removes the stale run output, the
   compatibility shims and the duplicate pieces of the console, and it
   quarantines the Windows-only path off the Linux-first runtime. A 1:1 port
   would preserve the bloat.
6. **One phase per branch.** No mixed cleanup + port + feature commits on the
   integration branch. Every task has a claim and a gate.
7. **Rewind is a first-class requirement.** The tag, the backup branch and the
   side-by-side binary make rollback a checkout, not a rescue.

---

## 4. Target architecture

```
┌──────────────────────────────────────────────────────────────┐
│ hive (Rust, single binary)                                   │
│                                                              │
│  clap CLI ── ratatui TUI                                     │
│      │                                                       │
│  supervisor (tokio) ── serve records / logs / health         │
│      │                                                       │
│  state (atomic JSON + locks)   model library (GGUF, HF)      │
│      │                                                       │
│  daemon (axum, Phase 2) ── /v1 API, SSE, static webui        │
│      │                                                       │
│  engine adapters: llama.cpp | Strata | DeepInfer | endpoint  │
│  sandbox: Landlock/seccomp | Docker (Phase 4)                │
│  plugins: wasmtime verifiers/tools (Phase 4)                 │
└───────────────┬──────────────────────────────┬───────────────┘
                │                              │
   llama-server / strata / deepinfer      Python sidecar
   (external engines, unchanged)          encoder / memory / eval
                                          dsh Node runtime
```

The Rust binary is the control plane. Expensive or research-sensitive work
stays behind a versioned local sidecar protocol until a phase explicitly moves
it. The sidecar is optional for serve-only installs.

---

## 5. Phase 0 — de-bloat and contract freeze

**Goal:** a smaller, clearer Python baseline and a frozen surface to port
against. **Estimated:** 2–4 weeks. **No Rust yet.**

### R0.1 — Set the Windows boundary: Linux first, Windows later

**Decision:** Windows remains a supported target, but it is **deferred**. The
active build, runtime and CI path are Linux-only until the Rust control plane
is nearly finished. This avoids deleting a surface the project still wants,
while taking it off the critical path.

Actions:

- Remove the Windows leg from the required push/PR CI matrix. Keep a
  `workflow_dispatch` job only (no scheduled run) so the arm can be checked on
  demand; it must not gate merges.
- Move `_setup_tier` / `_ensure_vhdx` / `_list_drives` / `_setup_health` and the
  `/v1/setup/*` VHDX/drive routes behind a Windows-only module or feature flag.
  They may stay in the tree; they must not be imported on Linux.
- Keep `taskkill`, `pythonw`, `CREATE_NO_WINDOW`, and the three `.ps1` scripts
  inside a clearly labelled `platform/windows` boundary. Do not add new Windows
  branches to Linux modules.
- Record a Windows review checklist for later: install, paths, process groups
  vs Job Objects, signals, Docker/WSL2, packaging, and CI.
- The Rust supervisor designs a platform trait now (process group on Unix, Job
  Object later); it does not implement Windows in Phase 1.

**Acceptance:** Linux CI is the only required leg; Windows is manual or
non-blocking; Windows-only imports are impossible on Linux and a test asserts
it; the Windows review checklist is in this doc; the unit suite is green.

### R0.2 — Consolidate the consoles, after closing the gap

**Audit result (2026-11-11, Appendix C):** at the audit, `harness/webui/` was
the target shell, but `/server` was still the release-gated console
(`docs/RELEASE-GATE.md` §4) and had workflows the shell did not. R0.2h has
since ported those gaps and removed it.

**Decision (2026-11-11):** port the gaps; keep the sandbox; drop only the
deprecated VHDX setup tab. The work lands on `phase0/console-gaps`.

| task | gap | status |
|---|---|---|
| R0.2a | sandbox status/policy panel in the shell | done (read-only status; CLI remains the write path) |
| R0.2b | process manager, engine log, server status | done (table, log tail, raw status, confirmed kill) |
| R0.2c | live engine A/B compare | done (Engines panel; tok/s, winner, per-profile details) |
| R0.2d | agent preset create/edit/save | done (edit user presets; duplicate system presets; atomic write) |
| R0.2e | prompt inspector (fix + port the backend payload) | done (standalone Inspector page; live state/inspect APIs) |
| R0.2e-b | Splinter chat mode in the shell (the Inspector currently sees API/external conversations) | done (Splinter mode; stream + Inspector handoff) |
| R0.2f | setup wizard / hardware recommendation | done (hardware cards, 8k fit ranking, Strata, one-click profile + load) |
| R0.2g | Cascade/docs links; `hive ui status` and TUI `w` point at `:8170` | done |
| R0.2h | move the release gate; delete `/server` and its tests | done |

Keep `harness/webui/` as the canonical shell. Before deleting
`render_server_page()` and the `/server` route, port or explicitly drop:

- Setup wizard and hardware recommendation (Home).
- Process manager, engine log and server status (Settings / launch panel).
- Live engine A/B compare (Engines); the shell's Benchmark page is a static
  matrix, not the live bench.
- Agent preset create/edit/save; the shell's Presets page only reads/duplicates.
- Prompt inspector. The backend already returns `inspection` and exposes
  `/v1/splinter/inspect/{id}`, but the legacy page calls undefined
  `renderInspection()` / `fetchInspection()`.
- The **Suggestions inbox** is dropped: a planner queue is not a local-first
  harness surface (decision 2026-11-12).
- Command palette and pinned commands, if still wanted.
- A visible Cascade link (`/cascade`) and Senses entry, if still wanted.
- A sandbox status/pin/fallback panel. The sandbox runtime itself is retained:
  `/v1/sandbox/exec`, `harness/sandbox.py`, `hive sandbox`, the pinned digest,
  the optional gVisor runtime and the local-fallback audit all stay. The shell
  currently exposes only the `sandbox_enabled` toggle; add a status panel or
  keep the CLI as the only surface.

The deprecated **Linux / Docker Setup** tab is not the sandbox. It is a note
about the old VHDX/WSL2 large-model path; dropping that tab does not remove
`/v1/sandbox/*` or the sandbox policy.

R0.2a ships the read-only sandbox status panel (Docker probe, image pin, limits,
fallback). Policy writes stay in `hive sandbox policy/pin` until the API exposes
a typed, validated policy contract.

R0.2b ships the process manager (process table, CPU/RAM, confirmed kill), the
engine-log tail and the raw server status. The legacy page's `refreshProcesses`
and `refresh` functions are undefined; the shell calls the live
`/v1/processes`, `/v1/server/log`, `/v1/server/status` and `/v1/server/memory`
routes directly.

R0.2c ships the live A/B compare on the Engines page. It uses
`/v1/engines/ab/bench` with two profiles and a base port, confirms before
loading, and reports tokens/sec, the noise-aware winner and per-profile
details. The shell proxy gives that route a one-hour timeout because two model
loads can exceed the generic 180 s proxy limit.

R0.2d makes user presets editable in the shell. It adds
`POST /v1/agent-presets/write` (existing user presets only, validated id,
atomic replace), keeps system presets read-only, and fixes trust detection in
the read route to compare the actual user root instead of guessing from the
string `.dsh`.

R0.2e adds the Inspector page. It reads `/v1/splinter/state` and
`/v1/splinter/inspect/{conversation}` and shows the summary, selected/dropped
chunks, assembled preview and stage timings. The legacy `renderInspection()`
and `fetchInspection()` functions are undefined, so none of that broken JS was
reused.

R0.2e-b adds Splinter mode to the shell chat. It streams
`/v1/splinter/stream`, persists the conversation id, and writes
`hive-inspect-conv` so the Inspector opens the turn that was just run. The
mock Studio's stream path bypasses its fake backend and talks to the configured
provider directly, so the smoke test used a tiny local OpenAI-compatible
stream server; the end-to-end chat → Inspector handoff passed.

R0.2f adds the Setup page: hardware cards, the local-model ranking at 8k
context, the optional Strata backend, and a confirmed one-click path that
creates the engine profile and starts the model. It uses
`/v1/server/status`, `/v1/models/local`, `/v1/engines/preset`,
`/v1/engines`, `/v1/server/start` and `/v1/provider/config`.

R0.2g adds Cascade and API-docs rail links and proxies `/cascade`, `/docs`,
`/openapi.json` and `/redoc` through the console port. `hive ui status` now
reports both the console (`:8170`) and the API (`:8765`), `hive ui web` passes
`--console-port`, and the TUI `w` key opens the console. This also fixed a
pre-existing `/cascade` 500: the route passed a `defaults=` argument the
renderer did not accept, while the page already loads that baseline from
`/v1/studio/settings`.

R0.2h ports the remaining `/server` safety surfaces: tool policy/grants on
Risk & Approvals, and the hardware profile/controller on Setup. The release
gate now exercises the `:8170` shell. The old page, its training pane, its
settings index and `harness/global_controls.py` are deleted; `harness/reports.py`
drops from 4,377 to ~588 lines. A regression test asserts `/server` is gone.

Post-review cleanup (2026-11-12): the Suggestions inbox page and the
`/api/hive_build/suggestions` dev-server endpoints were removed. A planner
queue is not a local-first harness surface. The Roles placement gate now uses
the built-in `harness/placement.py`: weights + architecture-aware KV at the
requested context, with a weights-only fallback and warning when a role points
at a model with no local GGUF metadata.

**Acceptance:** one canonical shell; the gap list is either ported or explicitly
dropped in this doc; the release gate exercises the surviving shell; tests
updated.

### R0.3 — Prune run artifacts (done)

- Keep the run bundles cited by the papers and `docs/STATUS.md`.
- The inventory is `experiments/cascade/results/MANIFEST.md`.
- 38 unreferenced run directories (~12 MB) moved to
  `../hive-artifacts-archive-2026-11-12/results`; they can be restored from
  there or from git history without a revert.
- One-off scripts whose only consumer was an archived run stay with it.

**Acceptance:** manifest committed, paper-cited runs still present, no broken
links in `docs/`.

### R0.4 — Remove compatibility shims (done)

- `--no-auto-start` removed (the Studio no longer auto-loads a model).
- The pre-rename `strata.server` fallback removed.
- Legacy settings migration is kept: it is user-data migration, not dead code.
- Routes that existed only for the deleted console went with `/server`.

**Acceptance:** `rg` finds no live references; tests updated.

### R0.5 — Freeze the contracts (done)

Recorded in `docs/PHASE0-BASELINE.md`:

- CLI: `hive help --json` catalog, exit codes, stdout/stderr split; covered by
  `test_cli.py` / `test_cli_ui.py`.
- REST: `hivebench/API.md` is the snapshot; `test_api_docs.py` fails on drift,
  and `/openapi.json` is live.
- Stack schema: `stacks/<name>.json` version 1; `test_stack_schema.py`.
- State: serve record, `settings/*.json`, audit JSONL, conversation JSON;
  `docs/STATE-STORE.md` and the state-store tests.
- Sidecar: `/v1/splinter/*`, `/v1/openai/*`, `/v1/mcp`;
  `splinter/docs/INTEGRATE.md`.

A separate generated snapshot was not added: the API map plus the drift test
already is the snapshot, and the CLI catalog is generated from the parser.

### R0.6 — Golden corpus

Capture, from the Python implementation:

- `hive help --json`, `version --json`, `config --json`, `doctor --json`,
  `status --json`, `models list --json`, `stacks list --json` for fixed
  fixtures.
- A selection of REST responses with IDs/timestamps normalized.
- One real serve record from a fake engine.

These become the conformance oracle for Phase 1/2. The first capture is the
`tests/conformance/` work in Phase 1, not a Phase 0 deliverable.

### R0.7 — Baseline metrics (done)

Recorded in `docs/PHASE0-BASELINE.md`: startup, RSS, imports, unit-suite
duration, cascade latency, checker latency, and the tracked-repo reduction.

### Phase 0 gate (G0)

- [x] Windows boundary recorded: Linux-first, Windows deferred, manual CI only.
- [x] Console decision recorded; gap list ported or explicitly dropped.
- [x] Artifacts pruned with a manifest; paper-cited runs present.
- [x] Compatibility shims gone.
- [x] Contract references recorded; golden capture starts in Phase 1.
- [x] Unit and integration suites green.

**G0 sign-off, 2026-11-12:** `pytest tests/unit tests/integration -q` →
**1,644 passed, 42 skipped in 222 s**. Phase 0 is complete.
- [x] Baseline metrics recorded.

---

## 6. Phase 1 — Rust `hive` CLI + supervisor

**Goal:** a drop-in `hive` binary for the operational surface, with the Python
Studio and experiments untouched. **Spike:** 2–4 weeks. **Parity:** 8–12 weeks.
**Depends on:** G0.

### Phase 1 status (2026-11-12)

Phase 1's CLI/supervisor scope is complete (G1, 2026-11-13). The command
surface below is ported and conformance-tested; only the Python Studio/eval
surfaces and a few interactive TUI actions stay on the Python side by design.

The first Rust slice is on `rust-migration`:

- Cargo workspace under `rust/` with `hive-core` (paths, config, hardware,
  backends, JSON), `hive-state` (serve records, PID checks, signals),
  `hive-models` (GGUF header parser + library), `hive-supervisor` (spawn,
  health, unload, PDEATHSIG cleanup), and `hive-cli` (clap surface).
- Ported: `serve` (foreground/`--once`/`--detach`, local llama-server tiers,
  endpoint adoption, managed Strata/DeepInfer launch tiers, generated
  single-tier stacks from a local GGUF or Hugging Face repo, and pre-apply
  `--fetch`), `tui` (four-tab ratatui dashboard), `version`, `config`,
  `doctor`, `help --json`, `setup`,
  `models list/search/files/dir/download/import/remove`,
  `stacks list/show/validate`, `backends`/`backends list`/`backends fetch`,
  `status`, `stop`, `logs`.
- `help --json` is generated from the live clap tree; every ported command is
  byte-identical to Python's catalog (the root catalog lists only ported
  commands until the rest land).
- A fake-engine process test covers `serve --once`, `--detach`, a second-shell
  `status`, `logs`, `kill -9` of the supervisor (engine children die via
  `PR_SET_PDEATHSIG`), and stale-record pruning.
- `serve` now runs the residency preflight before spawning: GGUF weights +
  architecture-aware KV, split by `pin`/`ts`, against live VRAM after the
  desktop's share and a per-tier compute buffer. An over-budget stack is
  refused before any engine starts. A box with no visible GPU proceeds with a
  warning instead of refusing CPU serving.
- Managed launch tiers are ported as a capability path, not a support claim:
  DeepInfer was a capability probe, and Strata needs per-model support. Config
  generation, start, health, detach/status/stop, and cleanup are fake-repo
  tested; the real engine lanes are deferred (see `P1-10` in the caveat
  ledger).
- Live engine test: llama.cpp b12329 CPU + SmolLM2-135M through `serve --once`,
  detached start, `/v1/models`, a chat completion, `status` probe, and stop.
  Run it with:
  `HIVE_LLAMA_SERVER=... HIVE_LIVE_MODEL=... python -m pytest tests/conformance/test_live_llama.py -q`.
- Measurement on this box (release build): `hive version` 0.9 ms, `hive
  status` 1.0 ms, 14.5 MB max RSS, 2.8 MB stripped binary.
- Conformance suite: `tests/conformance/` runs the same assertions against
  `harness.cli` and `rust/target/debug/hive`.

Not ported yet: the Python-only surfaces (`cascade`, `tools`, `sandbox`, `web`,
`browser`, `ui`, `profile`, `controller`, `encoder`). They stay on the Python
side until their phase. A generated stack can be detached only with `--save`
(the child reloads the saved file); the Python CLI can detach without saving.
The Rust TUI is read-only for now: the Python dashboard still owns interactive
stop/log/approval actions.

### Scope

Port, in this order:

| priority | command | why first |
|---|---|---|
| 1 | `hive version`, `config`, `doctor` | small, exercises state/hardware/backends |
| 1 | `hive help --json` | freezes the CLI contract |
| 2 | `hive models list/import/remove/dir` | GGUF parser + library |
| 2 | `hive status`, `stop`, `logs` | serve records, PID checks, signals |
| 2 | `hive serve` foreground, `--once`, `--detach` | the core supervisor |
| 3 | `hive stacks list/show/validate` | schema + residency |
| 3 | `hive backends` | manifest + install detection |
| 3 | `hive tui` | ratatui dashboard, four tabs |
| 4 | `hive setup`, `models search/download`, `doctor --strict` | network/download UX |

`hive cascade`, `tools`, `sandbox`, `web`, `browser`, `ui`, `profile`,
`controller`, `encoder` stay on the Python side in Phase 1 and are invoked via
the Python sidecar/binary.

### Crate layout

```
rust/
  Cargo.toml                 # workspace
  crates/
    hive-cli/                # clap surface, JSON output, exit codes
    hive-core/               # types, config, paths, errors
    hive-state/              # atomic JSON, locks, serve records, audit
    hive-models/             # GGUF memmap parser, HF download, library
    hive-supervisor/         # spawn, signals, health, logs, reaping
    hive-engines/            # llama.cpp, Strata, DeepInfer, endpoint
    hive-tui/                # ratatui dashboard
```

Suggested crates: `clap`, `serde`/`serde_json`, `thiserror`, `tokio`,
`reqwest`, `sysinfo`, `fs2`, `memmap2`, `hf-hub`, `tracing`,
`ratatui`/`crossterm`, `blake3`, `zstd`, `nix` (Unix); `windows-sys` is
deferred with the Windows arm. Test crates: `assert_cmd`, `predicates`,
`insta`, `tempfile`.

Keep the CLI synchronous except the supervisor. Do not use async as a style
choice; use it where the process/health/log loops need it.

### Supervisor design

The supervisor is the reason to do this in Rust. Requirements:

- Spawn each tier in its own process group/session. Windows Job Objects are a
  later platform-trait implementation, not part of Phase 1.
- Record `pid`, process start time, argv hash, port, engine, log path.
- Before signalling, verify the PID **and** the command line; never kill a
  reused PID.
- Health check with a bounded timeout; start failure includes the log tail.
- Graceful stop: SIGTERM, wait N seconds, then SIGKILL the group.
- Reap children; detect and clean stale records on `status`.
- Allocate ports without races; reserve before spawn.
- Tail/rotate logs with a size cap; `logs -f` stops when the serve dies.
- `--detach` starts a child in a new session, writes a record, and returns only
  after health.
- A `kill -9` of the supervisor must not leave engine children behind; a
  restart must prune the stale record.

### State and model library

- Atomic write: temp file → fsync → rename; `flock` around read-modify-write.
- Schema version on every durable file; unknown keys preserved.
- GGUF header parsing with `memmap2`; cache metadata by path + size + mtime.
- HF downloads with resume, checksum and progress on stderr.
- Import: hardlink → symlink → copy, refusing paths outside the library.

### Conformance and tests

- Add `tests/conformance/` that runs the same assertions against
  `HIVE_BIN=python` and `HIVE_BIN=rust`.
- Reuse the existing CLI tests where they are black-box; keep Python unit tests
  for Python internals until the port lands.
- Add process tests: `kill -9` supervisor, SIGTERM, stale PID, PID reuse,
  port collision, log follow, detach.
- Snapshot the JSON output with `insta`; normalize volatile fields.

### Phase 1 gate (G1)

- [x] Ported commands produce byte-compatible JSON for the golden corpus
      (volatile fields normalized): `version`, `config`, `doctor`,
      `help --json`, `models list/search/files/dir/download/import/remove`,
      `stacks list/show/validate`, `status`, `stop`, `logs`,
      `serve --list/--backends`, `backends`/`backends list`/`backends fetch`,
      `setup`; `tui` matches the four-tab surface (no JSON form).
- [x] `hive serve` starts/stops a real `llama-server`; `status` finds it from a
      second shell; `logs` tails both supervisor and tier logs. (Live-tested
      with llama.cpp b12329 CPU + SmolLM2-135M: detached start, `/v1/models`,
      a chat completion, `status` probe, and a clean stop.)
- [x] `kill -9` of the supervisor leaves zero engine processes; restart prunes
      the stale record.
- [x] Startup < 20 ms and RSS < 20 MB for `hive version` and `hive status`
      (0.9/1.0 ms, 14.5 MB measured).
- [x] Binary < 25 MB stripped (2.8 MB, without the ONNX encoder).
- [x] TUI parity on Status/Models/Stacks/Runs (four tabs, same data and
      navigation; interactive stop/log/approvals still live on the Python
      dashboard).
- [x] The Python CLI is available as `hive-py`; `hive` still points at Python
      until the final switch.

**If G1 fails:** keep the Rust binary as an optional tool and do not proceed to
Phase 2. The Python cleanup still stands.

---

## 7. Phase 2 — Rust daemon and Studio API

**Goal:** move the control-plane API to Rust while Python keeps the encoder,
memory and evaluation. **Estimated:** 3–6 months. **Depends on:** G1.

### Approach

- Run the Rust daemon on the public port (8765) and a Python sidecar on a
  private port.
- The sidecar interpreter is explicit: read `HIVE_PYTHON` first; only fall back
  to venv/checkout discovery in development. A set-but-unusable path is a
  startup error, not a silent fallback.
- Implement route groups in Rust one at a time. Unimplemented paths proxy to
  the sidecar, with a header naming the owner.
- Flip the default only when the group's conformance tests pass.
- Keep the existing `harness/webui/` frontend; serve it as static assets.

### Phase 2 status (2026-11-13)

The first daemon slice is on `rust-migration`:

- New `hive-daemon` crate/binary. Native routes: `GET /health`,
  `GET /v1/setup/status` (410 on Linux, matching the Python Windows guard),
  `GET /v1/version`, `GET /v1/config`.
- Every other path is streamed to the Python sidecar at `--sidecar-url` /
  `HIVE_SIDECAR_URL` (default `http://127.0.0.1:8766`). The proxy preserves
  method, path/query, headers, and body, and marks responses with
  `x-hive-owner: rust|python`.
- `HARNESS_TOKEN`/`x-splinter-token` guards `/v1/*`; CORS is permissive for
  this slice.
- **Managed sidecar.** With no `--sidecar-url`/`HIVE_SIDECAR_URL`, the daemon
  resolves Python strictly (a set-but-unusable `HIVE_PYTHON` is a startup
  error), starts `python -m harness --no-console --no-open` on a private
  loopback port (`--sidecar-port`, 0 picks a free port), waits for `/health`
  plus the `service`/`version` handshake, restarts it with 1s→30s backoff, and
  stops its process group on daemon shutdown. `--no-sidecar` runs native routes
  only; an explicit URL never spawns.
- **Static console.** `--webui DIR` (default `harness/webui` when present)
  serves the shell and assets with `no-store` for HTML and a short cache for
  versioned assets. `/api/*` is rewritten to the sidecar root (`/api/v1/x` →
  `/v1/x`; `/api/roles` → `/v1/roles`); `/view/*`, `/cascade`, `/docs`,
  `/openapi.json`, and `/redoc` pass through. `--no-webui` disables it.
- **Stacks and server lifecycle.** `GET/PUT/DELETE /v1/stacks*`, `validate`,
  `apply`, `unload`, and `GET /v1/stacks/status` run through the Rust
  supervisor. `/v1/server/start|stop|unload|unload-loaded|status|log|memory|
  metrics|fit` share the same runtime. One saved stack is applied at a time
  (matching Python); `/v1/server/start` adds one-tier fleets that coexist and
  stop individually. A keyed server stop also removes one tier from an applied
  stack without unloading the others.
- **Model library and hub.** `GET/DELETE /v1/models/local` scan and safely
  delete inside the model library; `POST /v1/models/local/path` switches and
  persists the library, and `import-path` links an external folder (the library
  walk follows links); `GET /v1/models/hub`, `GET
  /v1/models/hub/files/{repo}`, `POST /v1/models/hub/download`, and `GET
  /v1/models/hub/downloads` use the same wire shapes as Python. Downloads run
  as daemon jobs (queued/downloading/done/error), resume `.part` files, check
  the final byte count against `Content-Length`, and keep the `.part` on
  mismatch.
- **OpenAPI.** `GET /openapi.json` serves the native Rust document merged with
  the sidecar's spec, so proxied routes stay documented while native operations
  win on overlap. The conformance test checks native paths, proxied paths, and
  component merging.
- **Live Studio chat.** With a managed sidecar and a native CPU model,
  `/v1/splinter/stream` returns SSE deltas and a `done` event through the
  daemon; the test loads, chats, and stops the model.
- **Provider/engine mirror.** A native server start/stop updates the sidecar's
  `local-<key>` provider and engine entries (and defaults) through
  `/v1/provider/config` and `/v1/engines`, so chat and the engine picker both
  resolve the Rust-started engine.
- **Daemon golden contracts.** `tests/conformance/golden/daemon_version.json`
  and `daemon_config.json` freeze the native `/v1/version` and `/v1/config`
  payloads with paths/platform/python normalized.
- **Service packaging.** `rust/packaging/systemd/hive-daemon.service` runs the
  daemon as a user service with restart-on-failure; `docs/DAEMON.md` covers the
  install layout, `HIVE_PYTHON`, and operations.
- **Python-only verbs proxy.** `hive cascade|profile|controller|encoder|tools|
  activity|sandbox|web|browser` and `hive help <verb>` forward to
  `python -m harness.cli` with stdio inherited. This keeps the root CLI usable
  while those phases remain Python-owned.
- **`hive ui`.** `hive ui status` reports the API and shell surfaces, `hive ui
  terminal` opens the TUI, and `hive ui web` starts `hive-daemon` (foreground or
  `--detach`, pidfile `logs/daemon.pid`). The Rust daemon serves the shell on
  the API port, so `--console-port` is accepted for parity but ignored.
- **Run bundles.** `GET /v1/runs` lists `runs/` newest-first with kind,
  `has_report`, and mtime; `GET /v1/report/{run}` returns `run_report.json`
  after path-traversal checks. The rendered `/view/*` and `/runs` HTML pages
  still proxy to Python's report renderer.
- **Frozen CLI corpus.** `tests/conformance/golden/` snapshots `version`,
  `config`, root `help`, `models list/dir`, and `stacks list/show`. Tests compare
  the Rust binary to these files without starting Python; fixture paths,
  interpreter versions, and the platform string are normalized. Regenerate with
  `tests/conformance/golden/generate.py` only for a reviewed contract change.
- Tests: fifteen daemon integration tests. The fake sidecar covers native routes
  and proxy method/body/header/status/token preservation; the managed test
  starts the real FastAPI sidecar, proxies `/v1/commands`, kills the sidecar and
  observes a restart, then checks shutdown leaves no sidecar process; a final
  test proves an unusable `HIVE_PYTHON` exits non-zero with a clear error. The
  static test covers HTML/JS/CSS types, cache policy, `/api` rewriting, path
  traversal, and native-route precedence. Stack CRUD/validate runs everywhere;
  the live test applies a real CPU stack, checks status/log/unload, then starts
  and stops a one-tier server (skipped without `HIVE_LLAMA_SERVER` and
  `HIVE_LIVE_MODEL`). The hub test runs against a local fake HF API and checks
  search, files, a completed download, local listing/deletion, traversal
  rejection, and a truncated-download error with the `.part` retained.

### Route groups, in order

1. [x] `health`, `setup/status`, `config`, `version` (daemon slice; fake sidecar
   tested).
2. [x] `server/*` lifecycle and `stacks/*` (the supervisor already owns this; one applied stack at a time).
3. [x] `models/*` library, hub, download (local path/import and WSL routes remain proxied).
4. [x] `runs/*`, `report/*`, static pages (runs index/report JSON native; rendered `/view` and `/runs` HTML still proxy).
5. [x] `cascade/*` (thin proxy by design at first).
6. [x] `splinter/*`, `openai/*`, `mcp` (sidecar until Phase 3).
7. `tools/*`, `sandbox/*`, `web/*`, `browser/*` (gateway).
8. `agent/*`, `activity/*`, `controller/*`, `training/*` (sidecar/proxy).

### Daemon design

- `axum` + `tower` middleware: auth token, CORS, request IDs, body limits.
- Typed request/response models; generate OpenAPI with `utoipa` and a TypeScript
  client with `ts-rs` (or `typeshare`).
- SSE/streaming with cancellation: when the client disconnects, cancel the
  upstream request and close the provider connection.
- Static assets with cache-busting; no Python templating for new pages.
- Sidecar protocol: `GET /health`, version handshake, restart with backoff,
  clear error when the sidecar is missing.
- Audit/telemetry: structured `tracing` spans; JSON logs; no secrets.

### Security and sandbox

- Keep the Python gateway until the Rust policy engine passes the same
  conformance tests.
- Then implement the policy core in Rust: tiers, grants, approvals, hashes,
  audit.
- Add a native Linux sandbox (Landlock + seccomp + user namespaces) behind the
  existing `/v1/sandbox/exec` contract; Docker remains a backend, not the only
  one.
- Keep the existing sandbox throughout: pinned image/digest, optional gVisor
  runtime, workspace confinement, timeout/limits, and the audited local
  fallback. The deprecated VHDX/Docker Setup tab is unrelated and may go.
- Do not remove the local-fallback audit path.

### Phase 2 gate (G2)

- [ ] All route groups in Rust, with the sidecar used only for memory/eval.
  Groups 1–4 are native; 5–8 proxy by design because they are Python runtimes
  the staged plan keeps until Phase 3 (P2-37). The proxy boundary is smoke-
  tested against the real sidecar (P2-38).
- [ ] OpenAPI/TS generation is part of CI; the frontend compiles against it.
  OpenAPI is native and merged (P2-09); TypeScript client generation is
  deferred because the console is vanilla JS (P2-35).
- [ ] Studio end-to-end: load a model, chat, cascade run, stop, unload.
  Load/chat/stop/unload and provider/engine mirroring are tested live; cascade
  runs still depend on the Python sidecar's runtimes (P2-36).
- [x] Streaming cancellation closes the upstream connection.
- [ ] Security/gateway conformance suite green.
- [x] Sidecar absent: serve-only paths still work with a clear message
  (`--no-sidecar`, 503 on proxied routes, native stacks/server still work).

Remaining Phase 2 work is tracked in `docs/REWRITE-CAVEATS.md` (P2-19…P2-33):
console-only `/api` endpoints, multi-instance server start, per-tier unload,
local-library path/import routes, rendered report pages, and OpenAPI/TS.

---

## 8. Phase 3 — memory and encoder (conditional)

**Goal:** remove Python/torch from the runtime install and make long-context
memory cheap. **Estimated:** 2–4 months. **Depends on:** G2 and a measured
need.

Do this only if one of these is true:

- install size/RSS is the top product complaint;
- conversations regularly exceed the current chunk count and assembly time is
  visible;
- the encoder is the only remaining reason to ship torch.

### Work

- Export `paraphrase-MiniLM-L3-v2` to ONNX; run it with `ort` + `tokenizers`.
- Match sentence-transformers pooling/normalization exactly; build a 1k-text
  parity corpus with a stated cosine tolerance.
- Port `ContextStore`, decay, dedup, drift, comb, assembly and scoring.
- Keep the Python implementation as the reference and the paper replay path.
- Replay the P1–P11 fixtures and the 28-task cascade; compare reports.
- Switch the default only after parity and an A/B with no quality regression.

### Phase 3 gate (G3)

- [ ] Install no longer needs torch/transformers/sentence-transformers for the
      default runtime.
- [ ] Encoder parity within tolerance on the corpus.
- [ ] P1–P11 and the 28-task cascade match the reference.
- [ ] Assembly latency not worse; RSS/install materially lower.
- [ ] Python reference still reproducible from the same commit.

---

## 9. Phase 4 — optional differentiators

Only after G3, and only if they serve the product:

- **Zero-downtime engine swaps:** blue/green ports, health-gated cutover.
- **Native sandbox by default:** Landlock/seccomp, no Docker requirement.
- **WASM verifier/tool plugins:** `wasmtime`, fuel and memory limits.
- **Per-tier observability:** tok/s, TTFT, VRAM, queue depth, Prometheus.
- **Auto-update and signed releases:** `cargo-dist`, checksums, SBOM.
- **Multi-stack supervision:** cgroup budgets, restart policies.

---

## 10. Work breakdown

| phase | deliverable | estimate | gate |
|---|---|---|---|
| 0 | de-bloat + frozen contracts | 2–4 weeks | G0 |
| 1a | Rust CLI + state + models + status/stop/logs | 3–5 weeks | — |
| 1b | Rust supervisor + serve/detach + TUI | 4–7 weeks | G1 |
| 2a | Rust daemon + lifecycle/stacks/models routes | 6–10 weeks | — |
| 2b | remaining API + gateway + sidecar | 8–14 weeks | G2 |
| 3 | ONNX encoder + memory core | 8–16 weeks | G3 |
| 4 | sandbox/WASM/observability | as needed | — |

Estimates assume one experienced engineer with the existing test suite as the
oracle. Agent-assisted work can compress the coding but not the review, live
smoke tests or research-parity checks.

---

## 11. Testing and gates

### Conformance suite

- `tests/conformance/` is language-neutral: it invokes a binary or an HTTP
  endpoint and compares normalized JSON.
- Run it in CI against Python (oracle) and Rust (candidate) on every PR.
- Golden fixtures under `tests/conformance/golden/`; changes require an explicit
  diff review.

### Rust tests

- Unit tests in each crate.
- `assert_cmd` tests for the CLI.
- Process tests for the supervisor (signals, kill -9, PID reuse, ports).
- `insta` snapshots for JSON and TUI frames.
- `cargo clippy -- -D warnings`, `cargo fmt --check`, `cargo deny check`.

### Live smoke

The release gate (`docs/RELEASE-GATE.md`) is the manual acceptance test. Add a
Rust column and run it on the release candidate. Do not weaken it.

### Performance budget

| metric | target |
|---|---|
| `hive version` startup | < 20 ms |
| `hive version` RSS | < 20 MB |
| stripped binary | < 25 MB |
| orphan engines after supervisor `kill -9` | 0 |
| serve record recovery | automatic |
| API p95 on localhost | no worse than Python |

---

## 12. CI, packaging and release

- Add a Rust job: fmt, clippy, test, `cargo deny`, cross-compile Linux x86_64.
  Windows is manual only until the Windows review in §16.
- Keep the Python job for `experiments/`, `splinter/` and the reference tests.
- Build artifacts with `cargo-dist` or an equivalent: tarballs, checksums,
  SBOM, release notes.
- Ship `hive` (Rust) and `hive-py` (Python) side by side for one release.
- The Python wheel remains the research install; the Rust binary is the
  runtime install.
- Version both from the same `VERSION` source; the sidecar protocol has its own
  version.
- **No `main` merge or release tag until the entire rewrite passes the final
  gate** (see §13). `rust-migration` is the integration branch until then.

---

## 13. Branch, merge and rollback policy

- **`main` is frozen at `pre-rust-migration` (`008d758`) for the entire
  rewrite.** No Phase branch, cleanup, or intermediate Rust phase merges to
  `main`. Only the final rewrite gate may update it.
- `pre-rust-migration` tag and `backup/hive-test-008d758` branch mark the exact
  pre-migration base.
- `rust-migration` is the long-lived integration branch; each task lands on a
  short-lived branch and merges into it only at its gate. That is not a
  `main` merge.
- The Python implementation is not deleted until one release after the Rust
  default and a green release gate.
- The Rust binary is additive until G2: users can run `hive-py` or the Python
  module directly.
- State files are forward-compatible: Rust reads the Python schemas; any new
  field is optional and preserved.
- If a phase fails, revert the integration merge and keep the tag. No data
  migration is destructive.

---

## 14. Risks

| risk | impact | mitigation |
|---|---|---|
| research drift while the runtime moves | papers cannot be reproduced | Python reference stays; replay P1–P11 before every cutover |
| test gate rewrite | lost coverage | black-box conformance + keep Python unit tests until ported |
| two-language maintenance | slower iteration | one owner per contract; sidecar boundary; phase gates |
| ecosystem gaps (Playwright, HF, ONNX) | feature loss | keep Python for optional extras; prove parity before removing |
| agent/claims workflow | gate friction | task IDs and claims per phase; one hotspot at a time |
| Windows deferred | drift in the Windows arm | quarantine + manual/non-blocking CI + review checklist; no required Windows gate until late |
| packaging/signing | release delay | cargo-dist spike in Phase 1, not at the end |
| security regression | tool/sandbox exposure | keep Python gateway until Rust passes the same policy tests |

---

## 15. Decision gates

- **G0 (after Phase 0):** bloat removed, contracts frozen, baseline captured.
  If this is not worth doing, stop.
- **G1 (after Phase 1 spike):** 10 commands at parity, supervisor survives
  `kill -9`, targets met. If not, keep Rust optional.
- **G2 (after Phase 2):** API parity, Studio end-to-end, sidecar stable. If not,
  keep the Rust CLI only.
- **G3 (after Phase 3):** memory parity, install/RSS win, no quality regression.
  If not, keep the Python memory.

Each gate is a written go/no-go in this document with the measured evidence.

---

## 16. Open questions

1. **Windows review:** scheduled before the first Windows release; confirm the
   checklist (install, paths, Job Objects, signals, Docker/WSL2, packaging, CI)
   and whether the deferred arm still compiles.
2. **Console gaps:** confirm which R0.2 items are ported to `webui/` and which
   are explicitly dropped; move the release gate to the survivor.
3. **Product target:** a single-binary local appliance, or a research harness
   with a supported CLI? The answer decides whether Phase 2/3 are funded.
4. **Encoder parity:** what cosine tolerance and what corpus are acceptable
   before the Rust encoder can be the default?
5. **Ownership:** who owns the Rust gate in the claims workflow, and which
   agent/provider is allowed to merge Phase 1?

---

## Appendix A — baseline commands

```bash
# gate
.venv/bin/python -m pytest tests/unit -q
.venv/bin/python -m pytest --collect-only -q tests/integration

# scale
git ls-files '*.py' | xargs wc -l
git ls-tree -r -l HEAD | awk '{s+=$4} END {print s/1048576 " MB"}'

# CLI surface
.venv/bin/hive help --json

# startup / RSS
.venv/bin/python - <<'PY'
import subprocess, resource
subprocess.run([".venv/bin/hive", "version"], check=True)
print(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss, "KB")
PY

# live smoke
hive doctor --json
hive status --json
hive serve <stack> --once
```

## Appendix B — first files to read

- `harness/cli.py` (3,155) — command surface and JSON output.
- `harness/serve.py` (703) — foreground/detached serve.
- `harness/stack/manager.py` (504) — tier lifecycle.
- `harness/models.py` (1,866) — GGUF, KV estimate, server manager.
- `harness/serve_state.py` (358) — records, PID checks, signals.
- `harness/state_store.py` (416) — atomic writes, locks, audit.
- `harness/stack/schema.py` (688) — the stack contract.
- `harness/tui.py` (905) — the dashboard to port.
- `harness/app.py` (5,747) and `harness/reports.py` (4,377) — API and the
  console to consolidate.

## Appendix C — console audit, 2026-11-11

**Question:** what does `/server` do, and can it be deleted?

**Method:** read `render_server_page()` (`harness/reports.py:796-4164`),
`harness/webui/index.html` and its panels, the `/v1` references in both,
`harness/__main__.py`, `harness/tui.py`, `harness/cli.py`,
`docs/RELEASE-GATE.md`, and `hivebench/README.md`.

**Finding:** neither console is a superset. `/server` is still the
release-gated console and owns several operational workflows; the shell owns
the newer analysis pages.

| `/server` surface | what it does | `harness/webui/` equivalent | action |
|---|---|---|---|
| Home: setup wizard | detect hardware, recommend backend and model | partial: Engines fit panel, Models | port a compact wizard or link to Engines/Lifecycle |
| Home: deep research queue | queue for the background worker | none (the Suggestions inbox was removed) | dropped as out of scope |
| Home: backend selector | Strata / llama.cpp / API | Settings, Providers, Engines | confirm coverage |
| Setup: Docker/VHDX | deprecated note page for the old VHDX/WSL2 large-model setup; **not the sandbox** | none | drop the tab; keep `/v1/sandbox/*`, `harness/sandbox.py`, `hive sandbox` |
| Agent: presets | create, edit, save, duplicate | Presets only reads/duplicates | port create/edit/save |
| Engines: profiles | model, endpoint, ctx, sampling, load | Engines + fit panel | covered |
| Engines: live A/B | compare two profiles live | Benchmark is a static matrix | port the live A/B |
| Library | local models, import, remove, dir, move | Models | covered except Windows move (deferred) |
| Providers | config and keys | Providers | covered |
| Hub | HF search, files, downloads | Models: HF finder + downloads | covered |
| Inspector | last turn, selected/dropped chunks, timings | none | fix and port; backend exists |
| Settings: index | searchable 35-setting index | Settings global form | port the index if wanted |
| Settings: process manager, engine log, server status | CPU/RAM, kill, log, status | Overview/Lifecycle partial | port |
| Settings: Global controls | encoder, memory, quality, MCP, runtime, controller | Settings JSON blocks | verify controller/profile coverage |
| Training | training runs | Training | covered |
| Chat pane | chat/agent, sessions, commands, attachments | Chat | verify agent mode and commands |
| Launch panel | load/unload/status | Lifecycle + Engines | partial |
| Command palette/pinned commands | Ctrl+K and pinned row | none | port if wanted |
| Cascade / Senses | links to `/cascade` and senses | none | add nav links if wanted |

Pages that exist only in the shell: Overview, Roles, Benchmark, Routing, Judge,
Risk & Approvals, Activity, Lifecycle, Runs, Presets, Models, Settings.

**Entry points are inconsistent:**

| surface | opens (after R0.2g) |
|---|---|
| `hive ui web` | `python -m harness`, console on `:8170` (shell) |
| `hive ui status` | probes and reports both the console `:8170` and API `:8765` |
| TUI `w` | `studio_api.console_url()` default `:8170` |
| `:8765/` | redirects to `/runs` |
| `docs/RELEASE-GATE.md` §4 | tests the `:8170` shell |
| `hivebench/README.md` | page table lists the shell pages |

**Latent bug (fixed):** the old `/server` page called `renderInspection()` and
`fetchInspection()`, which were not defined anywhere in the tree, so its
Inspector tab could not populate. R0.2e replaced it with a working Inspector
page backed by `Splinter.inspect_turn()`, `/v1/splinter/stream` and
`GET /v1/splinter/inspect/{conversation_id}`.

**Recommendation (executed):** keep `harness/webui/` as the canonical shell;
port the gap list; add nav links to `/cascade` and `/docs`; point
`hive ui status` and TUI `w` at `:8170`; move the release gate; then delete
`render_server_page()` and its tests. R0.2h did exactly that.

## Appendix D — Windows review checklist (deferred)

Run this before the first Windows release; do not gate Linux work on it.

- [ ] Fresh install from a clean checkout: venv, pip, extras, `hive setup`.
- [ ] Paths: `HIVE_HOME`, `%LOCALAPPDATA%`, long paths, spaces, case-insensitive
      filesystems.
- [ ] Process supervision: Job Objects, process groups, `taskkill /T`, PID
      reuse, orphan cleanup.
- [ ] Signals: SIGTERM equivalent, graceful stop, console-less `pythonw`.
- [ ] WSL2/VHDX/Docker: decide whether that path is still product or legacy.
- [ ] Engine backends: Vulkan/ROCm/CPU, `llama-server.exe`, GPU probing (WMI).
- [ ] Model library: hardlink/symlink/copy semantics, drive moves, permissions.
- [ ] Sandbox: Docker Desktop, named pipes, path translation.
- [ ] Packaging: wheel, standalone binary, code signing, SmartScreen.
- [ ] CI: manual/nightly leg, runtime, disk space, cache.
- [ ] Tests: platform-specific tests, line endings, file locks.
- [ ] Docs: Windows install path and caveats.

Update R0.1 from “deferred” when this review passes.
