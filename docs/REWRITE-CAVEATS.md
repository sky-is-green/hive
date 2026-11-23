# Rewrite caveat ledger

This is the running list of known gaps, parity differences, and deliberate
trade-offs in the Rust rewrite. It exists so phases can keep moving without
pretending every gap is fixed immediately.

**Rule:** a caveat does not block the next phase unless it invalidates that
phase's assumptions. It *does* get triaged before `main` is updated.

## How to use this list

- Add a row as soon as a caveat is discovered. Do not delete rows.
- Close a row by setting `status` to `fixed <commit>` or `accepted <reason>`.
- Every caveat has a phase. Cross-phase items live in their own section.
- Before merge, every `blocker` must be fixed or explicitly accepted; every
  `should` must be fixed or accepted with a reason; `minor` may ship.
- Keep the closure condition concrete enough to turn into a test.

Severity:

| tag | meaning |
|---|---|
| `blocker` | must be resolved or explicitly accepted before merging to `main` |
| `should` | real gap; fix before merge if feasible |
| `minor` | cosmetic or edge case; may ship |
| `intentional` | deliberate design decision; revisit only if assumptions change |
| `deferred` | out of the current platform/phase scope |

## Caveat pass 2026-11-15

- Full pre-approval gate: **24/24 steps passed** (`scripts/full-gate.sh`),
  including Rust fmt/clippy/tests, Python unit/integration, non-live
  conformance, every live daemon/llama test, install/uninstall, and
  release signing. Evidence: `/tmp/opencode/hive-gate/20261115T174410Z/`.
- Every row is now triaged: **0 rows remain `open`**. Rows are `fixed`,
  `accepted` with a reason, or `deferred` with a trigger.
- Fixed in this pass: CPU-only residency charged VRAM (P1-52), logs rotate at
  16 MB (P1-13), `logs -f` handles SIGINT (P1-15), `--model --detach` works
  without `--save` (P1-16), `HARNESS_MEMORY_GUARD=0` is honored (P1-17),
  NVIDIA/AMD hardware probing and CUDA visibility (P1-21/P1-23/P1-26),
  toolchain pin (P1-47), bounded status probes (P1-50), and the docs link audit
  (P0-12/X-07).
- `P1-01` is fixed on `rust-migration`; the `main` merge remains the owner's
  approval gate. `P4-05` keeps signed artifacts as fixed tooling while
  production key/CI provisioning is a deployment follow-up.
- Language fit is recorded in `docs/FINAL-REVIEW.md`: Rust remains where
  startup/RSS/process safety or the torch-free runtime wins; Python remains for
  research/Studio breadth; C++ is only a benchmark-driven hotspot candidate.

## UX walkthrough 2026-11-23

- Full pre-approval gate on commit `dbd8be5`: **24/24 steps passed**
  (`scripts/full-gate.sh`), including every live daemon/llama test, install
  layout, and release signing. Evidence:
  `/tmp/opencode/hive-gate/20261123T183103Z/`.
- Walkthrough stopped at the first hard failures and fixed them in place:
  clean-install manifest/package shadowing (P2-33), Studio `/api/v1` lifecycle
  routing (P2-34), canonical chat mock + model IDs (P2-35), Bearer auth
  (P2-36), native encoder onboarding (P2-37), sandbox workspace fallback
  (P2-38), responsive/deep-link console (P2-39), Python sidecar install
  docs (P2-40), live Overview cards (P2-41), and descriptor-based `ui status`
  (P2-42), live endpoint/role pickers (P2-43), the managed-sidecar
  descriptor collision (P2-44), multi-daemon status (P2-45), and per-stack
  doctor ports (P2-46). TUI actions are P1-40. CPU-only/display warnings and the first-run hint are P1-53/P1-54.
- Verified: clean install and `hive setup`, first-run `hive`/`doctor`/`config`/
  `status`, model search/download/import/remove, stack validate/serve/logs/stop,
  daemon health/ready/metrics/swap, Studio Load/chat/Splinter/Inspector/
  Lifecycle/Models/Sandbox at 388px, explicit failure states, and
  reinstall/uninstall/`hive-py` rollback.
- Owner approved and the rewrite merged to `main` at `01d90ed`; tag `rust-g4`.

## Merge-blocker shortlist

- `P1-01` Fixed on `main` at `01d90ed` (tag `rust-g4`): Rust is the default `hive`.
- `P1-05` Fixed on `main`: the Rust runtime has an install path; production release key/CI remains P4-05.

---

## Phase 0 — console de-bloat and contract freeze

| ID | Caveat | Severity | Status | Close when |
|---|---|---|---|---|
| P0-01 | Windows remains quarantined: no required Windows CI, no validated Windows CLI/TUI path. | deferred | deferred — Windows review | Windows review resumes; the deferred Windows checklist is executed. |
| P0-02 | `/v1/setup/*` returns 410 on Linux; the Windows setup flow is intentionally unavailable. | intentional | accepted — Linux-only setup | Linux setup remains the only supported path, or Windows work resumes. |
| P0-03 | Suggestions inbox and `/api/hive_build/suggestions` were removed; external consumers break. | intentional | accepted — removed by de-bloat; no supported consumers | No external consumer depends on the old endpoints. |
| P0-04 | The legacy `/server` page/routes were removed; old bookmarks and scripts break. | intentional | accepted — legacy page removed; links audited | No live links/scripts target `/server`. |
| P0-05 | `--no-auto-start` was removed; scripts using it fail. | intentional | accepted — removed flag; replacement documented | No supported script uses the old flag. |
| P0-06 | The `strata.server` fallback was removed; deployments relying on it break. | intentional | accepted — fallback removed; supported launch blocks only | Deployment configs use the supported launch/endpoint blocks. |
| P0-07 | Run artifacts were archived locally (`work/hive-artifacts-archive-2026-11-12`), not uploaded to HF. | deferred | deferred — local archive retained | Archive is uploaded or no longer needed. |
| P0-08 | Placement gate falls back to weights-only with a warning for hosted/unknown models. | intentional | accepted — placement preflight warns on hosted models | A hosted-model KV source exists, or the warning is accepted. |
| P0-09 | Placement gate is a preflight, not a load probe; context/KV are operator-selected defaults. | intentional | accepted — preflight contract | Accepted as a preflight contract. |
| P0-10 | Placement gate refuses models with unknown weight sizes. | intentional | accepted — safety rule | Accepted as a safety rule. |
| P0-11 | The release gate is still a manual browser sweep. | minor | accepted — manual UX gate; scripted checks in scripts/full-gate.sh | A repeatable automated console sweep is added. |
| P0-12 | Historical docs are marked instead of deleted; external deep links may drift. | minor | fixed — link audit passed 2026-11-15 | Link audit passes before merge. |

## Phase 1 — Rust CLI and supervisor

### Switch, packaging, and surface

| ID | Caveat | Severity | Status | Close when |
|---|---|---|---|---|
| P1-01 | `scripts/install-rust.sh` installs the Rust runtime as `hive`; `pyproject.toml` only provides `hive-py`. `main` stays frozen until the final gate merges the switch. | blocker | fixed on rust-migration — final merge pending | The final rewrite gate switches the `hive` entry point and keeps `hive-py`. |
| P1-02 | `pyproject.toml` provides `hive-py`; `docs/INSTALL.md` covers reinstalling existing venvs so the stale Python `hive` script is removed. `test_install_layout.py` verifies the split. | minor | fixed — install docs + packaging test | Install/upgrade docs and a packaging test cover the new script. |
| P1-03 | Python-only verbs (`cascade`, `tools`, `sandbox`, `web`, `browser`, `ui`, `profile`, `controller`, `encoder`) are not proxied by Rust. | blocker | fixed — CLI proxy for Python-only verbs | Phase 2 proxy covers them, or they are explicitly accepted as Python-only. |
| P1-04 | Root `hive help --json` lists only ported commands. | intentional | accepted — Python-only verbs proxied; native topics listed | P1-03 lands, or the subset is accepted. |
| P1-05 | `scripts/install-rust.sh` builds and installs the Rust binaries, webui, and user service unit; `scripts/release_manifest.py` emits checksums/SBOM and `scripts/sign_artifact.py` signs them. `test_install_layout.py` covers the install/uninstall path. | blocker | fixed — install/release path documented and tested | `cargo build --release` is wrapped in a documented install/release path. |
| P1-06 | Rust Python discovery is heuristic (`HIVE_PYTHON`, `VIRTUAL_ENV`, checkout `.venv`, PATH). Decision: reuse `HIVE_PYTHON` as the explicit sidecar contract; discovery is only a development fallback. | should | fixed — strict sidecar resolve | Phase 2 daemon reads `HIVE_PYTHON` first and fails clearly when it names a missing/unusable interpreter. |
| P1-07 | `hive version --json` takes ~21 ms because it spawns Python; plain `version`/`status` meet the 20 ms budget. | minor | accepted — 21 ms JSON-only cost; plain version meets budget | Cache the Python version or accept the JSON-only cost. |
| P1-08 | Human output and clap help text are not byte-identical to argparse; JSON is the compatibility contract. | minor | accepted — JSON is the compatibility contract | Accepted, or human output is snapshotted. |
| P1-09 | No shell completions or man pages. | minor | accepted — completions/man pages deferred | Completions/man pages are generated. |

### Supervisor

| ID | Caveat | Severity | Status | Close when |
|---|---|---|---|---|
| P1-10 | Managed Strata/DeepInfer launch tiers are fake-repo tested only. DeepInfer was a capability probe, not runtime support; Strata needs per-model support. | deferred | accepted — revisit when the real lanes return | A supported Strata model and the DeepInfer capability lane are run end-to-end; DeepInfer is not presented as supported runtime. |
| P1-11 | A `kill -9` of the Rust supervisor may leave launch-tier grandchildren behind if the Python launcher ignores signals. | deferred | accepted — tied to P1-10 | Revisit with P1-10; a real launch-tier kill -9 test leaves no engine processes. |
| P1-12 | No `drain_slots` wait before unload; a model killed mid-generation loses work. | should | fixed — supervisor drain | Unload polls `/slots` with the Python timeout before SIGTERM. |
| P1-13 | No log rotation or size cap for supervisor/engine logs. | minor | fixed — 16 MB log rotation | A size cap or rotation policy is implemented/accepted. |
| P1-14 | Port allocation is check-then-spawn; a race can lose the port. | minor | accepted — check-then-spawn race is negligible | A reservation/lock covers the spawn window, or the race is accepted. |
| P1-15 | `hive logs -f` exits 130 on Ctrl-C; Python exits 0. | minor | fixed — SIGINT returns 0 | The follow loop catches SIGINT and returns 0. |
| P1-16 | `serve --model … --detach` requires `--save`; Python can detach without saving. | should | fixed — temporary stack for --model --detach | The child argv carries the model flags or a temporary stack. |
| P1-17 | `HARNESS_MEMORY_GUARD=0` bypass is not ported; Rust residency always runs when devices exist. | should | fixed — HARNESS_MEMORY_GUARD=0 honored | The env override is honored or explicitly rejected. |
| P1-18 | Local engine health checks hardcode `127.0.0.1`; a non-loopback `--host` can fail. | minor | accepted — loopback engines; wildcard host maps to loopback | Health probes use the configured host/address. |
| P1-19 | Serve records and error strings are not in the golden corpus; only selected fields are asserted. | minor | accepted — selected fields asserted | Serve records are snapshotted with volatile fields normalized. |
| P1-20 | `stop --force` process-group behavior for launch tiers is not tested against a real engine tree. | deferred | accepted — tied to P1-10 | Revisit with P1-10; a real launch-tier force-stop leaves no children. |

### Hardware and residency

| ID | Caveat | Severity | Status | Close when |
|---|---|---|---|---|
| P1-21 | Hardware probing is AMD sysfs only; NVIDIA/Intel devices are invisible to Rust. | should | fixed — sysfs AMD + nvidia-smi | NVIDIA/Intel probes or an explicit Linux-AMD scope decision. |
| P1-22 | With no visible GPU, Rust proceeds with a warning; Python refuses. | intentional | accepted — CPU-only warning | CPU-only behavior is documented/accepted. |
| P1-23 | Only `HIP_VISIBLE_DEVICES` is honored, not `CUDA_VISIBLE_DEVICES`. | should | fixed — CUDA/HIP visibility | Visibility handling matches the supported platforms. |
| P1-24 | Residency is a conservative port, not byte-identical; no `fit_gpu_layers` auto-ngl. | should | accepted — conservative residency port | A parity test covers the plan numbers and auto-ngl is ported or accepted. |
| P1-25 | `stacks validate` does not include VRAM residency (matches Python; residency is in `serve`). | intentional | accepted — validate is structural | Accepted, or validate gains a residency section. |
| P1-26 | `doctor` hardware/memory details differ on non-AMD boxes. | should | fixed — shared AMD/NVIDIA hardware probe | Hardware probes share one implementation or differences are documented. |

### Downloads and onboarding

| ID | Caveat | Severity | Status | Close when |
|---|---|---|---|---|
| P1-27 | `models download` has no Hugging Face cache layout, ETag, or checksum verification. | should | accepted — simpler download path documented; length checks cover truncation | Download integrity/cache behavior is verified or documented as a simpler path. |
| P1-28 | Downloads do not compare the final byte count with `Content-Length`; a truncated stream can be renamed to the final file. | should | fixed — CLI + daemon length check | Final length is checked and `.part` is kept on mismatch. |
| P1-29 | `backends fetch` handles Linux `tar.gz` only; Windows `zip` is unsupported. | deferred | deferred — Windows zip | Windows backend install is implemented. |
| P1-30 | Backend archives have no checksum/size verification beyond extraction and `--version`. | should | accepted — manifest has no checksums; extraction+version check | Archive length/hash is verified before install. |
| P1-31 | `setup` always fetches the default Vulkan build; no `--tag` passthrough. | minor | accepted — fixed default Vulkan build | `setup` accepts a tag or documents the fixed default. |
| P1-32 | Proxy, custom-CA, and TLS behavior is not verified against Python's `requests` stack. | minor | accepted — loopback/HF traffic only | Network commands are tested with a proxy/custom CA or documented. |
| P1-33 | Network failure messages differ from Python. | minor | accepted — JSON/error text contract | Error text is normalized or accepted. |
| P1-34 | Folder imports create a directory symlink that `models list` does not traverse (inherited Python `rglob` behavior). | minor | accepted — inherited Python import/list behavior | The import/list contract is documented or fixed. |
| P1-35 | Remote `serve --model` with several GGUFs lists names but not sizes. | minor | accepted — names are enough for selection | The error lists file/size rows like Python. |

### Schema and validation

| ID | Caveat | Severity | Status | Close when |
|---|---|---|---|---|
| P1-36 | `parse_stack` does not fully validate `launch` blocks or the curated `engine_args` set. | should | fixed 4c766dc | Invalid launch/engine args are rejected like Python's schema. |
| P1-37 | Endpoint URL validation is weaker (no host/netloc check). | minor | fixed 4c766dc | Endpoint URLs are parsed and validated like Python. |
| P1-38 | `stacks show` routing key order can differ; parsed JSON is equal. | minor | accepted — parsed JSON equal | Ordered routing serialization or accepted as parsed-equal. |
| P1-39 | `serve` passes `engine_args` through without the schema's allowed-flag check. | should | fixed 4c766dc | Same schema validation as P1-36 before spawn. |

### TUI

| ID | Caveat | Severity | Status | Close when |
|---|---|---|---|---|
| P1-40 | Rust TUI was read-only: no stop, log, approvals, or open-web actions. | should | fixed — TUI log/stop/Studio approvals/web shortcuts | Actions are ported or explicitly accepted as Python-only. |
| P1-41 | No help overlay, mouse support, or scrollable detail pane; long JSON is clipped. | minor | accepted — basic TUI; UX review follow-up | Basic help/scroll is added or accepted. |
| P1-42 | TUI is Linux-terminal only. | deferred | deferred — Linux terminal | Windows/other terminal support resumes. |

### Testing and CI

| ID | Caveat | Severity | Status | Close when |
|---|---|---|---|---|
| P1-43 | No frozen golden-file corpus; conformance compares against a live Python install. | blocker | fixed — frozen CLI corpus | `tests/conformance/golden/` snapshots exist, with volatile fields normalized. |
| P1-44 | CI is Linux-only; Rust build/test/conformance run inside the Python job. | deferred | deferred — Linux CI | A dedicated Rust job or Windows matrix is added. |
| P1-45 | No `cargo audit`, license audit, or fuzzing for the GGUF/tar parsers. | should | accepted — audit/fuzz tracked for release CI | Audit/fuzz jobs run in CI. |
| P1-46 | No performance regression CI; 20 ms/RSS numbers are manual measurements. | minor | accepted — manual measurements; full gate records regressions | A benchmark guard records/compares them. |
| P1-47 | No `rust-toolchain.toml` or explicit MSRV; CI and local builds can drift. | minor | fixed — rust-toolchain.toml | Toolchain/MSRV is pinned. |
| P1-48 | `tests/conformance/test_install_layout.py` installs the Rust runtime into a temp prefix, runs `hive version --json`, checks the webui/unit layout, uninstalls it, and asserts `hive-py` remains the Python entry point. | should | fixed — install layout test | A packaging/install test covers both entry points. |

### Status and probes

| ID | Caveat | Severity | Status | Close when |
|---|---|---|---|---|
| P1-49 | `hive status` probing is HTTP-only; HTTPS endpoints and chunked responses are unsupported. | should | accepted — loopback HTTP endpoints | Probe transport covers the supported endpoint set. |
| P1-50 | `status` spawns one thread per unique endpoint; Python caps at 8. | minor | fixed — max 8 concurrent probes | A bounded pool is used or accepted. |
| P1-51 | Probe failure text differs from Python. | minor | accepted — JSON/error text contract | Error text is normalized or accepted. |
| P1-52 | CPU-only launches (`ngl=0`) were still charged against VRAM, and `serve --model` generated `ngl=99` even for CPU backends. Residency now skips VRAM for `ngl=0` and generated stacks use `ngl=0` when no GPU/backend is selected. | should | fixed — CPU-only residency + generated ngl | CPU-only tiers do not require VRAM, and generated CPU stacks set `ngl=0`. |
| P1-53 | CPU-only stacks still printed per-card display warnings, and `hive serve --model <library.gguf>` did not resolve a bare file name from the model library. Residency now suppresses display warnings for uncharged cards, and `serve` checks the library after the local path. | minor | fixed — UX walkthrough 2026-11-23 | CPU-only plans have no GPU warnings and library model names resolve. |
| P1-54 | The bare `hive` dashboard always suggested `hive serve peer-2tier`, even on a clean home with no such stack. It now suggests the first saved stack, a library model, or `hive models download`. | minor | fixed — UX walkthrough 2026-11-23 | The first-run hint is valid for the current home. |

---

## Phase 2 — Rust daemon and Studio API

| ID | Caveat | Severity | Status | Close when |
|---|---|---|---|---|
| P2-01 | The daemon now spawns and owns the Python sidecar, but it is still a standalone binary: `hive ui`/the console entry point does not start it. | should | fixed — `hive ui web` | The normal console entry point starts the daemon instead of Python uvicorn. |
| P2-02 | Sidecar lifecycle is not implemented: no spawn, restart/backoff, version handshake, or drain on daemon shutdown. | should | fixed — managed sidecar | Sidecar start/stop/restart and a `/health` version handshake are covered by tests. |
| P2-03 | `/v1/config` and `/v1/version` are new Rust routes with no Python counterpart or frozen golden contract. | should | fixed — daemon golden snapshots | Their schemas are frozen in the golden corpus and documented. |
| P2-04 | `/health` returns 200 with `ok:false` when the sidecar is down; Python `/health` only reports sidecar liveness. | intentional | accepted — daemon readiness semantics | Accepted as daemon readiness semantics, or aligned with Python. |
| P2-05 | Auth is only the `HARNESS_TOKEN` env and a plain string compare; no config file, rotation, or constant-time compare. | should | fixed — constant-time compare, env-only source | Token source and comparison are hardened/documented. |
| P2-06 | Proxy has no request IDs, body limits, or per-route timeout policy; blanket 300 s client timeout. | should | fixed — request IDs + per-route timeouts | Middleware limits/timeouts/request IDs match the existing API contract. |
| P2-07 | CORS is permissive; Python uses `_cors_origins()`. | should | fixed — HARNESS_CORS_ORIGINS parity | CORS origins are read from the same config as Python. |
| P2-08 | The daemon serves the `harness/webui/` shell, assets, and `/api` rewrite; the console's Python-only `/api` endpoints are tracked separately in P2-19. | should | fixed — static shell + `/api` rewrite | All console `/api` endpoints are native or explicitly proxied, then Python's console thread can be retired. |
| P2-09 | No OpenAPI/TypeScript generation or generated client. | should | fixed — merged OpenAPI document | OpenAPI/TS output is part of CI. |
| P2-35 | No TypeScript client generation; the console is vanilla JS, so nothing compiles against the OpenAPI document yet. | deferred | deferred — no typed frontend | A typed client or SDK exists and is generated in CI. |
| P2-36 | The live Studio chat test exercises `/v1/splinter/stream` only; it needs the Python encoder/torch and a live model, and cascade runs are not covered end-to-end. | should | fixed — live chat + cascade run | A live cascade run and a chat turn run in the release gate without skips, or the remaining Python dependency is accepted. |
| P2-37 | G2 says "all route groups in Rust", but groups 5–8 (cascade, splinter/openai/mcp, gateway, agent/activity/controller/training) are Python runtimes the staged plan keeps until Phase 3 or later. | intentional | fixed — G2 revised 2026-11-13 | G2 is revised to native control-plane groups + verified proxies, or those runtimes move native. |
| P2-38 | Groups 5–8 are verified as proxies against the real sidecar (`activity`, `tools`, `agent`, `sandbox` smoke), but their deeper conformance still runs against the Python gateway. | should | accepted — Python policy oracle until the gateway phase | The security/gateway conformance suite runs against the Rust public port, or the proxy boundary is accepted. |
| P2-10 | Every response carries the new `x-hive-owner` header; clients do not expect it. | minor | accepted — harmless owner header | Accepted as harmless, or gated behind a debug/version flag. |
| P2-11 | `/v1/setup/status` is 410 on Linux (matching Python's guard); the Windows proxy branch is untested. | deferred | deferred — Windows | Windows review resumes; the Linux 410 remains intentional. |
| P2-12 | Proxy tests use a fake Python HTTP server, not the real FastAPI sidecar. | should | fixed — managed-sidecar conformance | A conformance test starts the real `harness` sidecar and proxies a route. |
| P2-13 | Daemon is loopback/HTTP only; no TLS or remote-bind hardening. | intentional | accepted — loopback only | Remote/TLS deployment becomes a supported mode. |
| P2-14 | No rate limits or request-size limits at the public port. | should | accepted — loopback-only | Limits are configured and tested. |
| P2-15 | No packaging, systemd unit, or restart policy for `hive-daemon`. | should | fixed — systemd unit + daemon docs; release tarball tracked by P1-05 | Install/service packaging ships with the daemon. |
| P2-16 | SSE/streaming cancellation is not explicitly tested; body streams rely on drop propagation. | should | fixed — disconnect test | A disconnect test proves the upstream request is cancelled. |
| P2-17 | Proxy strips response `Content-Length` and re-chunks; request `Content-Length` is preserved. | minor | accepted — required for streaming | Accepted after real-sidecar tests, or preserved end-to-end. |
| P2-18 | No audit/structured request logs beyond basic `tracing` spans. | should | fixed — access log + request IDs | Audit fields and request IDs are added. |
| P2-19 | Console-local endpoints are partly native (`/api/roles`, `/api/routing/traces`, `/api/route-targets`, `/api/decision`, `/model/*`, `/probe/*`); `/api/preferences*`, `/api/judge/*`, `/api/benchmark`, and `/api/route` still need the Python console. | should | accepted — Python console endpoints; UX review covers panels | A conformance test covers each remaining console endpoint, or they are accepted as Python-console-only. |
| P2-20 | `POST /v1/server/start` now keeps a multi-model fleet and mirrors providers/engines, but still ignores `auto_fit`, `embedding`, and HF-repo passthrough. | should | accepted — embedding/HF launches use Python; native fleet covers local GGUFs | The remaining launch options are native or explicitly accepted. |
| P2-21 | `GET /v1/server/status` probes Strata now, but hardware fields are a subset and per-model page-cache share is absent. | should | accepted — status subset; hardware probe shared | Hardware/cache-share fields match Python and status parity is tested. |
| P2-22 | Fleet starts stop per key, but stopping a tier by key inside an applied stack still unloads the whole stack; per-tier unload is not implemented. | should | fixed — supervisor stop_role | The supervisor can stop one stack tier while keeping the rest applied. |
| P2-23 | Stack PUT uses the Rust shape parser and an atomic rename; Python's exact normalization/error text is not frozen in a golden corpus. | should | fixed — daemon stack golden | Stack CRUD responses match the frozen golden corpus field-for-field. |
| P2-24 | Native server/start launches are not registered as providers/engines in the Python sidecar, so chat routing cannot see them yet. | should | fixed — providers mirrored | Provider/engine registration moves native with the server group, or the daemon mirrors it into the sidecar. |
| P2-33 | The daemon mirrors `local-<key>` providers into the sidecar but not engine profiles, so the engine picker does not show native launches. | should | fixed — engines mirrored | Engine profiles are mirrored or the engine registry moves native. |
| P2-25 | Native server status reports `healthy` from the last apply, not a fresh engine probe; a crashed engine still looks loaded. | should | fixed — pid + endpoint probe | Status probes each engine and prunes dead instances. |
| P2-26 | Native unload does not drain an in-flight generation before stopping the engine (the Python manager waits). | should | fixed — supervisor drain | Unload drains slots with a bounded timeout before terminating, closing P1-12 for the daemon too. |
| P2-27 | Linux model routes are native (list/delete/path/import-path/multipart import, hub); `/v1/models/linux` and move-to-linux/windows remain Python/WSL-only. | should | fixed — Linux routes; WSL deferred with Windows | The remaining import/WSL routes are native or accepted as Python-only. |
| P2-28 | `GET /v1/models` (served model IDs from a provider endpoint) still proxies; the native server status covers loaded engines only. | should | fixed — native fleet listing with proxy fallback | The served-model listing reads native instances and provider records. |
| P2-29 | Daemon hub tests use `HIVE_HF_API`/`HIVE_HF_BASE` overrides; production defaults are the public Hugging Face endpoints. | minor | accepted — test-only env overrides | The overrides are documented as test hooks, or the hub client is injectable without env. |
| P2-30 | `/v1/runs` and `/v1/report/*` are native, but the rendered `/view/*` and `/runs` HTML pages still proxy to Python's report renderer. | should | accepted — report renderer stays Python | The report renderer is native, or those pages are accepted as Python-only until the report group moves. |
| P2-31 | `hive ui web` serves the console on the API port; `--console-port` is accepted for Python CLI parity but ignored. `--detach` writes `logs/daemon.pid`; there is no `hive ui stop` verb. | minor | accepted — console shares API port; --detach documented | The console can bind its historical port, or the divergence is documented and accepted. |
| P2-32 | Python-only verbs are proxied through `python -m harness.cli`; they still need the Python reference install and do not appear in `hive help --json` native topics. | intentional | accepted — Python-only verbs proxied | The verbs move native in their phases; until then the proxy is the supported path. |
| P2-33 | The clean installer did not copy `tools/backends/manifest.json`, and an earlier copy under `$HIVE_HOME/harness/data` shadowed the Python `harness` package when the sidecar ran with cwd `$HIVE_HOME`. The manifest now installs under `$HIVE_HOME/tools/backends` and `$PREFIX/share/hive`; the installer removes the old shadow copy. | blocker | fixed — UX walkthrough 2026-11-23 | Clean installs can run `hive setup`/`doctor` outside the checkout and the sidecar imports. |
| P2-34 | The Studio shell's `/api/v1` lifecycle calls were proxied to the Python sidecar, so Load failed and Lifecycle showed no loaded engines even when the Rust daemon owned them. Native server/stack routes now also answer `/api/v1/*`. | blocker | fixed — UX walkthrough 2026-11-23 | Studio Load/Lifecycle use the Rust runtime state. |
| P2-35 | The canonical `/v1/chat/completions` returned the sidecar's model-free mock response, and engine model IDs were absolute GGUF paths. The route now forwards to the curated OpenAI passthrough, and llama tiers default to a `--alias` from the model file name. | should | fixed — UX walkthrough 2026-11-23 | OpenAI clients get real replies and stable model IDs. |
| P2-36 | `HARNESS_TOKEN` accepted only `x-splinter-token`; standard OpenAI clients send `Authorization: Bearer`. Both forms are accepted in the daemon and the sidecar. | should | fixed — UX walkthrough 2026-11-23 | Bearer-authenticated OpenAI clients can use a protected daemon. |
| P2-37 | A clean home without the native encoder model and without torch returned a 500 and could take the sidecar down. The daemon now auto-detects `$HIVE_HOME/models/encoder`, `hive setup` fetches it, and the sidecar returns a 503 with the install command. | blocker | fixed — UX walkthrough 2026-11-23 | Studio memory works from a clean install or fails with a clear 503. |
| P2-38 | The native sandbox defaulted to `/srv/hive-ai/workspaces`, which is not writable for a normal user, so every native exec failed. It now falls back to `$HIVE_HOME/harness_state/workspaces`. | should | fixed — UX walkthrough 2026-11-23 | Sandbox exec works in a user install; AI-user deployments keep `/srv/hive-ai`. |
| P2-39 | At narrow widths the rail/nav forced horizontal overflow and hash links did not switch pages. The responsive CSS constrains the nav and the console listens for `hashchange`. | minor | fixed — UX walkthrough 2026-11-23 | Narrow viewport has no horizontal scroll and deep links work. |
| P2-40 | The documented Python reference install omitted `splinter[serve]`, so `python -m harness` failed with `ModuleNotFoundError`. `docs/INSTALL.md`, the installer output, and `hive doctor` now name the package. | blocker | fixed — UX walkthrough 2026-11-23 | A clean sidecar install is documented and diagnosed. |
| P2-41 | The Overview hardcoded the DS4.1/LFM3 role endpoints and showed them as down on a clean home. It now reports the daemon and only the engines it actually owns. | minor | fixed — UX walkthrough 2026-11-23 | Overview cards reflect live state, with no role presets. |
| P2-42 | `hive ui status` assumed `127.0.0.1:8765`, so a daemon on any other port looked stopped. The daemon writes `$HIVE_HOME/harness_state/daemon.json`; status reads it, checks the PID, and probes the recorded URL. `hive-py` reads the same descriptor. | should | fixed — UX walkthrough 2026-11-23 | Status reports the live port or off. |
| P2-43 | The Chat model picker and role picker listed hardcoded serving endpoints even when they were down, and dead probes filled the browser console with 502s. The pickers now probe endpoints and show only live ones; `/probe/{port}/health` answers a quiet 200 `{"status":"down"}`. | minor | fixed — UX walkthrough 2026-11-23 | Only reachable endpoints are selectable; dead probes are quiet. |
| P2-44 | A managed Python sidecar wrote the same `daemon.json` as the public daemon, replacing the public URL with the private sidecar port. Managed sidecars now set `HIVE_MANAGED_SIDECAR=1` and do not write the descriptor. | should | fixed — UX walkthrough 2026-11-23 | `ui status` always reports the public daemon port. |
| P2-45 | The daemon descriptor was a single `daemon.json`, so two daemons sharing one `HIVE_HOME` overwrote one another and `ui status` showed only the last. Each daemon now writes `daemons/<port>.json`; status lists all of them, with the primary descriptor kept for compatibility. | should | fixed — UX walkthrough 2026-11-23 | Multiple daemons in one state are all reported. |
| P2-46 | `hive doctor` checked only port 1234, ignoring a saved stack with multiple local tiers. It now checks one engine port per local tier in the largest saved stack (endpoint tiers excluded). | minor | fixed — UX walkthrough 2026-11-23 | Doctor checks the ports a stack will actually need. |


## Phase 3 — memory and encoder

| ID | Caveat | Severity | Status | Close when |
|---|---|---|---|---|
| P3-01 | The native ONNX encoder is built and parity-tested, but not integrated into the sidecar or daemon; the runtime still computes embeddings in Python. | blocker | fixed — daemon-managed encoder + sidecar client | The sidecar/daemon uses `hive-encoder` for embeddings and can run without torch/transformers/sentence-transformers. |
| P3-02 | The daemon hosts a native per-conversation memory API; the Python Splinter uses it by default whenever the managed native encoder is active. Component parity, a 5-turn conversation, a 15-turn fixture replay, and the P1–P11 protocol replay match Python; the native path is the default runtime path. | blocker | fixed — daemon-hosted native memory default | The Rust memory core matches the P1–P11 replay and is the default path. |
| P3-11 | Native assembly now runs the remembrance pass on deletion candidates and accepts comb candidates for scoring, dedup, decay exemption, budget selection, and relevance-history updates. `test_daemon_memory_comb_and_remembrance_match_python` compares ids, content, tokens, budget, raw scores, remembrance state, and history against the Python assembler. | should | fixed — comb/remembrance wired into native assembly | Comb/remembrance are native and match the Python replay. |
| P3-13 | The earlier ~66 ms figure was a debug build with an O(n²) cosine matrix that recomputed both norms per pair. Release measurement for 71 chunks: warm 3.55 ms native vs 2.98 ms cached Python (1.19x); evolving turns with a new query + chunk 13.5 ms vs 12.0 ms (1.12x). `experiments/bench_memory_assembly.py` reproduces it. | should | fixed — release parity within 1.2x | A persistent encoder connection/in-process path brings assembly latency to Python parity or the overhead is accepted. |
| P3-12 | P1–P11 replays against Python and native memory with routing pinned to ultra_small and the real MiniLM drone: every status matches; P1 differs only in timing noise and P10 by 0.2 points from ONNX embedding float noise. The 28-task cascade is a generation-only path with no memory calls; it is out of scope for the memory replay and remains covered by the Phase 2 cascade gate (one-task live report, P2-36). | should | fixed — P1–P11 replay matches | The protocol replay produces the same reports on Python and native memory. |
| P3-03 | ONNX Runtime adds a 27 MB release binary and a `libstdc++` runtime dependency; the combined install size has not been measured against the Python/torch path. | should | fixed — torch-free install measured 381 MB vs 1.6 GB | Release packaging compares the two installs and accepts the smaller/leaner path. |
| P3-04 | The encoder model is discovered from `HIVE_ENCODER_DIR`, `<repo>/models/encoder`, or the HF cache; it is not packaged or downloaded by Hive. | should | fixed — `hive-encoder fetch` with SHA256 | The encoder asset ships with the install or has a documented, checksummed fetch path. |
| P3-05 | Vocabulary boost is ported; MC-dropout confidence is fixed at 1.0. | should | fixed — stock Model confidence is 1.0 in both | The native scorer matches `UltraSmallDrone.score` including confidence knobs. |
| P3-06 | Bulk throughput is roughly at parity and varies by machine; the native advantage is startup/RSS and one-text latency. | minor | fixed — benchmark documented | A stable benchmark gate records the accepted tradeoff. |
| P3-07 | Only `paraphrase-MiniLM-L3-v2` is ONNX-validated; other `ultra_model` choices need their own export and parity corpus. | should | deferred — default model only | Supported encoders are listed with per-model parity tests. |
| P3-08 | Native encoder tests require the HF model cache and Python sentence-transformers as the oracle; they skip otherwise. | minor | fixed — frozen encoder vectors | A frozen vector/golden corpus removes the Python oracle requirement. |
| P3-09 | No torch-free environment test yet; the encoder path avoids loading sentence-transformers, but the sidecar install still carries torch for other surfaces. | blocker | fixed — torch-free venv runs chat | A minimal runtime venv without torch/transformers runs chat and memory end-to-end. |
| P3-10 | Daemon-managed encoder starts only for a managed sidecar (`HARNESS_EMBEDDING_BACKEND=rust`); an external `--sidecar-url` must be pointed at `hive-encoder` manually. | minor | fixed — documented external-sidecar step | External-sidecar mode accepts an encoder URL or documents the manual step. |

## Phase 4 — differentiators

| ID | Caveat | Severity | Status | Close when |
|---|---|---|---|---|
| P4-01 | `/metrics` exposes daemon request/latency counters, stream TTFT/delta events, process RSS/threads, per-engine up/pid gauges, and healthy llama-server samples. VRAM/queue/tok-s labels only appear if the engine build exposes them. `/ready` gates on sidecar and expected engines. | should | fixed — native Prometheus metrics + readiness | The operational metrics contract is native and tested. |
| P4-02 | Zero-downtime swaps start the replacement first, health-gate it, flip the sidecar provider/engine default, then drain the old engine; the original key is preserved. | should | fixed — `/v1/server/swap` + live test | A live swap leaves the old port closed and the same key serving on the new port. |
| P4-03 | `hive-sandbox` applies Landlock filesystem rules plus a seccomp deny-list (network, mount, ptrace, bpf, io_uring, keyring, process_vm) in a single-threaded helper, with rlimits and a scrubbed environment. `/v1/sandbox/exec` and `/v1/sandbox/status` use it by default (`mode=auto`); `mode=landlock` fails closed, `mode=docker` keeps the Docker/Python path, and unavailable native support falls back only in auto mode. The timeout kills the process group. | blocker | fixed — native sandbox is the default backend | `mode=landlock|auto` routes the existing sandbox API through the native helper, with Docker as an explicit fallback. |
| P4-04 | A `hive-plugins` host runs import-free WASM modules under wasmtime with deterministic fuel, a linear-memory cap, one instance/table/memory, a 1 MB input cap, and a SHA256 module pin. Signed manifests are required by default and verified against `plugins/trusted-keys.json` or `HIVE_PLUGIN_TRUSTED_KEYS`; `scripts/sign_plugin.py` generates Ed25519 keys, signs, and verifies. `GET /v1/plugins` and `POST /v1/plugins/{name}/run` expose the host. Tests cover round-trip JSON, fuel exhaustion, memory growth, hash mismatch, trusted-key absence, and tampered signatures. | should | fixed — wasmtime host + signed manifests | Plugins run under wasmtime with fuel and memory limits and a signed manifest. |
| P4-05 | `scripts/release_manifest.py` generates `release-manifest.json`, `SHA256SUMS`, and a CycloneDX 1.5 SBOM; `scripts/sign_artifact.py` signs/verifies them with Ed25519. A development key (`dev-release-2026`) validated the path locally; `dist-workspace.toml` declares the Linux cargo-dist targets. Production key/CI provisioning and an auto-update client remain. | should | accepted — signed artifacts + dev key; production key/CI + updater are deployment follow-ups | Release artifacts carry signatures/checksums plus an SBOM, and update verification is documented. |
| P4-06 | Stack roots accept `restart: never/on-failure/always` and `restart_backoff_s`; the daemon replaces a crashed stack with a fresh healthy engine and exponential backoff, preserving the public key. Tiers accept a `cgroup` object (`memory_max`, `cpu_max`, `pids_max`); budgets are applied when cgroup v2 is delegated, best-effort by default and fatal under `HIVE_CGROUP_STRICT=1`. `test_live_cgroup_budget_is_applied` validated real parameters and cleanup under the user systemd cgroup. | should | fixed — restart + cgroup budgets | Stacks run under cgroup limits and restart with bounded backoff. |
| P4-07 | Stream TTFT and delta counts are route-level, and delta counts are approximate SSE event counts rather than tokenizer tokens. | minor | accepted — minor precision; engine metrics carry tokens where available | Per-tier TTFT/token metrics come from engine usage data. |

---

## Cross-phase

| ID | Caveat | Severity | Status | Close when |
|---|---|---|---|---|
| X-01 | `main`/`hive_test` are frozen while `rust-migration` advances; branches will diverge. | intentional | accepted — merge gate reconciles | The final merge gate reconciles the branches. |
| X-02 | Windows support is deferred across the rewrite. | deferred | deferred — Windows | Windows review resumes. |
| X-03 | JSON is the compatibility contract; human output/help text is approximate. | minor | accepted — JSON contract | Accepted or human snapshots are added. |
| X-04 | `docs/MIGRATION-HIVE.md` is the user-facing switch guide (install, what moved, rollback, compatibility). | should | fixed — migration guide | A migration note ships with the final switch. |
| X-05 | `scripts/install-rust.sh --uninstall` removes the Rust binaries, webui copy, and unit without touching state; `docs/INSTALL.md` documents the `hive-py` rollback. | should | fixed — install script uninstall | The install/switch path documents or automates rollback. |
| X-06 | The Python Studio/eval surfaces stay Python by design; memory is the Phase 3 native target. | intentional | accepted — Python research/Studio boundary | Accepted as the Phase 3 boundary. |
| X-07 | Docs can drift as phases move; historical markers are not a substitute for a link audit. | minor | fixed — link audit passed 2026-11-15 | Pre-merge link/doc audit passes. |
| X-08 | No structured audit/telemetry in the Rust supervisor yet. | intentional | accepted — access logs/request IDs; deeper audit later | Phase 2 adds tracing/audit where needed. |
