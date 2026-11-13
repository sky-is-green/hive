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

## Merge-blocker shortlist

- `P1-01` Rust is not the default `hive` yet.
- `P1-05` No installable Rust release/package yet.

---

## Phase 0 — console de-bloat and contract freeze

| ID | Caveat | Severity | Status | Close when |
|---|---|---|---|---|
| P0-01 | Windows remains quarantined: no required Windows CI, no validated Windows CLI/TUI path. | deferred | open | Windows review resumes; the deferred Windows checklist is executed. |
| P0-02 | `/v1/setup/*` returns 410 on Linux; the Windows setup flow is intentionally unavailable. | intentional | open | Linux setup remains the only supported path, or Windows work resumes. |
| P0-03 | Suggestions inbox and `/api/hive_build/suggestions` were removed; external consumers break. | intentional | open | No external consumer depends on the old endpoints. |
| P0-04 | The legacy `/server` page/routes were removed; old bookmarks and scripts break. | intentional | open | No live links/scripts target `/server`. |
| P0-05 | `--no-auto-start` was removed; scripts using it fail. | intentional | open | No supported script uses the old flag. |
| P0-06 | The `strata.server` fallback was removed; deployments relying on it break. | intentional | open | Deployment configs use the supported launch/endpoint blocks. |
| P0-07 | Run artifacts were archived locally (`work/hive-artifacts-archive-2026-11-12`), not uploaded to HF. | deferred | open | Archive is uploaded or no longer needed. |
| P0-08 | Placement gate falls back to weights-only with a warning for hosted/unknown models. | intentional | open | A hosted-model KV source exists, or the warning is accepted. |
| P0-09 | Placement gate is a preflight, not a load probe; context/KV are operator-selected defaults. | intentional | open | Accepted as a preflight contract. |
| P0-10 | Placement gate refuses models with unknown weight sizes. | intentional | open | Accepted as a safety rule. |
| P0-11 | The release gate is still a manual browser sweep. | minor | open | A repeatable automated console sweep is added. |
| P0-12 | Historical docs are marked instead of deleted; external deep links may drift. | minor | open | Link audit passes before merge. |

## Phase 1 — Rust CLI and supervisor

### Switch, packaging, and surface

| ID | Caveat | Severity | Status | Close when |
|---|---|---|---|---|
| P1-01 | `hive` still points at the Python CLI; the Rust binary is opt-in on `rust-migration`. | blocker | open | The final rewrite gate switches the `hive` entry point and keeps `hive-py`. |
| P1-02 | `hive-py` exists in `pyproject.toml`, but existing venvs need `pip install -e .` before the command appears. | minor | open | Install/upgrade docs and a packaging test cover the new script. |
| P1-03 | Python-only verbs (`cascade`, `tools`, `sandbox`, `web`, `browser`, `ui`, `profile`, `controller`, `encoder`) are not proxied by Rust. | blocker | fixed — CLI proxy for Python-only verbs | Phase 2 proxy covers them, or they are explicitly accepted as Python-only. |
| P1-04 | Root `hive help --json` lists only ported commands. | intentional | open | P1-03 lands, or the subset is accepted. |
| P1-05 | There is no installable Rust release artifact, signed binary, or package-manager story. | blocker | open | `cargo build --release` is wrapped in a documented install/release path. |
| P1-06 | Rust Python discovery is heuristic (`HIVE_PYTHON`, `VIRTUAL_ENV`, checkout `.venv`, PATH). Decision: reuse `HIVE_PYTHON` as the explicit sidecar contract; discovery is only a development fallback. | should | fixed — strict sidecar resolve | Phase 2 daemon reads `HIVE_PYTHON` first and fails clearly when it names a missing/unusable interpreter. |
| P1-07 | `hive version --json` takes ~21 ms because it spawns Python; plain `version`/`status` meet the 20 ms budget. | minor | open | Cache the Python version or accept the JSON-only cost. |
| P1-08 | Human output and clap help text are not byte-identical to argparse; JSON is the compatibility contract. | minor | open | Accepted, or human output is snapshotted. |
| P1-09 | No shell completions or man pages. | minor | open | Completions/man pages are generated. |

### Supervisor

| ID | Caveat | Severity | Status | Close when |
|---|---|---|---|---|
| P1-10 | Managed Strata/DeepInfer launch tiers are fake-repo tested only. DeepInfer was a capability probe, not runtime support; Strata needs per-model support. | deferred | accepted — revisit when the real lanes return | A supported Strata model and the DeepInfer capability lane are run end-to-end; DeepInfer is not presented as supported runtime. |
| P1-11 | A `kill -9` of the Rust supervisor may leave launch-tier grandchildren behind if the Python launcher ignores signals. | deferred | accepted — tied to P1-10 | Revisit with P1-10; a real launch-tier kill -9 test leaves no engine processes. |
| P1-12 | No `drain_slots` wait before unload; a model killed mid-generation loses work. | should | fixed — supervisor drain | Unload polls `/slots` with the Python timeout before SIGTERM. |
| P1-13 | No log rotation or size cap for supervisor/engine logs. | minor | open | A size cap or rotation policy is implemented/accepted. |
| P1-14 | Port allocation is check-then-spawn; a race can lose the port. | minor | open | A reservation/lock covers the spawn window, or the race is accepted. |
| P1-15 | `hive logs -f` exits 130 on Ctrl-C; Python exits 0. | minor | open | The follow loop catches SIGINT and returns 0. |
| P1-16 | `serve --model … --detach` requires `--save`; Python can detach without saving. | should | open | The child argv carries the model flags or a temporary stack. |
| P1-17 | `HARNESS_MEMORY_GUARD=0` bypass is not ported; Rust residency always runs when devices exist. | should | open | The env override is honored or explicitly rejected. |
| P1-18 | Local engine health checks hardcode `127.0.0.1`; a non-loopback `--host` can fail. | minor | open | Health probes use the configured host/address. |
| P1-19 | Serve records and error strings are not in the golden corpus; only selected fields are asserted. | minor | open | Serve records are snapshotted with volatile fields normalized. |
| P1-20 | `stop --force` process-group behavior for launch tiers is not tested against a real engine tree. | deferred | accepted — tied to P1-10 | Revisit with P1-10; a real launch-tier force-stop leaves no children. |

### Hardware and residency

| ID | Caveat | Severity | Status | Close when |
|---|---|---|---|---|
| P1-21 | Hardware probing is AMD sysfs only; NVIDIA/Intel devices are invisible to Rust. | should | open | NVIDIA/Intel probes or an explicit Linux-AMD scope decision. |
| P1-22 | With no visible GPU, Rust proceeds with a warning; Python refuses. | intentional | open | CPU-only behavior is documented/accepted. |
| P1-23 | Only `HIP_VISIBLE_DEVICES` is honored, not `CUDA_VISIBLE_DEVICES`. | should | open | Visibility handling matches the supported platforms. |
| P1-24 | Residency is a conservative port, not byte-identical; no `fit_gpu_layers` auto-ngl. | should | open | A parity test covers the plan numbers and auto-ngl is ported or accepted. |
| P1-25 | `stacks validate` does not include VRAM residency (matches Python; residency is in `serve`). | intentional | open | Accepted, or validate gains a residency section. |
| P1-26 | `doctor` hardware/memory details differ on non-AMD boxes. | should | open | Hardware probes share one implementation or differences are documented. |

### Downloads and onboarding

| ID | Caveat | Severity | Status | Close when |
|---|---|---|---|---|
| P1-27 | `models download` has no Hugging Face cache layout, ETag, or checksum verification. | should | open | Download integrity/cache behavior is verified or documented as a simpler path. |
| P1-28 | Downloads do not compare the final byte count with `Content-Length`; a truncated stream can be renamed to the final file. | should | fixed — CLI + daemon length check | Final length is checked and `.part` is kept on mismatch. |
| P1-29 | `backends fetch` handles Linux `tar.gz` only; Windows `zip` is unsupported. | deferred | open | Windows backend install is implemented. |
| P1-30 | Backend archives have no checksum/size verification beyond extraction and `--version`. | should | open | Archive length/hash is verified before install. |
| P1-31 | `setup` always fetches the default Vulkan build; no `--tag` passthrough. | minor | open | `setup` accepts a tag or documents the fixed default. |
| P1-32 | Proxy, custom-CA, and TLS behavior is not verified against Python's `requests` stack. | minor | open | Network commands are tested with a proxy/custom CA or documented. |
| P1-33 | Network failure messages differ from Python. | minor | open | Error text is normalized or accepted. |
| P1-34 | Folder imports create a directory symlink that `models list` does not traverse (inherited Python `rglob` behavior). | minor | open | The import/list contract is documented or fixed. |
| P1-35 | Remote `serve --model` with several GGUFs lists names but not sizes. | minor | open | The error lists file/size rows like Python. |

### Schema and validation

| ID | Caveat | Severity | Status | Close when |
|---|---|---|---|---|
| P1-36 | `parse_stack` does not fully validate `launch` blocks or the curated `engine_args` set. | should | fixed 4c766dc | Invalid launch/engine args are rejected like Python's schema. |
| P1-37 | Endpoint URL validation is weaker (no host/netloc check). | minor | fixed 4c766dc | Endpoint URLs are parsed and validated like Python. |
| P1-38 | `stacks show` routing key order can differ; parsed JSON is equal. | minor | open | Ordered routing serialization or accepted as parsed-equal. |
| P1-39 | `serve` passes `engine_args` through without the schema's allowed-flag check. | should | fixed 4c766dc | Same schema validation as P1-36 before spawn. |

### TUI

| ID | Caveat | Severity | Status | Close when |
|---|---|---|---|---|
| P1-40 | Rust TUI is read-only: no stop, log, approvals, or open-web actions. | should | open | Actions are ported or explicitly accepted as Python-only. |
| P1-41 | No help overlay, mouse support, or scrollable detail pane; long JSON is clipped. | minor | open | Basic help/scroll is added or accepted. |
| P1-42 | TUI is Linux-terminal only. | deferred | open | Windows/other terminal support resumes. |

### Testing and CI

| ID | Caveat | Severity | Status | Close when |
|---|---|---|---|---|
| P1-43 | No frozen golden-file corpus; conformance compares against a live Python install. | blocker | fixed — frozen CLI corpus | `tests/conformance/golden/` snapshots exist, with volatile fields normalized. |
| P1-44 | CI is Linux-only; Rust build/test/conformance run inside the Python job. | deferred | open | A dedicated Rust job or Windows matrix is added. |
| P1-45 | No `cargo audit`, license audit, or fuzzing for the GGUF/tar parsers. | should | open | Audit/fuzz jobs run in CI. |
| P1-46 | No performance regression CI; 20 ms/RSS numbers are manual measurements. | minor | open | A benchmark guard records/compares them. |
| P1-47 | No `rust-toolchain.toml` or explicit MSRV; CI and local builds can drift. | minor | open | Toolchain/MSRV is pinned. |
| P1-48 | No packaging test that `hive-py` installs and resolves alongside the Rust binary. | should | open | A packaging/install test covers both entry points. |

### Status and probes

| ID | Caveat | Severity | Status | Close when |
|---|---|---|---|---|
| P1-49 | `hive status` probing is HTTP-only; HTTPS endpoints and chunked responses are unsupported. | should | open | Probe transport covers the supported endpoint set. |
| P1-50 | `status` spawns one thread per unique endpoint; Python caps at 8. | minor | open | A bounded pool is used or accepted. |
| P1-51 | Probe failure text differs from Python. | minor | open | Error text is normalized or accepted. |

---

## Phase 2 — Rust daemon and Studio API

| ID | Caveat | Severity | Status | Close when |
|---|---|---|---|---|
| P2-01 | The daemon now spawns and owns the Python sidecar, but it is still a standalone binary: `hive ui`/the console entry point does not start it. | should | fixed — `hive ui web` | The normal console entry point starts the daemon instead of Python uvicorn. |
| P2-02 | Sidecar lifecycle is not implemented: no spawn, restart/backoff, version handshake, or drain on daemon shutdown. | should | fixed — managed sidecar | Sidecar start/stop/restart and a `/health` version handshake are covered by tests. |
| P2-03 | `/v1/config` and `/v1/version` are new Rust routes with no Python counterpart or frozen golden contract. | should | fixed — daemon golden snapshots | Their schemas are frozen in the golden corpus and documented. |
| P2-04 | `/health` returns 200 with `ok:false` when the sidecar is down; Python `/health` only reports sidecar liveness. | intentional | open | Accepted as daemon readiness semantics, or aligned with Python. |
| P2-05 | Auth is only the `HARNESS_TOKEN` env and a plain string compare; no config file, rotation, or constant-time compare. | should | fixed — constant-time compare, env-only source | Token source and comparison are hardened/documented. |
| P2-06 | Proxy has no request IDs, body limits, or per-route timeout policy; blanket 300 s client timeout. | should | fixed — request IDs + per-route timeouts | Middleware limits/timeouts/request IDs match the existing API contract. |
| P2-07 | CORS is permissive; Python uses `_cors_origins()`. | should | fixed — HARNESS_CORS_ORIGINS parity | CORS origins are read from the same config as Python. |
| P2-08 | The daemon serves the `harness/webui/` shell, assets, and `/api` rewrite; the console's Python-only `/api` endpoints are tracked separately in P2-19. | should | fixed — static shell + `/api` rewrite | All console `/api` endpoints are native or explicitly proxied, then Python's console thread can be retired. |
| P2-09 | No OpenAPI/TypeScript generation or generated client. | should | fixed — merged OpenAPI document | OpenAPI/TS output is part of CI. |
| P2-35 | No TypeScript client generation; the console is vanilla JS, so nothing compiles against the OpenAPI document yet. | should | deferred — no typed frontend | A typed client or SDK exists and is generated in CI. |
| P2-36 | The live Studio chat test exercises `/v1/splinter/stream` only; it needs the Python encoder/torch and a live model, and cascade runs are not covered end-to-end. | should | open | A live cascade run and a chat turn run in the release gate without skips, or the remaining Python dependency is accepted. |
| P2-10 | Every response carries the new `x-hive-owner` header; clients do not expect it. | minor | open | Accepted as harmless, or gated behind a debug/version flag. |
| P2-11 | `/v1/setup/status` is 410 on Linux (matching Python's guard); the Windows proxy branch is untested. | deferred | open | Windows review resumes; the Linux 410 remains intentional. |
| P2-12 | Proxy tests use a fake Python HTTP server, not the real FastAPI sidecar. | should | fixed — managed-sidecar conformance | A conformance test starts the real `harness` sidecar and proxies a route. |
| P2-13 | Daemon is loopback/HTTP only; no TLS or remote-bind hardening. | intentional | open | Remote/TLS deployment becomes a supported mode. |
| P2-14 | No rate limits or request-size limits at the public port. | should | accepted — loopback-only | Limits are configured and tested. |
| P2-15 | No packaging, systemd unit, or restart policy for `hive-daemon`. | should | fixed — systemd unit + daemon docs; release tarball tracked by P1-05 | Install/service packaging ships with the daemon. |
| P2-16 | SSE/streaming cancellation is not explicitly tested; body streams rely on drop propagation. | should | fixed — disconnect test | A disconnect test proves the upstream request is cancelled. |
| P2-17 | Proxy strips response `Content-Length` and re-chunks; request `Content-Length` is preserved. | minor | accepted — required for streaming | Accepted after real-sidecar tests, or preserved end-to-end. |
| P2-18 | No audit/structured request logs beyond basic `tracing` spans. | should | fixed — access log + request IDs | Audit fields and request IDs are added. |
| P2-19 | Console-local endpoints are not ported: `/api/routing/traces`, `/api/preferences*`, `/api/judge/*`, `/api/benchmark`, `/api/route-targets`, `/api/route`, `/api/decision`, plus `/model/*` and `/probe/*`. | should | open | A conformance test covers each console endpoint against the Rust daemon, or they are accepted as Python-console-only. |
| P2-20 | `POST /v1/server/start` is one applied stack at a time and ignores provider/engine registration, `auto_fit`, `embedding`, and HF-repo passthrough. | should | open | Multi-instance start, provider/engine records, and the remaining launch options are native or explicitly accepted. |
| P2-21 | `GET /v1/server/status` approximates Python's hardware/strata/cache shape; Strata is reported as down and cache share is absent. | should | open | Strata probing and cache-share readout are ported, and status parity is tested. |
| P2-22 | `POST /v1/server/stop?key=` unloads the whole applied stack; per-tier unload is not implemented. | should | open | The supervisor can stop one engine while keeping the rest applied. |
| P2-23 | Stack PUT uses the Rust shape parser and an atomic rename; Python's exact normalization/error text is not frozen in a golden corpus. | should | open | Stack CRUD responses match the frozen golden corpus field-for-field. |
| P2-24 | Native server/start launches are not registered as providers/engines in the Python sidecar, so chat routing cannot see them yet. | should | fixed — providers mirrored | Provider/engine registration moves native with the server group, or the daemon mirrors it into the sidecar. |
| P2-33 | The daemon mirrors `local-<key>` providers into the sidecar but not engine profiles, so the engine picker does not show native launches. | should | fixed — engines mirrored | Engine profiles are mirrored or the engine registry moves native. |
| P2-25 | Native server status reports `healthy` from the last apply, not a fresh engine probe; a crashed engine still looks loaded. | should | fixed — pid + endpoint probe | Status probes each engine and prunes dead instances. |
| P2-26 | Native unload does not drain an in-flight generation before stopping the engine (the Python manager waits). | should | fixed — supervisor drain | Unload drains slots with a bounded timeout before terminating, closing P1-12 for the daemon too. |
| P2-27 | Native model routes cover local list/delete/path/import-path and hub search/files/download; multipart import, `/v1/models/linux`, and move-to-linux/windows still proxy to Python. | should | open | The remaining import/WSL routes are native or accepted as Python-only. |
| P2-28 | `GET /v1/models` (served model IDs from a provider endpoint) still proxies; the native server status covers loaded engines only. | should | open | The served-model listing reads native instances and provider records. |
| P2-29 | Daemon hub tests use `HIVE_HF_API`/`HIVE_HF_BASE` overrides; production defaults are the public Hugging Face endpoints. | minor | open | The overrides are documented as test hooks, or the hub client is injectable without env. |
| P2-30 | `/v1/runs` and `/v1/report/*` are native, but the rendered `/view/*` and `/runs` HTML pages still proxy to Python's report renderer. | should | open | The report renderer is native, or those pages are accepted as Python-only until the report group moves. |
| P2-31 | `hive ui web` serves the console on the API port; `--console-port` is accepted for Python CLI parity but ignored. `--detach` writes `logs/daemon.pid`; there is no `hive ui stop` verb. | minor | open | The console can bind its historical port, or the divergence is documented and accepted. |
| P2-32 | Python-only verbs are proxied through `python -m harness.cli`; they still need the Python reference install and do not appear in `hive help --json` native topics. | intentional | open | The verbs move native in their phases; until then the proxy is the supported path. |


## Phase 3 — memory and encoder

No caveats recorded yet.

## Phase 4 — optional differentiators

No caveats recorded yet.

---

## Cross-phase

| ID | Caveat | Severity | Status | Close when |
|---|---|---|---|---|
| X-01 | `main`/`hive_test` are frozen while `rust-migration` advances; branches will diverge. | intentional | open | The final merge gate reconciles the branches. |
| X-02 | Windows support is deferred across the rewrite. | deferred | open | Windows review resumes. |
| X-03 | JSON is the compatibility contract; human output/help text is approximate. | minor | open | Accepted or human snapshots are added. |
| X-04 | No user-facing changelog or migration guide for the `hive` → Rust switch. | should | open | A migration note ships with the final switch. |
| X-05 | No automated rollback from the Rust `hive` to `hive-py`. | should | open | The install/switch path documents or automates rollback. |
| X-06 | The Python Studio/eval/memory surfaces stay Python by design. | intentional | open | Accepted as the Phase 3 boundary. |
| X-07 | Docs can drift as phases move; historical markers are not a substitute for a link audit. | minor | open | Pre-merge link/doc audit passes. |
| X-08 | No structured audit/telemetry in the Rust supervisor yet. | intentional | open | Phase 2 adds tracing/audit where needed. |
