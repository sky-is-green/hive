# Pre-approval review

**Status:** main frozen at `008d758`; `rust-migration` is the review candidate.
No merge until the owner signs off.

**Exhaustive gate (2026-11-23, candidate `dbd8be5`):** `scripts/full-gate.sh` passed **24/24**
steps on the post-walkthrough tree: Rust fmt/build/clippy/tests, Python
unit/integration, non-live conformance, all live daemon and llama tests,
install/uninstall, and release manifest/signing. See
`/tmp/opencode/hive-gate/20261123T183103Z/` and
[UX-WALKTHROUGH.md](UX-WALKTHROUGH.md).

The goal is maximum performance with maximum usability, not Rust everywhere.
Each component is reviewed on four axes: measured benefit, operational
complexity, failure modes, and user-visible behavior. A component can stay
Python, stay Rust, or become a small C++/C hotspot when the measurement
justifies it.

## 1. Exhaustive test gate

Run `scripts/full-gate.sh` on the candidate. It records:

- Rust fmt, clippy, workspace tests.
- Python unit + integration.
- Conformance (non-live), then each live test individually to avoid resource
  spikes on a busy host.
- Install/uninstall layout.
- Release manifest, checksums, SBOM, signing/verification.
- Encoder and memory parity/benchmark smoke.

Evidence lives under `/tmp/opencode/hive-gate/<timestamp>/` by default.

## 2. UX walkthrough

Order:

1. Clean install of the Rust runtime and the Python `hive-py` reference.
2. First run: `hive`, `hive doctor`, `hive config`, `hive status`.
3. Model discovery/download and `hive models`.
4. Stack create/validate/apply/unload, then `hive logs`/`stop`.
5. Daemon/service start, `/health`, `/ready`, `/metrics`, restart/swap.
6. Studio console: first load, status, models, stacks, chat, memory inspector,
   sandbox/tool panels, narrow viewport.
7. Failure states: missing model, missing sidecar, bad token, sandbox denial,
   killed engine, full disk, cancel a stream.
8. Upgrade and rollback: reinstall Rust, `--uninstall`, `hive-py`.

Every step records expected vs actual, a screenshot or JSON, and a verdict.
Usability regressions are merge blockers even when the tests pass.

## 3. Technology fit / revert audit

No component is kept merely because it was ported. Criteria:

- **Keep Rust** when it wins startup/RSS/process safety/concurrency and the
  code is not disproportionately complex.
- **Keep Python** for research, evaluation, browser/Playwright, provider
  breadth, and surfaces whose maintenance cost exceeds their runtime benefit.
- **C++/C hotspot** only with a measured >2x win in a narrow, stable boundary
  (FFI or subprocess), plus a safe fallback. llama.cpp/ONNX Runtime already
  bring C++ internally; wrapping them again is not automatically a win.
- **Revert** when the Rust port has no measurable user benefit and adds a
  failure mode or maintenance burden.

| Component | Current | Evidence | Review |
|---|---|---|---|
| CLI, state, supervisor | Rust | 0.9 ms startup, 14.5 MB RSS, zero orphaned engines | keep |
| Daemon/API/proxy | Rust | single binary, low idle RSS, sidecar boundary | keep unless UX shows proxy gaps |
| Encoder | Rust ONNX | >0.99999 cosine parity, torch-free, 166 MB peak | keep |
| Memory core | Rust | 1.12–1.19x cached Python, native default | measure again on real conversations; Python fallback stays |
| Sandbox | Rust helper | real Landlock/seccomp, real cgroup budgets | keep |
| WASM plugins | Rust + wasmtime | fuel/memory limits, signed manifests | keep (wasmtime is Rust) |
| TUI | Rust | four tabs; in-place log view, stop confirmation, Studio approvals/web shortcuts | keep; advanced overlays remain Studio |
| Studio/eval/cascade/tools/browser | Python | breadth and research coupling | keep |
| Report rendering | Python | report HTML still proxied | keep until a measured UX need |
| Memory hot loops | Rust scalar | no SIMD benchmark yet | C++/SIMD candidate only if measured |

Open candidates for a small revert are recorded as they are found; none are
assumed yet.

The exhaustive gate also fixed two real usability/performance bugs:

- CPU-only launches (`ngl=0`) were charged against VRAM, so a busy GPU made
  every live CPU launch fail.
- `serve --model` generated `ngl=99` even for CPU backends; generated stacks
  now use `ngl=0` when there is no GPU or the backend is CPU.

The caveat pass and UX walkthrough are complete. The remaining approval step
is the owner sign-off, then the `main` merge gate. The walkthrough fixed the
clean-install manifest/shadow issue, Studio lifecycle routing, canonical chat
path, Bearer auth, native-encoder onboarding, sandbox workspace fallback, and
the responsive/deep-link console; details and evidence are in
[UX-WALKTHROUGH.md](UX-WALKTHROUGH.md).
