# CROSS_CHECK.md — Independent Review & Verification Plan

**Deliverable:** OpenCode sidebar for IntelliJ ("stack") — external OpenCode integration with session browser, message actions, diff panel, session to-do panel, subagent monitoring panel, using the V2 named-pipe protocol via JCEF + `CefMessageRouter`, no WebSocket on the critical path.

**Reference spec:** (the "detailed spec" the user gave me — the authoritative source). **Name/implementation may differ**; the requirement/behavior surface may not.

**How to use this document:** print it, put it beside the screen or give it to the reviewer agent, then follow it top-to-bottom. Each area lists pass criteria, exact probes, and worked failure signatures. Keep evidence with commands/outputs; a score without evidence is not a score.

**Canonical test invocation** (from the runbook `docs/OPENCODE-SIDEBAR.md`): run everything from the repo root; all commands use `./gradlew` or `./run-ide.sh`. If the runbook differs, the runbook wins — but no test may require manual IDE GUI setup beyond it.

---

## 0. Reading order (recommended)

1. Read this rubric cold (10 min).
2. Run **§2 Veto rules** — if ≥1 veto, still collect all vetoes, but stop scoring areas beyond what evidence supports.
3. Walk the **five areas** (§3–§7) in order; collect findings as one-line items with evidence.
4. Produce the report (§10) — short, specific, decisive.

---

## 1. Scoring rubric

**Base:** five areas × 17 points.

| Area | Theme | What it covers |
|---|---|---|
| **A** | Reliability & Correctness | Session lifecycle, panel behavior, streaming semantics, edge cases |
| **B** | Performance & Efficiency | Startup, latency, FPS, session scale, cost |
| **C** | Security & Compliance | Trust boundaries, permissions, no telemetry/secrets, safe file ops |
| **D** | Tests & Verification | Does the evidence actually prove the features. Real binary tests, no mocks/helpers where the spec says otherwise |
| **E** | Scope Discipline | Only the stated surfaces touched; no refactors, no launch/build/test-script changes, no provider/store changes |

**Bonus (0–15):**

| Bonus area | Points | Meaning |
|---|---|---|
| Robustness | ≤5 | The implementation holds under edge cases, concurrency, repeated actions, and malformed input, not just the happy path |
| Future-proofing | ≤5 | Clean internal seams, no hard-coded magic paths/API assumptions, backward/forward tolerance, designed for reuse |
| Documentation | ≤5 | Doesn't just cover the happy path; explains design choices, known limits, and how to diagnose problems |

**Final score** = `Sum(areas) + bonus`, then apply the veto caps from §2. If the deliverable is imperfect but complete and functional, say so. If it's broken, say precisely where. No hedging.

> Note on the spec's "global cap of 85": the five areas sum to at most 85; the bonus section is what carries a clean implementation to 100. Treat the final rubric above (`areas ≤ 85 + bonus ≤ 15 → veto caps`) as authoritative. If a stricter reading (total capped at 85 regardless) was intended, state that at the top of the report; do not silently pick one.

---

## 2. Veto rules — single triggers (any one caps the score)

| Veto | How to check | Evidence to collect |
|---|---|---|
| Core loop is a shell (submit does nothing) | Send a prompt; verify a turn starts and completes; verify session state changes | UI/session/message evidence |
| Websocket implementation on the critical path (spec explicitly says **avoid WebSocket**) | Grep new code for WebSocket/ws://; inspect the sidebar's transport; check process/browser network | Grep output + transport proof |
| No CLI E2E test (there must be ≥1 per feature, and the real binary must be invoked) | Look at tests: do they run the actual OpenCode binary, or a stub/helper? | Test list + test invocation log |
| Placeholder/partial file left in the repo | `rg` for `TODO`, `FIXME`, `NotImplemented`, `stub`, empty methods, dead buttons; check every declared UI action is wired | Match list with file:line |
| Bypassing existing IDE/OpenCode auth/permissions | Inspect for approval-bypass flags, permission-mode overrides, or direct session mutation | Code path + observed behavior |
| Auto-install/uninstall of OpenCode, or flashing consoles | Inspect launch path; grep for install/msiexec/clam; run under a recorder | Path + recording (Windows: flash guard check) |
| Destructive action without explicit user request (deleting project files, killing unrelated processes) | Run flows in a scratch project; monitor file system + process table | `inotify`/`fsmonitor` or process list diff |
| Changing session store format/extension | Diff against baseline: file addresses/format/extension must be unchanged | `git diff`/hash comparison |
| New hard dependency without huge benefit | Inspect the dependency change; ask what breaks without it | Build without it |
| Replacing JetBrains services (file editor, diff, terminal, etc.) with custom equivalents | UI inspection + code paths | Evidence of original service usage vs replacement |
| Using the existing OpenCode server instead of starting a dedicated new instance | Process tree after launch | Process evidence |
| Settings written to disk (spec: settings are UI/transient only) | Run, change settings, quit, rerun, inspect disk/state | Before/after diff |
| Changing launch logic or build scripting | Diff the launch/build paths | `git diff` + behavior check |

---

## 3. Area A — Reliability & Correctness (0–17)

### A1. Startup & process lifecycle
- [ ] Sidebar can start OpenCode after explicit user action; no auto-start on IDE boot.
- [ ] Exactly one OpenCode process tree per start; repeated start clicks do not spawn duplicates.
- [ ] Quit/stop terminates cleanly; no orphan processes remain (`ps`/Task Manager check).
- [ ] Process start failure shows an error in the UI, not a silent no-op or stack trace.
- [ ] `opencode` is resolved from `PATH`; missing binary produces a clear message (no auto-install).
- [ ] (Windows) no console window flash on start — manual check, message-mode subprocess.

**Probes**
```bash
which opencode && opencode --version
# after "start":
pgrep -fa 'opencode' | sort
# after "stop":
pgrep -fa 'opencode' || echo "no leftovers"
```

### A2. Sessions list is the real source of truth
- [ ] Sessions shown are the binary's sessions (same as `opencode`'s own session listing), not a filesystem walk or cached guess.
- [ ] New sessions appear without restart; page listing refreshes.
- [ ] Session switch does not duplicate state in the panel/to‑do panel.
- [ ] One session update produces exactly one switch (no loops, no cascades).
- [ ] Session auto-reload after close-all / switch / new-session; no IDE restart required.

**Probe:** create a session from the OpenCode CLI, confirm the sidebar updates; delete a session from the CLI, confirm the sidebar refreshes.

### A3. Named-pipe contract (V2)
- [ ] Reader accepts newline-delimited JSON events.
- [ ] Idempotent: applying the same delta twice does not duplicate text.
- [ ] Oversized payloads are skipped or truncated safely, never freeze the reader.
- [ ] A slow subscriber is dropped without blocking the sender.
- [ ] Ordering: events render in arrival order; no reordering, no loss of the final state.
- [ ] Disconnect/reconnect re-syncs state rather than silently stopping.

**Probe (Linux path):** run the binary under the plugin's transport and stream events; send a duplicate delta in a test harness and verify the UI text is unchanged; send a >64 KB payload and confirm the app stays alive.

### A4. Delete semantics
- [ ] Deleting a session removes it from disk within **3 seconds**.
- [ ] No stale cursors/caches; subsequent actions don't point at the deleted session.
- [ ] No 404 storms: after delete, requests referencing it stop; failures are handled once and clearly.
- [ ] Page/session listing is refreshed after delete.

**Probe:** delete session whose id is known; watch file mtime for deletion; monitor backend log for repeated 404s.

### A5. Fork semantics
- [ ] Fork creates an independent new session; `session.id` differs from the source.
- [ ] No history contamination: the fork starts from the correct point, no duplicate/merged history.
- [ ] The original session is untouched (id, content, continuation).
- [ ] Refreshing/listing shows both sessions.

### A6. To‑do panel
- [ ] Painted directly from the live stream — **no polling**.
- [ ] No duplicate entries when the same update is delivered twice.
- [ ] Stale skeleton entries are removed once real updates arrive.
- [ ] Empty/never-used state renders sanely.
- [ ] Works across session switch (doesn't leak previous session's items).

### A7. Subagent monitoring panel
- [ ] Each subagent gets **exactly one** card (no appending a second card for the same subagent).
- [ ] Cards default to collapsed.
- [ ] Expanding shows nested tool calls and outputs, including parallel dispatches.
- [ ] Parallel dispatches don't mix streams; each card owns its own updates.
- [ ] Cards remain correct across session switch/refresh.
- [ ] Agent/model notifications (if present) map to the right card.

### A8. Message actions (copy / edit / delete / regenerate / fork)
- [ ] Each action is wired and does what it says (no dead buttons).
- [ ] Copy returns exactly the selected content (no invisible markup).
- [ ] Edit updates the session through the same path OpenCode uses; the message on disk reflects the edit.
- [ ] Delete removes the message and does not break ordering/threading.
- [ ] Regenerate produces a new assistant message from the same point without duplicating the old one incorrectly.
- [ ] Fork matches §A5.
- [ ] Actions are only offered where valid (e.g., no regenerate on user message).

### A9. Diff panel
- [ ] Deterministic, minimal hunks: same input pair → same diff, and contiguous changes are merged where reasonable.
- [ ] Correct line numbers (old/new) for all hunks.
- [ ] Whole-file hunk when whole-file rewrite is the truthful representation.
- [ ] Every changed line appears in some hunk (no lost changes).
- [ ] Identical content shows the clean state message — not a fake or empty panel with broken chrome.
- [ ] Multi-file changes list each file with correct path; step navigation works.
- [ ] Blank diffs are replaced by the clean state (or a documented truthful message).

### A10. Tool cards (both schema forms)
- [ ] Cards render for both schema forms (argument shapes) the backend can emit.
- [ ] Error states render distinctly (not silently empty).
- [ ] Empty arguments render sanely ("no args" or similar), not never-ending spinners.
- [ ] Missing optional fields use a safe fallback.
- [ ] Very long arguments/output don't break layout (scroll/truncate).

### A11. Streaming UX (correctness half)
- [ ] No flicker/reset/jump during streaming.
- [ ] Markdown is not re-parsed per token (visual stability).
- [ ] Scroll position is preserved when the user is reading; latest token is visible when at the bottom.
- [ ] Typing indicator appears only before the first token and disappears/replaces on first text; it doesn't reappear mid-message.
- [ ] Long messages segment cleanly; no dropped tokens at chunk boundaries.

### A12. Edge cases
- [ ] Invalid path, paths with spaces, not-a-git-repo: clear error, UI survives, no crash.
- [ ] Repeated start/stop/switch cycles leave no leaks or wedged state.
- [ ] Interrupted turn (cancel) settles deterministically; the next turn works.

---

## 4. Area B — Performance & Efficiency (0–17)

Measure with wall-clock timestamps (log line timestamps + UI timestamps) and a screen recorder for visual metrics; measure on the reference machine with the documented setup.

| # | Metric | Threshold | How |
|---|---|---|---|
| B1 | Startup → first usable turn | ≤ 45 s | timestamp start action → UI ready + prompt accepted |
| B2 | First token after submit | ≤ 4 s | prompt sent → first text delta painted |
| B3 | Animation/streaming smoothness | ~60 FPS | screen recording + frame analysis |
| B4 | Input latency | ≤ 80 ms | keystroke → paint |
| B5 | Session switch (100-msg session) | ≤ 1 s | click session → transcript visible |
| B6 | Session stress (1000 msgs) | no functional break; no O(n²) behavior | open/render/scroll; watch CPU/memory growth |
| B7 | Streaming UX | no flicker/jump/reparse; latest token visible | recorder + log correlation |
| B8 | Cost/token budget | tiny ≤ $0.05 (≤2k tok), small ≤ $0.30 (≤10k), large ≤ $1.00 (≤50k), huge ≤ $3.00 (≤200k) | token/cost report from a representative run |
| B9 | Idle polling | none: CPU flat, no periodic requests | 30 s idle monitor (process CPU + backend log) |

**Probes**
```bash
# B1/B2 timestamps: tail the plugin/binary log and the stream events
# B9 idle: two snapshots 30s apart
ps -o pid,etime,%cpu,rss -p <ide-pid>,<opencode-pid>
```
- [ ] Each threshold measured at least once; place the numbers in the report (or mark "not measured" and score the affected area down).
- [ ] No busy-wait/no timers found in the new code (`rg 'Thread.sleep|setInterval|Timer'` over new files).

---

## 5. Area C — Security & Compliance (0–17)

### C1. Trust boundaries
- [ ] Uses the existing auth/permission flow; no approval bypass, no permission-mode override in code.
- [ ] Prompts/requests go only to the local backend the IDE already uses; no new external endpoints.
- [ ] All file reads/writes go through the plugin/session APIs; no raw side-channel edits.
- [ ] Destructive operations require an explicit user action and are reversible where applicable.

### C2. No installation/launch deviation
- [ ] No auto-install/uninstall/update of OpenCode; missing binary → clear error.
- [ ] PATH resolution only; no hidden flags; no console flash (Windows, message mode).
- [ ] Launch parameters match the existing documented launch path.

### C3. Secrets & telemetry
- [ ] No telemetry, analytics, or "phone home"; no external HTTP from the new code.
- [ ] No secrets/tokens logged, on disk, or in test fixtures.
- [ ] Logs use the existing logging mechanism, not a new log file system.

### C4. Dependency hygiene
- [ ] Any new dependency is justified (huge benefit), pinned, and vulnerability-free enough for a desktop plugin.
- [ ] No vendored dumps or copied framework code.

### C5. Disposal & leaks
- [ ] Resources disposed on panel close/IDE exit; no leaked threads/handles/child processes.
- [ ] Named pipe closed and reader stopped on shutdown.

**Probe**
```bash
rg -n "Telemetry|Analytics|http(s)?://|api_key|token" <new-code-dirs>
rg -n "install|msiexec|winget|choco|curl .*opencode" <new-code-dirs>
```

---

## 6. Area D — Tests & Verification (0–17)

### D1. Coverage per feature (all five)
- [ ] For each of the five features (session browser, message actions, diff panel, to‑do panel, subagent monitoring), there is **at least one automated test at the appropriate level** — pure logic unit tests for logic, OpenCode/IDE integration for platform glue, end-to-end for the acceptance flow.
- [ ] Tests run from a clean checkout with only the documented environment setup (venv, `pip install -e`, npm); no manual setup beyond the runbook.
- [ ] No internet required; no real LLM required; deterministic (same result on repeat).
- [ ] The only tested path is Linux; any Windows-only behavior is either tested manually with evidence or explicitly marked as untested.

### D2. The CLI + UI seam test
- [ ] At least one test invokes the **actual OpenCode CLI** and asserts on the UI-facing result through the real seam.
- [ ] It must not pass by calling helper classes directly; the binary must be involved.

### D3. End-to-end test
- [ ] The E2E test starts OpenCode the way the product does (the real plugin entry point, real transport), sends a real prompt, and asserts the **real plugin** was actually invoked (instrumentation, hook, or observable state) — not a mock, not a stub, not a helper-only path.
- [ ] E2E asserts the user-visible outcome, not just internal state.

### D4. Test quality
- [ ] Failure messages are diagnostic (say what was expected/seen).
- [ ] Tests assert behavior, not implementation details.
- [ ] Edge cases covered: spaces/invalid paths, identical-content diff, session delete, fork id difference, duplicate stream deltas.
- [ ] Existing tests still pass; no regressions.
- [ ] Any skipped test has a written reason.

### D5. Honest coverage
- [ ] Coverage is reported honestly (generated from the shipped code, not stale artifacts).
- [ ] Missing/old coverage files do not count; delete them or regenerate.
- [ ] The report lists exactly which features are covered by which tests (file:line).

### D6. Manual QA evidence
- [ ] Interactions that can't be automated (mouse-over activation, typing-indicator timing, tool-card expand, visual smoothness) have manual evidence: recorder timestamps, screenshots, logs.
- [ ] At least one run from a **clean checkout** and one **incremental** run.

---

## 7. Area E — Scope Discipline (0–17)

- [ ] Only the declared surfaces changed: the new sidebar/panels/actions for the stack + necessary integration code.
- [ ] No changes to OpenCode launch logic or build/test scripting.
- [ ] No session-store format/extension changes.
- [ ] No switch to other providers (Anthropic/OpenAI/etc.) or auth mechanisms.
- [ ] No unrelated refactors, renames, or dependency churn.
- [ ] Settings live in UI/memory only — nothing written to disk.
- [ ] No JetBrains service replacement (editor, diff, terminal, file tree, etc. stay as they are).
- [ ] The existing OpenCode server is not reused; a dedicated instance is started.
- [ ] New files are minimal and reviewable; diff contains no vendored dumps or generated noise.
- [ ] Every declared action in the UI is wired (no dead buttons) and every scope addition is justified by a requirement.

**Probe**
```bash
git diff --stat <base>..HEAD
git diff --name-only <base>..HEAD | sort
rg -n "TODO|FIXME|NotImplemented|placeholder" <new-code-dirs>
```

---

## 8. Bonus (0–15)

### Bonus A — Robustness (0–5)
- [ ] Handles: oversized payloads, duplicate/out-of-order deltas, malformed JSON, slow/dead subscriber, process crash mid-turn, session deleted while open, disk-full during write, file with spaces/non-ASCII.
- [ ] Concurrent operations: two actions in quick succession don't corrupt UI state; multiple sessions opened sequentially.
- [ ] Retry/backoff exists where transient failures are likely; failures degrade to a clear message, never a hang.

### Bonus B — Future-proofing (0–5)
- [ ] Transport/protocol version is checked or tolerated; unknown event types ignored safely.
- [ ] No hard-coded user paths, session ids, or locale strings; model/agent names come from the backend.
- [ ] Clean seams: panels depend on interfaces, not one-off globals; easy to add another panel/action.
- [ ] Documented assumptions where the format is not fully specified.

### Bonus C — Documentation (0–5)
- [ ] Updated the runbook/architecture docs with: how to run, how to test, how to debug the stream, known limitations.
- [ ] Explains "why" for non-obvious decisions (V2 pipe, JCEF routing, no-WebSocket).
- [ ] Includes a troubleshooting section for the common failures (no binary, no stream, deleted session, tool card schema mismatches).
- [ ] Feature manifest: exactly what was added, where, and what was intentionally not done.

---

## 9. Worked failure signatures (typical)

| # | Signature | Area | Verdict |
|---|---|---|---|
| 1 | Session deleted on disk, but sidebar keeps it and requests keep 404ing | A4 | fail delete semantics / stale cache |
| 2 | Fork produces a session whose `session.id` equals the source, or history appears twice | A5 | fail fork independence |
| 3 | The diff panel shows "no changes" when files differ, or shows a blank panel | A9 | fail diff correctness |
| 4 | Tool cards only render one argument shape; the other falls through to a spinner | A10 | fail schema coverage |
| 5 | Typing indicator appears again after tool calls (mid-message) | A11 | fail indicator contract |
| 6 | Subagent panel appends a second card for the same subagent | A7 | fail exactly-one-card |
| 7 | Streaming flickers/reparses: text resets on each chunk | B7 | fail streaming UX |
| 8 | First token 12 s; startup 70 s; 1000-msg session becomes unusable | B1/B2/B6 | performance threshold miss |
| 9 | E2E test passes calling helper classes; the binary is never spawned | D2/D3 | fail test validity |
| 10 | `git diff` shows launch/build/test scripts changed | E2 | scope violation |
| 11 | Settings written to disk; a rerun picks them up | E6 | scope violation |
| 12 | WebSocket used anywhere on the critical path | §2 | veto |
| 13 | Dead GUI actions (buttons do nothing) | A8/E | veto (placeholder) / fail |
| 14 | Auto-install/update attempted, or console flashes on Windows | C2/§2 | veto / fail compliance |

---

## 10. Report template (keep it this short)

```markdown
# Review — <deliverable>, <commit/date>

## Score
| Area | Points | Notes |
|---|---|---|
| A Reliability & Correctness | x/17 | |
| B Performance & Efficiency | x/17 | |
| C Security & Compliance | x/17 | |
| D Tests & Verification | x/17 | |
| E Scope Discipline | x/17 | |
| Bonus | x/15 | |
| **Vetoes** | n | cap applied: 75/45/25/10/VETO |
| **Total** | **x/100** (or "VETO — not scored") | |

## Vetoes (if any)
- <rule> — <evidence (file:line, command, output)>

## Findings (one line each: location | observation | expected | severity | fix)
- …

## Evidence
- tests run + results, measurements (B1–B9), manual QA artifacts, diffs reviewed.

## Assessment
<≤10 lines: what it does, what's missing, accept / accept-with-fixes / rework. Be decisive.>
```

---

## Appendix — File map to inspect first

| File | Delivers |
|---|---|
| `docs/OPENCODE-SIDEBAR.md` (runbook) | how to run/test; the canonical commands |
| new sidebar/panel code | the five features |
| transport/client code | named-pipe reader, event shaping, idempotency, disposal |
| diff/panel code | diff hunks and step navigation |
| tests (unit/integration/E2E) | the evidence for D1–D6 |
| `git diff` | scope discipline (E) |

> "It compiles" and "the unit tests pass" are not evidence of feature completeness. Look for the five features working against the **real** OpenCode binary, on Linux, from a clean checkout, with the runbook commands.
