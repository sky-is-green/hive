# Hive release gate — HG-1

**This is a manual gate.** It is intentionally not a CI job: CI proves code,
this proves that a person can find, understand, and safely operate the
surface. Run it on the release-candidate commit, from a clean checkout, with
no unrelated work in flight.

**Gate ID:** `HG-1` · **Applies to:** Hive 0.2.x CLI/TUI/Studio · **Status:** draft / not yet signed off

Do not tag or publish with an unresolved **FAIL**. A **BLOCKED** item needs an
owner and a date; it is not a pass.

---

## 0. Preflight and evidence

Record these in the gate log before touching the UI.

- [ ] `git status --short --branch` is clean except known local artifacts.
- [ ] `git log -1 --oneline` is the commit under test.
- [ ] `python --version`, `hive --version`, OS, and desktop/headless mode recorded.
- [ ] No serve, training run, or capture is active:
      `hive status --no-probe`; `pgrep -af 'ds41_generate|run_cascade|ui_gate_fixture'`.
- [ ] Disk and memory recorded: `df -h .`, `free -h`.
- [ ] `hive doctor --json` succeeds and records cards/RAM/backends.
- [ ] A disposable state root exists: `/tmp/hive-ui-gate` (do not use the real state).
- [ ] The tester has another terminal available for the UI fixture.
- [ ] No credentials or personal paths appear in screenshots.

Evidence folder: `artifacts/release-gate/HG-1/<date>/`

Minimum evidence:

- terminal captures or screenshots of every TUI tab and overlay;
- the `doctor`, `help --json`, and `status --json` outputs;
- Studio screenshots at 1200 px, 760 px, and narrow/mobile width;
- wheel/sdist names and SHA-256;
- one signed result row per section.

---

## 1. Automated baseline

- [ ] Full offline unit/integration suite passes on the candidate.
      `python scripts/ci_scope.py -q --durations=30`
- [ ] No new warnings/errors in the test tail.
- [ ] Build the release artifacts:
      `python -m build && python -m twine check dist/*`
- [ ] Verify the wheel contains `harness`, `hivebench`, and the CLI entry point.
- [ ] Confirm `main` is still the release branch; `hive_test` is not tagged by accident.

If a test is flaky, rerun it once and record the first failure. Do not hide it.

---

## 2. CLI — static and scriptable

Use `NO_COLOR=1` unless checking colour explicitly.

| ID | Command | Expected | Result |
|---|---|---|---|
| C1 | `hive` in a pipe / no TTY | Hive wordmark, version, running/models/stacks/backends, `get started`, `hive --help` footer. No menus or cursor control. | |
| C2 | `hive --help` | Commands include serve, tui, status, stop, logs, models, stacks, cascade, backends, setup, doctor, config, profile, controller, encoder, version, help. | |
| C3 | `hive help models list` | Same body as `hive models list --help`; description and every option present. | |
| C4 | `hive help --json \| jq .` | Valid JSON; commands and options have descriptions; aliases appear. | |
| C5 | `hive help models --json \| jq .` | `name=models`, `path="hive models"`, seven subcommands. | |
| C6 | `hive doctor` | No fail; cards show used/free VRAM; memory shows RAM and GTT (where available); exit 0. | |
| C7 | `hive doctor --json` | Same data, parseable; no ANSI. | |
| C8 | `hive config` / `--json` | Resolved paths, backends, versions; read-only. | |
| C9 | `hive status` with no serves | Friendly empty state, not an error; exit 0. | |
| C10 | `hive status --json` | `ok`, `serves`, `pruned` keys; valid JSON. | |
| C11 | `hive models list`, `stacks list`, `cascade runs`, `backends list` | Tables render with headers and no clipped essential values. | |
| C12 | `hive nonsense` | One-line usage error, exit 2, no traceback. | |
| C13 | `hive stop does-not-exist` | Clear error and exit 1; no signal to the wrong process. | |
| C14 | `hive profile scan --consent hardware --state-dir /tmp/hive-ui-gate` | Saves a local profile; prints CPU/RAM/cards; makes no settings change. | |
| C15 | `hive controller propose --state-dir /tmp/hive-ui-gate --model-gib 12 --native-ctx 32768 --ctx 16384` | Proposal only; allowlisted keys, evidence, warnings; no apply. | |
| C15b | Add `--request "How do I test this?"` to C15 | Request features appear; raw request text is absent from the controller trace. | |
| C15c | `hive controller outcome <id> accepted --metric latency_ms=42` | Outcome and metric append to the trace; proposal record is unchanged. | |
| C15d | `hive controller propose ... --request "..." --embed-request` | Explicit CPU encoder vector appears in the trace; raw text does not. | |
| C16 | `hive controller apply <id> --state-dir /tmp/hive-ui-gate` | Refused without `--yes`; no settings change. | |
| C17 | Apply with `--yes`, then `hive controller revert --yes` | Applied values restored; newly added keys are absent, not `null`; audit files exist. | |
| C18 | `hive encoder status` / `hive encoder embed "hello"` | Status does not load; embed produces a finite CPU vector of the reported dimension (local-only). | |
| C19 | `hive profile scan --consent` with no value | Clear consent error, exit 2. | |

Also check one `--json` command with `FORCE_COLOR=1`: JSON must contain **no ANSI**.

---

## 3. TUI — empty state and live state

Start the disposable fixture in terminal A:

```bash
python scripts/ui_gate_fixture.py --state /tmp/hive-ui-gate
```

It prints the exact `HIVE_STATE_DIR` and a fake endpoint. In terminal B:

```bash
HIVE_STATE_DIR=/tmp/hive-ui-gate HIVE_NO_SPLASH=1 hive tui
```

The fixture is a UI fixture, not a real serve. It must not load a model.

### 3A. Opening and labels

- [ ] Header shows `hive 0.2.0`, RAM free, visible VRAM free, GTT used, and a timestamp.
- [ ] Tabs are exactly `1 Status`, `2 Models`, `3 Stacks`, `4 Runs`.
- [ ] Footer says `enter details  l log  x stop`, then `r refresh`, `? help`, `q quit`.
- [ ] Status table headers are `NAME PID MODE UPTIME STATE`; values align under them.
- [ ] The fixture row is `ui-gate-demo`, with one `face` tier, endpoint and `up`.
- [ ] Resource values are plausible against `hive doctor`; no fake zero.

### 3B. Menu traversal

Send these exact keys and confirm the destination:

| Keys | Expected screen |
|---|---|
| `2` | Models: local GGUF table; `Enter` opens model metadata; `q` back. |
| `3` | Stacks: saved stack table; `Enter` opens full stack JSON; `q` back. |
| `4` | Runs: cascade-run table; `Enter` opens **run summary**, not a claim to open the report; full-report command is printed. |
| `?` | Key help: tabs, movement, Enter, `l`, `x` with confirmation, `r`, `q`; resource-strip meaning. |
| `q` | Leaves help/overlay; in the main screen it quits. |

Check:

- [ ] Long model shard names show distinct suffixes (`...00001` vs `...00002`).
- [ ] Overlay footer contains only actions that work in that overlay.
- [ ] `↑`/`↓`/`j`/`k` move; `PgUp`/`PgDn` scroll details and logs.
- [ ] `r` refreshes the header and active tab.
- [ ] `Tab`, `←`/`→`, and `1`–`4` switch tabs consistently.
- [ ] At `50x12` the app refuses gracefully with “terminal too small”; no traceback.
- [ ] At `80x24`, text does not overlap or lose required labels.

### 3C. Serve details, log, stop

- [ ] On Status, `Enter` shows name, PID, mode, stack, start, log path, tier endpoint, model, ctx, backend.
- [ ] `q` returns to Status.
- [ ] `l` shows the last screenful of the supervisor log; the first lines of a multi-line log are reachable by scrolling.
- [ ] `q` returns.
- [ ] `x` opens **stop <name>?** with PID and stack; it does **not** stop immediately.
- [ ] Any key other than `y`/Enter cancels; the process and record remain.
- [ ] `x`, then `y`, stops the fixture; the record disappears; message says `stopped ...`.
- [ ] The empty Status screen says `No serves are running` once, gives the start command, and the footer no longer advertises `l`/`x`.
- [ ] Press `q` in the fixture terminal if it is still alive.

### 3D. Fallbacks

- [ ] `HIVE_NO_TUI=1 hive` prints the static dashboard, not the full-screen TUI.
- [ ] `TERM=dumb hive` does not corrupt the terminal.
- [ ] `NO_COLOR=1` removes ANSI; `FORCE_COLOR=1` on a TTY restores it.
- [ ] `HIVE_NO_SPLASH=1` skips the splash; with the splash enabled, “press any key” works.
- [ ] Resize the terminal while the TUI is open; no crash and redraw remains usable.

Known non-gating TUI limits:

- The TUI does not start a serve or download/import/remove a model.
- It does not stop all serves or force-kill.
- It does not expose filter/search, tier-specific log selection, a full report viewer, or auto-refresh.
- There is no Senses tab until an encoder is wired.

---

## 4. Studio web UI

Start a model-free Studio:

```bash
python -m harness --mock --no-open --port 8765
```

- [ ] `/server` loads with no uncaught JavaScript errors.
- [ ] Top navigation shows Studio, Runs, Cascade, **Senses**, API docs.
- [ ] The active page has `aria-current="page"`; every nav item has a title.
- [ ] Top tabs are Home, Settings, Global, Cascade, More tools; each tooltip says what it changes.
- [ ] `/runs` lists bundles and opens a report.
- [ ] `/cascade` launches the form; the Senses panel says it is disabled and never fabricates an embedding.
- [ ] `POST /v1/cascade/senses` returns **503 unconfigured** while no encoder is attached; this is expected.
- [ ] Global → **Hardware profile & controller** is visible. **Scan this machine** refuses until the consent checkbox is ticked.
- [ ] After consent, scan shows the profile summary; no settings are applied by the scan.
- [ ] **Propose** shows `changes`, `evidence`, and any warnings. **Apply with confirmation** asks first; **Revert last** restores the previous values.
- [ ] The Global tab's ordinary **Save the baseline** still works and does not overwrite the controller panel state.
- [ ] `POST /v1/controller/apply` without `confirm=true` is refused.
- [ ] `/docs` matches the actual routes (the API-doc test covers this).
- [ ] At 1200 px, 760 px, and 390 px wide, the page remains usable; no horizontal scroll except intentional code/tables.
- [ ] Keyboard focus is visible; the command palette opens with `Ctrl+K`; Escape closes it.
- [ ] Empty/error states are readable and do not show raw tracebacks.
- [ ] Stop the Studio cleanly; no orphan engine child is left behind.

---

## 5. Packaging and install

- [ ] In a clean venv, install the built wheel **without** extras.
- [ ] `hive --help`, `hive config`, and `hive doctor` work; no FastAPI import required.
- [ ] `hive` uses `HIVE_HOME` for mutable state when installed outside a checkout.
- [ ] Install `[studio,eval,test]`; Studio and the test suite start.
- [ ] `pip check` passes.
- [ ] A stale source checkout cannot leak into the wheel (inspect the wheel contents).
- [ ] Re-run `hive --version` and `hive config --json`; paths are writable.

---

## 6. Release decision

- [ ] All C, TUI, Studio, and packaging items are PASS.
- [ ] Every FAIL is fixed or has an explicit owner/date and a release exception.
- [ ] Artifacts and SHA-256 recorded.
- [ ] Release notes list known limitations (read-only TUI, senses unconfigured, unimplemented training methods).
- [ ] `main` is the intended base; no tag on `hive_test`.
- [ ] If a tag is made: `git tag -a vX.Y.Z -m ...` only after explicit approval.
- [ ] No CI rerun was required solely for docs; docs-only changes did not trigger a release job.

**Sign-off**

| field | value |
|---|---|
| candidate commit | |
| version / tag | |
| tester | |
| date | |
| platform / GPU | |
| result | PASS / FAIL / BLOCKED |
| exceptions | |
| evidence folder | |

A release is not ready until a human has signed this table.
