# HiveBench — Next Plan (post-cascade, UI phase)

## Where we are

The Hive Cascade (generate → detect → repair → escalate) is complete, pushed, and
user-controllable from both the Studio UI (`/cascade`) and CLI. The learned gate
is built, wired, tested, and **off by default** by its own pre-registered rule
(3 measurements agree: no 0.8B head separates on the silent distribution). M5
(path A / 176B `/embedding`) is shelved — requires strata engineering. The
cascade serves 59–62/66 with 0 false rejects.

The Studio UI has been through 8 cleanup passes: progressive disclosure, on-theme
colors, shared nav, settings search, command palette, backend selector, sandbox
endpoint, AFK removal, model consolidation, Docker deprecation.

## Immediate next steps (ordered by impact)

### 1. Settings consolidation Step 2 — Engines reference providers by name
- **What:** Replace `eng-url`/`eng-apikey` in engine profiles with a provider
  dropdown (from Providers tab). Store `provider` name, resolve at launch time.
- **Why:** Credentials live in one place (Providers tab). Engine profiles don't
  store raw secrets.
- **Effort:** ~30 lines (HTML + JS + backend `provider` field in EngineProfile).
- **Files:** `harness/reports.py`, `harness/models.py`, `harness/app.py`.

### 2. Settings consolidation Step 3 — Chat provider → read-only display
- **What:** Remove per-conversation provider override. Chat shows loaded model
  name. Model switching via Engines/Launch only.
- **Why:** Single source of truth for model selection.
- **Effort:** ~15 lines.
- **Files:** `harness/reports.py`.

### 3. HF integration polish
- **What:** Model cards in Hub search results (description, tags, last updated).
  Download progress bar. Better empty states.
- **Why:** Unsloth-level model discovery. Current Hub tab is functional but
  sparse.
- **Effort:** ~50 lines (JS + minor backend).
- **Files:** `harness/reports.py`, `harness/models.py`.

### 4. Encoder / AI assistant at the core
- **What:** An agent that helps users navigate the app and auto-configure
  settings based on their hardware. The Studio already has a chat (Splinter/Agent)
  with tools. The idea: give the agent awareness of the app's settings, hardware
  state, and the ability to recommend/apply configurations.
- **Why:** "Select engine → select models → go" should be achievable by asking
  the assistant, not by navigating 10 tabs.
- **Design:**
  - Add a "setup wizard" flow: probe hardware → recommend backend + model +
    context → apply with one click.
  - The agent gets tools: `get_hardware()`, `list_models()`, `set_engine_profile()`,
    `launch_model()`, `run_cascade()`.
  - The Home tab's Backend selector + model picker already provide the manual path;
    the assistant provides the guided path.
- **Effort:** Large (multi-session). Start with the setup wizard (no agent needed).
- **Files:** `harness/reports.py`, `harness/app.py`, `harness/agent.py`.

### 5. Setup wizard (no-agent version of #4)
- **What:** A guided flow on the Home tab: "Detect hardware → Recommend settings →
  Apply." Probes VRAM/RAM/CPU, picks a backend + model + context, shows the
  recommendation, one-click apply.
- **Why:** Removes the "too many settings" problem for new users. The advanced
  tabs remain for power users.
- **Effort:** ~100 lines (JS + backend hardware probe, already exists in
  `/v1/server/status`).
- **Files:** `harness/reports.py`, `harness/app.py`.

### 6. More OpenCode-style features
- **Session export/import** — `/save` exists; add UI button + import.
- **File explorer sidebar** — browse local files in chat context.
- **More /commands** — `/compact`, `/clear`, `/sessions`, `/diff`.
- **Command palette improvements** — already added; could add keyboard nav.
- **Effort:** Medium, incremental.

### 7. Documentation audit
- **What:** Update all docs (CASCADE-PILOT, WHITEPAPER, README, ENGINE) to
  reflect the UI changes, cascade, sandbox, settings search, etc.
- **Effort:** Medium (mostly writing).

## What we should NOT do

- **M5 (serve restart)** — shelved. Requires strata engineering.
- **Gate retraining** — no head separates; more data won't change the conclusion.
- **Docker/VHDX setup** — deprecated; sandbox uses Docker but the VHDX flow is unused.
- **AFK mode** — shelved.

## Recommended order

1. Step 2 + 3 (settings consolidation) — small, clear, improves UX.
2. Setup wizard (no-agent version of #4/#5) — biggest UX win for new users.
3. HF integration polish (#3) — improves model discovery.
4. Documentation audit (#7) — needed before advertising.
5. Encoder/assistant (#4) — long-term, after the wizard proves the concept.
6. OpenCode features (#6) — incremental, as time allows.

## Exit criteria for "ready to advertise"

- New user can: open Studio → pick backend → pick model → Start → chat, in < 3 clicks.
- Setup wizard recommends and applies settings automatically.
- All docs updated and consistent.
- Cascade is documented as a core feature with a UI.
- Suite is green (1013+ passed, 0 failures).