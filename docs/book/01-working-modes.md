# 1. Working modes: terminal and web

[Back to the book](README.md) · [CLI reference](../CLI.md) · [Studio API](../../hivebench/API.md)

Hive has two front doors to the same harness:

- **Terminal mode** — a curses dashboard for live state, logs, inspection and
  stop controls.
- **Web mode** — the Studio, for visual setup, cascade forms, reports, the
  controller panel and long-form browsing.

They are not separate products. They read the same models, stacks, settings,
run bundles, profiles, controller state and audit. The right choice is usually
“whichever is in front of me,” but they are genuinely better at different jobs.

## Contents

- [Quick switch](#quick-switch)
- [What each surface is for](#what-each-surface-is-for)
- [Pros and cons](#pros-and-cons)
- [What is shared, and what is not](#what-is-shared-and-what-is-not)
- [Recommended workflows](#recommended-workflows)
- [Manual smoke test](#manual-smoke-test)
- [Troubleshooting](#troubleshooting)
- [Where to read next](#where-to-read-next)

## Quick switch

Run these from the same checkout and, usually, the same `HIVE_STATE_DIR`:

```bash
hive ui status                 # is the web Studio running? where?
hive ui terminal               # terminal dashboard (same as: hive tui)
hive ui web                    # Studio in this terminal, foreground
hive ui web --detach           # Studio in the background; opens a browser
hive ui web --mock             # model-free Studio fixture
```

Switch while already inside a surface:

| From | Action | Result |
|---|---|---|
| TUI | press `w` | opens a **running** Studio in the browser; tells you to run `hive ui web` if it is not running |
| TUI | press `q` | returns to the shell; then `hive ui web` |
| Studio | Global → **Working modes** → **Copy terminal command** | copies `hive ui terminal` |
| Shell | `hive ui web --detach` | starts Studio in the background and opens the browser |
| Shell | `hive ui web` | runs Studio in the foreground; `Ctrl-C` stops it |

`hive ui web` does not search for an existing Studio; use `hive ui status`
first when unsure, so you do not start a second process on the same port. The
TUI deliberately cannot launch a web server from inside curses: a blocking
server would freeze the dashboard.

## What each surface is for

### Terminal mode

The TUI is a compact operations panel. It is for the questions you ask while
something is running:

- Is the serve alive, and on which endpoint?
- Which stack and model are loaded?
- How much RAM, VRAM and GTT are free?
- What is in the supervisor log?
- Stop this serve, with confirmation.
- Browse local models, saved stacks and recorded runs.

It is keyboard-first, works over SSH, and does not need a browser or a web
server. It is also the quickest way to check a machine before launching a large
stack.

### Web mode

The Studio is the control surface. It is for work that benefits from space,
forms, tables and graphs:

- Configure providers, engines and the model library.
- Build and launch stacks.
- Run the cascade and inspect reports.
- Attach sensory inputs and see the route (when an encoder is attached).
- Scan hardware, review controller evidence, apply and revert settings.
- Read long run reports and API docs.
- Copy the terminal command when you want to switch back.

The Studio is the same state under a different renderer; it is not a cloud
service and it does not upload the hardware profile or prompt text by default.

## Pros and cons

| | Terminal (`hive ui terminal`) | Web Studio (`hive ui web`) |
|---|---|---|
| **Startup cost** | Very low; curses only | FastAPI + browser + page assets |
| **Remote use** | Natural over SSH; no port forwarding needed for the UI | Needs a browser path to `127.0.0.1:8765`, usually an SSH tunnel |
| **Visual density** | Compact, text-first | Forms, cards, tables, reports |
| **Best for** | Live health, logs, quick inspection, stop controls | Setup, cascade forms, controller review, reports, API docs |
| **Approval dialogs** | Not implemented for tool approvals yet | Shared broker planned; current gateway is CLI/API-only |
| **Cascade** | Can launch the runner from the shell; no visual report | Full launch form and report pages |
| **Senses** | Not exposed | Senses panel exists; returns `503` until an encoder is attached |
| **Controller** | `hive controller ...` and `hive profile ...` | Global-tab panel with evidence, apply and revert |
| **Accessibility** | Depends on the terminal and screen reader | Browser semantics; `aria-current` on navigation |
| **Failure mode** | Terminal too small or no TTY → static dashboard | Browser/server unavailable → use the TUI or CLI |
| **Security** | No browser attack surface; terminal process sees local state | Local web app; set `HARNESS_TOKEN` if exposing it beyond localhost |
| **Concurrent edits** | Do not edit settings while another surface is saving | Same caution; both write the same state files |

Neither surface is “the real one.” The CLI remains the most scriptable, the
TUI the lightest, and the Studio the most complete for configuration and
review.

## What is shared, and what is not

Shared:

- model library and stack files;
- Studio settings and saved engine profiles;
- serve records and run bundles;
- controller profiles, recommendations and traces;
- tool policy and grants;
- conversations and durable agent sessions.

The web UI starts with the same `HIVE_STATE_DIR` (or the resolved Hive state
root) as the CLI. If you deliberately point one surface at another root, it is
a different Hive for state purposes even though the code is the same.

Not automatically shared:

- **The running server process.** `hive ui web` starts a Studio; the TUI reads
  serve records but does not own a Studio lifecycle.
- **Open browser tabs.** The TUI can open the running Studio, but it cannot
  know which tab you were reading.
- **Terminal state.** The web Studio can copy the command to start the TUI,
  but a web page cannot spawn a local terminal in your shell.
- **In-memory session grants.** They die with the process that owns them; this
  is intentional.

Practical rule: treat **settings, controller apply/revert and tool policy** as
single-writer operations. Open the surface that owns the change, make it, and
close or refresh the other before editing the same block.

## Recommended workflows

### Local desktop, setup or report review

Use the web Studio. Keep the TUI available for a quick `hive ui status` or a
Ctrl-C-friendly log view.

### Remote server over SSH

Use the terminal for live state and stops. If you need the Studio, forward the
port over SSH:

```bash
ssh -L 8765:127.0.0.1:8765 you@server
# on the server:
hive ui web --no-open
# on your machine:
open http://127.0.0.1:8765
```

If the Studio can be reached by anything other than localhost, set
`HARNESS_TOKEN` and use the token field in the browser. Do not expose an
unauthenticated agent-capable Studio to a network.

### Headless batch or GPU box

Use `hive ui terminal` for resource checks, `hive tools` for policy, and
`hive controller` for deterministic proposals. The TUI is designed to stay
lightweight while a large model is serving.

### Switching mid-task

1. Save the current settings form.
2. Run `hive ui status` to see whether the Studio is already up.
3. Start the other surface.
4. Use the Global **Working modes** card (web) or `w` (TUI) to move back.

The CLI can always answer the state question even when both interactive
surfaces are closed.

## Manual smoke test

This is safe with a disposable state root:

```bash
export HIVE_STATE_DIR=/tmp/hive-mode-gate
hive ui status
hive ui web --mock --detach
hive ui status
```

Then:

- Open the printed URL.
- Open **Global** and find **Working modes**.
- Press **Copy terminal command**; `hive ui terminal` is copied.
- Return to the shell and run the copied command; the TUI opens.
- Press `w`; the running Studio opens in the browser.
- Press `q` in the TUI; the shell returns.
- Stop the Studio with its recorded `--stop` path or close the foreground
  process.

For a foreground check:

```bash
hive ui web --mock --no-open
```

Stop it with `Ctrl-C`; no serve or model process is started by `--mock`.

## Troubleshooting

**`hive ui terminal` exits immediately**

There is no interactive TTY. Run it from a real terminal, or use
`HIVE_NO_TUI=1 hive` for the static dashboard.

**`hive ui web` says the port is in use**

Another Studio is already running. Check `hive ui status`; use its URL instead
of starting a second copy.

**The TUI `w` key says “web UI not running”**

Start it in a shell with `hive ui web` or `hive ui web --detach`.

**The web page works but the terminal has different state**

They may be using different `HIVE_STATE_DIR`, `HIVE_HOME` or a different
checkout. Run `hive config --json` and compare `state_dir`, `models_dir` and
`serve_dir`.

**The browser cannot reach a remote Studio**

Bind explicitly only when you mean it (`--host 0.0.0.0` is a network decision),
prefer an SSH tunnel, and set `HARNESS_TOKEN` for any non-local exposure.

**Settings changed in one surface and not the other**

Settings are stored on save. Refresh the other surface; unsaved form values are
not shared.

## Where to read next

- [CLI reference](../CLI.md) — every command, `--json` behavior, exit codes.
- [Studio API map](../../hivebench/API.md) — the HTTP surface behind the web UI.
- [State store](../STATE-STORE.md) — where the shared state lives.
- [Release gate](../RELEASE-GATE.md) — the manual checks that include both
  surfaces.
- [The Hive Book](README.md) — the rest of the map.
