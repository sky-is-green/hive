# The Hive CLI

`hive` is the command-line surface of the harness.  It is the same operational
layer as the Studio, without the browser: serve a stack, find it again, read its
logs, stop it, manage the local GGUF library, and check the box before a load.
Commands that return data accept `--json`; `hive help --json` emits the full
command/option catalog for scripts and agents.

The CLI is **engine-blind**.  A stack tier is a role with a model and either a
server backend or an endpoint; `hive serve` starts or adopts it, and the command
does not know how the engine works.

```bash
hive --help
hive doctor
hive stacks list
hive models list
hive serve peer-2tier --detach
hive status
hive logs peer-2tier --tier face -f
hive stop peer-2tier
```

## Install

From a checkout, the base install is serve-only:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/hive setup
.venv/bin/hive --help
```

The Studio and evaluation suite are extras:

```bash
.venv/bin/python -m pip install -e splinter
.venv/bin/python -m pip install -e ".[studio,eval,test]"
```

`hive` is the console script declared in `pyproject.toml`.  Serve-only needs
the GGUF library and a server build; FastAPI, the splinter memory system and the
suite's model downloads are not required.  An installed wheel keeps mutable
state under `HIVE_HOME` (default `~/.hive`) instead of site-packages; a source
checkout uses the repo.  `hive doctor` tells you what is missing.

## Help and the command catalog

`hive --help` is the human index.  `hive help <command>` opens the same help as
`hive <command> --help`, including nested commands:

```bash
hive help
hive help models
hive help models list
```

For an agent or a docs generator, `hive help --json` emits the live parser as
stable JSON: command paths, aliases, descriptions, options, arguments, defaults,
choices, and the `--json` behavior.  A topic narrows the same catalog:
`hive help models --json` returns only that subtree.  It is generated from the
parser itself, so it cannot drift from the human help.

## Interactive dashboard

A bare `hive` on a terminal opens a full-screen dashboard; there is also an
explicit `hive tui`.  It is a real navigation surface, not a printed page:

| key | action |
|---|---|
| `1`–`4`, `←`/`→`, `Tab` | Status, Models, Stacks, Runs |
| `↑`/`↓`, `j`/`k` | move the selection |
| `Enter` | details for the selected serve/model/stack; run summary for a cascade run |
| `l` | supervisor log (Status) |
| `x` | stop the selected serve; press `y` or Enter to confirm (Status) |
| `r` | refresh the header and active tab |
| `?` | key help |
| `q` | quit, or back out of an overlay |

It opens on a short splash with the ASCII wordmark (skip it with
`HIVE_NO_SPLASH=1`).  The Status tab shows the same records as `hive status`,
with per-tier `up` / `down`; Models, Stacks and Runs mirror their subcommands.
The dashboard is
read-mostly: details, logs and stopping are available in place; starting a
stack and downloading a model stay explicit CLI actions.  The Runs tab shows
the run summary; the full artifact stays available through
`hive cascade report <run>` (or the Studio report page).

A pipe, a captured stream, `HIVE_NO_TUI=1`, or a terminal without `curses`
gets the static dashboard instead.  `hive status`, `hive models list`, and the
rest remain scriptable and unchanged.

## Look and feel

The palette is the Studio's black and yellow throughout: `#FFDD00` for
headers, keys and selections, `#FFB703` for amber/secondary emphasis, white for
body text, grey for secondary detail, and red only for a failure.  A Hive ASCII
wordmark is printed at startup (and on the interactive splash); there is no
glyph in the middle of command output.  The static surface keeps the same rules
and uses a spinner for the slow operations (`serve` startup, `stop`, a model
import or library scan).

The dashboard header and `hive doctor` use the same live memory snapshot:
host RAM available, per-card VRAM used/free, and GTT when the driver exposes
it.  Check that line before launching a large stack; `hive doctor` also prints
the host memory and GTT totals so a tight box is obvious before a load.

Colour is automatic: it is used only on a terminal, and is disabled in a pipe,
when `NO_COLOR` is set, or with `--no-color` (accepted before or after the
command).  `FORCE_COLOR=1` forces it.  JSON output is never coloured and never
goes through the presentation layer.  `HIVE_NO_SPINNER=1` keeps the colour but
disables animation.

```bash
hive                      # interactive dashboard on a TTY
hive tui                  # the same, explicitly
hive doctor               # ✔ / ! / ✘ checks
hive status               # running serves with per-tier up/down
hive models list          # a readable library table
```

## `hive setup`

One-command onboarding: it checks the engine, fetches the portable Vulkan build
if none is installed, creates the model/stack/state directories, and prints the
next commands.

```bash
hive setup
hive setup --json
hive setup --no-fetch        # do not download anything
```

## `hive serve`

Apply one stack and keep it serving.  A stack is a JSON file under `stacks/`
(or any path to a JSON file).

```bash
# a saved stack, foreground (Ctrl-C unloads)
hive serve peer-2tier

# a saved stack, background
hive serve peer-2tier --detach

# a generated stack from a Hugging Face file (downloads on first use)
hive serve --model org/repo:model.Q4_K_M.gguf --cards 2

# a local GGUF you already have
hive serve --model /models/model.Q4_K_M.gguf --cards 1

# apply, print, unload — a smoke test
hive serve peer-2tier --once

# write the generated stack for later
hive serve --model org/repo:model.gguf --cards 2 --save two-card
```

| flag | meaning |
|---|---|
| `--cards N` | use the first N detected cards (default: all). A model that fits one card should use `--cards 1`; a layer split is capacity, not speed. |
| `--devices 0,1` | explicit HIP indices, overriding `--cards` |
| `--ctx N` | context size for a generated stack (default 8192) |
| `--backend NAME` | server build for a generated tier (default: the first installed of `rocm`, `vulkan`) |
| `--fetch` | fetch a missing backend from the manifest before applying |
| `--name NAME` | name this serve; default is the stack name |
| `--detach` | run in the background and return when the endpoints are healthy |
| `--replace` | stop a running serve with the same name first |
| `--save NAME` | write the generated stack to `stacks/NAME.json` |
| `--json` | print the applied status/record as JSON |
| `--once` | apply, print, unload, exit (mutually exclusive with `--detach`) |

A `repo:file` model is downloaded into the models library before the stack is
planned (progress on stderr; `HF_TOKEN` for gated repos).  The planner sizes a
tier from a real local file, so a remote-only stack is refused rather than
handed to an engine that would fail later.

`hive serve` is the only lifecycle owner.  It launches one process per spawned
tier, adopts an endpoint tier after a health check, and unloads everything on
SIGINT/SIGTERM.  A failed apply rolls back the tiers it started.

### Foreground and detached

A foreground serve is attached to its terminal.  `--detach` starts a child in a
new session, redirects its output to a log, waits until the stack is applied,
and returns.  Both write a small record under `harness_state/serve/<name>.json`
(override the root with `HIVE_STATE_DIR`) so `status`, `stop` and `logs` can
find the serve from a different process.  There is no daemon and no database.

A record is advisory.  `status` checks the PID **and** the command line that
wrote it; a PID that was reused is treated as stale, never killed.  A stale
record is removed by `hive status --prune` or by `hive stop`.

## `hive status`

```bash
hive status                 # table; probes each endpoint
hive status --no-probe      # PID check only
hive status --json          # records + probe results
hive status --prune         # drop records whose process is gone
```

Each running serve is printed with its PID, mode, uptime, tiers, endpoints and
`up`/`down` state.  `hive ps` is an alias.

## `hive stop`

```bash
hive stop peer-2tier        # one serve
hive stop                   # the only serve, if exactly one is running
hive stop --all             # every recorded serve
hive stop peer-2tier --force
```

SIGTERM first, with a 15-second wait; `--force` escalates to the process group
so engine children cannot be orphaned.  A stale record whose PID no longer
matches the recorded command line is removed without sending a signal.

## `hive logs`

```bash
hive logs peer-2tier                 # the serve's supervisor log
hive logs peer-2tier --tier face     # that tier's engine log
hive logs peer-2tier -n 200
hive logs peer-2tier --tier face -f  # follow until the serve stops
hive logs peer-2tier --json
```

A detached serve writes its startup output to
`harness_state/serve/<name>.log`; each spawned tier writes
`<log-dir>/llama_server_<port>.log`.  A foreground serve's supervisor output is
its terminal, so `hive logs <name>` explains that and `--tier` still works.

## `hive models`

The local GGUF library.  The directory is the configured one
(`hive models dir`), falling back to LM Studio's folder if present, then
`models/gguf/`.

```bash
hive models list
hive models search qwen --limit 8            # Hugging Face GGUF repos
hive models search                           # the popular ones
hive models files org/repo                   # the GGUFs in a repo
hive models download org/repo:model.gguf     # progress on stderr
hive models download org/repo                # only if the repo has one GGUF
hive models import /models/folder            # link a folder, no copy
hive models import /models/file.gguf --copy  # copy one file
hive models remove file.gguf --yes
hive models dir
hive models dir /data/models                 # persist the new library
```

`download` uses `HF_TOKEN` for gated or private repos.  `import` hardlinks when
possible, then symlinks, then copies; `--copy` forces a copy.  `remove` refuses
to delete without `--yes` and cannot escape the models directory.

## `hive stacks`

```bash
hive stacks list
hive stacks show peer-2tier
hive stacks validate peer-2tier
hive stacks validate stacks/my.json --json
```

`validate` checks the stack shape, that each spawned tier's backend is
installed, that the model is local or remote-fetchable, and that a `launch`
repository exists.  It never starts a process.  `--strict` treats warnings as
failures.  `--stacks-dir` names the directory holding the JSON files (default:
`<repo>/stacks`).

## `hive cascade`

The verification / repair / escalation runner, and its recorded runs.

```bash
hive cascade runs                         # recorded runs and their summary
hive cascade report strata-cascade-...    # one run's summary
hive cascade report <run> --json          # the full report
hive cascade run -- --limit 3 --tasks experiments/cascade/tasks.json
hive cascade run --help                   # the runner's own options
```

`run` is a pass-through: every argument after `run` goes to
`experiments/cascade/run_cascade.py` unchanged, including `--help`, so the
runner's full option surface stays in one place.  It writes the usual
`experiments/cascade/runs/<name>/report.json`; `runs` and `report` read that.
The runner needs its endpoint (`--scion-base`) and, for an API fallback, its
own key; neither is a Hive setting.

## `hive backends`

```bash
hive backends                  # declared, installed, and where
hive backends --json
hive backends fetch            # portable Vulkan build for this host
hive backends fetch rocm       # or a named build
hive backends fetch vulkan --tag b11869
```

Lists every declared server build (`tools/backends/<name>/llama-server`), its
accelerators, whether it is installed, and its path.  `-` means not installed.

`fetch` resolves the newest matching llama.cpp release for this platform,
downloads and verifies it with `--version`, and atomically replaces the
backend directory only after verification.  `--force` is required to replace
an existing install.  The custom ROCm/Scion builds still come from their
scripts; `fetch` is the portable path a new user needs.

## `hive ui`, `hive tools`, and `hive sandbox`

The two working surfaces share the same state:

```bash
hive ui status              # web URL + whether it is running
hive ui terminal            # the curses dashboard
hive ui web                 # Studio in the foreground
hive ui web --detach        # Studio in the background, opens a browser
hive ui web --mock          # model-free Studio fixture
```

`hive ui` never starts a second web server. `hive ui terminal` is the same
dashboard as `hive tui`; the TUI's `w` key opens a **running** Studio in the
browser and tells you to run `hive ui web` if it is not running. Press `a` in
the TUI to open pending tool approvals: `o` allows once, `s` allows the
session, `g` allows always (global), `d` denies, and `r` refreshes. The web
Studio's Global tab carries a copyable `hive ui terminal` command. The pros and
cons of each surface are in the [interface chapter](book/01-working-modes.md).

```bash
hive tools status
hive tools policy
hive tools policy guarded
hive tools grants
hive tools revoke hive:bash@1.0.0 --yes
hive tools audit --limit 20
hive tools pending --session conv-123
hive tools decide req-abc --session conv-123 --choice once
```

`tools pending` and `tools decide` talk to a running Studio
(`HIVE_STUDIO_URL`, default `http://127.0.0.1:8765`). The Studio's Global panel
shows the same pending request with **Allow once / Allow for session / Always
allow / Deny**; use whichever surface is in front of you.

The policy core is in `harness/security.py`: a side-effect call needs a reason,
headless calls are denied, session grants disappear with the session, global
grants are explicit and revocable, and arguments are hashed in the audit.
`harness/agent_hooks.py` installs a DSH `PreToolUse` hook that blocks on a
Hive approval decision; if it cannot install, agent turns are refused. Set
`HIVE_DSH_HOOK=off` only for an explicitly unguarded development run. The
Studio Global panel shows pending approvals. An external runtime selected with
`HIVE_DSH_BIN=/path/to/dsh` is preflighted; its selected profile must provide
`@deepseek-ai/dsh-hooks-claude-code` and its peer packages. If DSH cannot load
the hook, the agent refuses to start. `DSH_HOME` selects the runtime home, and
`/v1/agent/status` reports `gateway_enforced: true` only after that preflight.
The live DSH 0.2.1-rc.2 test has passed; the source-built runtime rerun is
still part of the release gate. See
[the gateway document](TOOL-GATEWAY.md).

```bash
hive sandbox status
hive sandbox policy --local-fallback approval
hive sandbox pin                 # resolve and save the image digest
hive sandbox pin --image hive-sandbox:1 --digest <64-hex>
```

`hive sandbox status` reports Docker, the pinned image, and the fallback rule.
A Docker image tag without a digest is refused. If Docker is unavailable,
`/v1/sandbox/exec` returns `409` until the caller retries with an explicit
`allow_local_fallback` and a session id; every such host execution is audited.

## `hive profile`, `hive controller`, and `hive encoder`

These are the deterministic E0 lane: a local hardware snapshot, an allowlisted
settings proposal, and a lazy text-encoder boundary. They do not load a model
unless `encoder embed` is called, and they never auto-apply settings.

```bash
# explicit consent; the snapshot stays under HIVE_STATE_DIR
hive profile scan --consent hardware
hive profile show
hive profile list

# propose: only runtime_defaults.threads / ctx_size / ngl / resident_gib
hive controller propose --model models/gguf/model.gguf --ctx 16384
hive controller propose --model-gib 12 --native-ctx 32768 --ctx 16384
hive controller propose --model-gib 12 --request "How do I test this?"
hive controller propose --model-gib 12 --request "..." --embed-request
hive controller apply <proposal-id> --yes
hive controller outcome <proposal-id> accepted --metric latency_ms=42
hive controller revert --yes
hive controller status

# inspect without loading; embed is CPU-only and local-only by default
hive encoder status
hive encoder embed "the request to classify"
```

`profile scan` needs an explicit `--consent hardware`; it reads the existing
`hive doctor` probes and does not upload anything. `controller propose` saves
an immutable recommendation under `HIVE_STATE_DIR/profiles/`; applying it
requires `--yes` (or `confirm=true` through the API), records the previous
present/absent values under `settings/controller.json`, and appends a redacted
audit record. The optional `--request` uses the deterministic feature baseline
to cap context for short work and give code/reasoning more room; only the
feature dictionary is written to `profiles/controller-traces-*.jsonl`, never
the raw request text. Add `--embed-request` to explicitly load the configured
CPU encoder and trace a vector alongside the features. `controller outcome`
appends the human/evaluation label
(`accepted`, `rejected`, `wrong`, `unknown` or `abandoned`) and optional
`key=value` metrics to the same trace; apply/revert append their lifecycle
events automatically. `revert` restores the previous values and removes keys
that did not exist before. The E0 allowlist deliberately excludes memory,
quality, launch, encoder and MCP blocks.

The Studio's **Global** tab exposes the same scan → propose → review → apply →
revert flow. `hive encoder embed` passes `--allow-download` only when asked;
without it, a missing model is an error, not a silent download.

The controller encoder has its own lifecycle settings in the `encoder` block:
`controller_idle_unload` (default on) and `controller_idle_seconds` (default
300, floor 0.1). The lazy model is dropped after that idle window; the next
request reloads it. `hive controller status` shows whether it is loaded.

## `hive doctor`

```bash
hive doctor
hive doctor --json
hive doctor --strict
```

Checks Python, the package, the models/stacks/state directories, installed
backends, the amdgpu cards (Linux sysfs), ROCm tooling, the HF token, and the
default ports.  Exit 1 on a failure; warnings are informational unless
`--strict`.

## `hive config` and `hive version`

```bash
hive config
hive config --json
hive version
```

`config` prints the resolved paths (`models_dir`, `stacks_dir`, `state_dir`,
`serve_dir`, `log_dir`), the version, the Python executable and the installed
backends.  `version` prints just the version, or JSON with the platform.

## JSON and exit codes

Every command that returns data accepts `--json`; JSON goes to stdout, progress
and diagnostics to stderr.  Exit codes are stable:

| code | meaning |
|---:|---|
| `0` | success |
| `1` | the operation failed (stack refused, model missing, endpoint down) |
| `2` | usage error (bad flag, missing argument, unknown command) |
| `130` | interrupted (Ctrl-C) |

Set `HIVE_DEBUG=1` to see a traceback for an unexpected failure; without it the
CLI prints one line and exits 1.

## Environment

| variable | meaning |
|---|---|
| `HIVE_STATE_DIR` | root for serve records (default `<repo>/harness_state`) |
| `HF_TOKEN` | Hugging Face token for gated/private downloads |
| `HIP_VISIBLE_DEVICES` | card visibility, set per tier by the stack's `pin` |
| `HARNESS_LLAMA_SERVER` | default `llama-server` path when no backend is selected |
| `HARNESS_MEMORY_GUARD` | `0` disables the pre-load VRAM/RAM check (not recommended) |
| `HIVE_DEBUG` | print tracebacks and the model-list timing line |
| `NO_COLOR` / `FORCE_COLOR` | force colour off / on (TTY detection otherwise) |
| `HIVE_NO_SPINNER` | keep colour but disable the spinner |
| `HIVE_NO_TUI` | never open the interactive dashboard; use the static one |
| `HIVE_NO_SPLASH` | skip the ASCII startup splash in the dashboard |

## Paths

| | |
|---|---|
| models | the configured library, else LM Studio's folder, else `<repo>/models/gguf` |
| stacks | `<repo>/stacks` |
| serve records | `<state_dir>/serve/<name>.json` |
| serve logs | `<state_dir>/serve/<name>.log` |
| engine logs | `<repo>/logs/llama_server_<port>.log` |
| state root | `<repo>/harness_state` (`HIVE_STATE_DIR` overrides) |

## Troubleshooting

| symptom | fix |
|---|---|
| `no saved stack at ...` | `hive stacks list`; pass a name in `--stacks-dir` or a `.json` path |
| `serve ... is already running` | `hive stop <name>`, or `hive serve ... --replace` |
| `backend 'x' is not installed` | `hive backends`, then `scripts/install_amd_engine.sh` or `--backend vulkan` |
| `status` shows `STALE` | the process is gone; `hive status --prune` |
| `logs` says the serve is foreground | use its terminal, or `hive logs <name> --tier <role>` |
| model download is slow | it is a real download; progress is on stderr, `HF_TOKEN` helps private repos |
| a failed load | `hive logs <name> --tier <role>` — the engine's startup error is in that file |

The benchmark and training runners keep their own commands in
`hivebench/README.md`; the serving/ops surface and the cascade wrapper are
here.
