# The Hive CLI

`hive` is the command-line surface of the harness.  It is the same operational
layer as the Studio, without the browser: serve a stack, find it again, read its
logs, stop it, manage the local GGUF library, and check the box before a load.
Every command has a `--json` form for scripts.

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

From a checkout:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/hive --help
```

`hive` is the console script declared in `pyproject.toml`.  For serving only,
the GGUF library and a server build are enough; the evaluation suite's model
downloads are not required.  `hive doctor` tells you what is missing.

## Look and feel

A bare `hive` is the front door, not a usage error: it prints the Hive mark, the
current serve/model/stack counts, and the next commands.  The rest of the CLI
uses the same small language — the amber hexagon, green/amber/red for
pass/warn/fail, hairline table rules, and a spinner for the slow operations
(`serve` startup, `stop`, a model import or library scan).

Colour is automatic: it is used only on a terminal, and is disabled in a pipe,
when `NO_COLOR` is set, or with `--no-color` (accepted before or after the
command).  `FORCE_COLOR=1` forces it.  JSON output is never coloured and never
goes through the presentation layer.  `HIVE_NO_SPINNER=1` keeps the colour but
disables animation.

```bash
hive                      # dashboard
hive doctor               # ✔ / ! / ✘ checks
hive status               # running serves with per-tier up/down
hive models list          # a readable library table
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
hive backends
hive backends --json
```

Lists every declared server build (`tools/backends/<name>/llama-server`), its
accelerators, whether it is installed, and its path.  `-` means not installed.

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
