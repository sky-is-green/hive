# HiveBench

[![CI](https://github.com/sky-is-green/hivebench/actions/workflows/ci.yml/badge.svg)](https://github.com/sky-is-green/hivebench/actions/workflows/ci.yml)

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)](https://github.com/sky-is-green/hivebench)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)](https://github.com/sky-is-green/hivebench)
[![pytest](https://img.shields.io/badge/pytest-0A9EDC?style=flat&logo=pytest&logoColor=white)](https://github.com/sky-is-green/hivebench)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://github.com/sky-is-green/hivebench)
[![Top language](https://img.shields.io/github/languages/top/sky-is-green/hivebench?style=flat&logo=python&logoColor=white)](https://github.com/sky-is-green/hivebench)
[![Last commit](https://img.shields.io/github/last-commit/sky-is-green/hivebench/main?style=flat&label=Last%20commit&logo=git)](https://github.com/sky-is-green/hivebench/commits/main)
[![Repo size](https://img.shields.io/github/repo-size/sky-is-green/hivebench?style=flat&label=Repo%20size)](https://github.com/sky-is-green/hivebench)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Evaluation suite + Studio sidecar harness for the [Splinter Memory](https://github.com/sky-is-green/splinter-memory) system.

**Status: work in progress.** Carved out of splinter-memory on 2026-09-12 as its
own repository. The offline suite is green (1010 unit + integration, verified
Oct 19); the live batteries and Studio are under active development — expect
moving parts here.

## What it is

HiveBench is a **local-first AI harness**: a Studio console that runs, measures,
and self-corrects models entirely on your own hardware — plus the deterministic
evaluation suite that proves the measurements.

Two halves, one repo:

- **HiveBench Studio** (`python -m harness`) — the console. Chat with your
  local models, manage engines/providers/models, run the cascade, inspect
  retrieval, train, and browse results. ~40 REST endpoints under `/v1/`,
  documented live at `/docs`.
- **The Hive Cascade** — the answer-quality loop: **generate → detect → repair →
  escalate**. Deterministic verification, one greedy self-repair round, human
  gate, optional API escalation. Gold-free at runtime, fully user-controlled.
- **The evaluation suite** — 1010 offline tests that falsify claims instead of
  asserting them. No LLM calls, no API keys, ~60 seconds.

## What makes it different

Most harnesses tell you *how a model performed*. HiveBench tells you **whether
the context you feed the model is the reason it works**, and it tells you
**whether the answer it produced is actually right** — deterministically,
offline, and replayably.

| | Typical harness | HiveBench |
|---|---|---|
| **Correctness** | Judge by another LLM (circular, biased) | Deterministic verifiers: arithmetic consistency, the task's own tests, canonical-form checks — no judge model in the evidence path |
| **Self-correction** | Manual re-prompting | Built-in **cascade**: detect → repair → escalate, with a human gate before any paid API call |
| **Where the answer key lives** | Sometimes in the prompt path | **Nowhere.** The runtime reads a gold-stripped `runtime_spec`; the checker never sees the answer key |
| **Evidence** | Vague scores | Every number is reproduced by a command; falsifiable predictions ship as executable tests with measured PASS/FAIL |
| **Hardware** | Cloud GPU assumed | Runs on CPU or whatever GPU you have; the drones stay on CPU; offline CI in ~60 s |
| **Backends** | One vendor | Strata (local 176B), llama.cpp (any GGUF), or any OpenAI-compatible API — one selector, three backends |
| **Your data** | Sent to a provider | Stays on your machine unless you explicitly enable API escalation |

### The hive architecture

HiveBench is designed as a **hive** — models with different roles cooperating:

```
User → Encoder (small model, app-aware) → Studio backend → Worker (176B / GGUF) → Cascade (detect/repair/escalate)
```

- **Encoder** — a small model (0.8B–2B) acting as app assistant: it can probe
  your hardware, list your models, launch a server, and run a cascade. The
  agent loop (dsh) already provides the tool-calling; the encoder just needs
  app-aware tools.
- **Worker** — does the actual generation (176B strata, or any local GGUF).
- **Cascade** — the quality layer between them.

## What it aims to achieve

Most evaluation harnesses tell you how a model performs in a sandbox. HiveBench
tells you *whether the context you feed the model is the reason it works*, and
it does it deterministically, offline, and replayably:

- **Falsifiable, not vibes.** The white paper's P1-P11 predictions ship as
  executable tests with measured PASS/FAIL verdicts (splinter repo,
  `SPLINTER-WHITE-PAPER.md` section 8). Every number in the splinter README is
  reproduced by a command in this repo.
- **No LLM-as-judge circularity in the evidence path.** The deterministic
  diagnostics score fact presence against fixture ground truth, stated-facts
  recall, first-mention exclusion, hedge filtering. **The Splinter auditor**, an
  asynchronous ground-truth layer that labels, after each turn, whether the
  assembled context was actually sufficient for the query, corroborates that
  evidence; because it shares the served model's biases, it never constitutes
  it (white paper section 9, Threat 1).
- **The full test suite runs offline in ~30 seconds**: no LLM calls and no API
  keys; CI-friendly via `--mock`. (Running the system *live* does require a
  local model backend, LM Studio / llama.cpp, which on most rigs means a GPU;
  the drones themselves stay on CPU.)
- **Paired head-to-head A/B** (`hivebench-ab`): the same turns, the same model,
  splinter-curated context vs the naive FIFO window, both answers scored
  deterministically (fixture-fact presence + context fidelity), with both
  arms' stores replaying identical history so the comparison isolates
  selection. The scoring path is unit-tested; interim live results are
  recorded per run under `runs/`.
- **Built for long evidence runs.** Checkpointed, resumable live runs survive
  crashes and reboots:

  ```powershell
  .\.venv\Scripts\python -m experiments.paired_ab --live --model prism-ml/bonsai-27b --max-turns 45 --fifo-budget 1500 --checkpoint-every 2 --output runs/paired_ab.json
  # killed mid-run? relaunch with --resume runs/paired_ab_trunc-style checkpoint,
  # or let tools/resume_evidence.ps1 loop until the final report exists.
  ```

- **Honest by design.** The suite surfaced its own failures first; the
  measurement fixes that made PES trustworthy (latency floor, stated-facts
  reframe, hedge poisoning) are documented in the paper's threats section
  (section 9), not hidden.

## Install

The checkouts live side by side:

```
~/Desktop/work/splinter-memory   # the system
~/Desktop/work/hivebench       # this repo
```

Windows (PowerShell):

```powershell
git clone https://github.com/sky-is-green/splinter-memory.git
git clone https://github.com/sky-is-green/hivebench.git
cd splinter-memory
python -m venv .venv
.\.venv\Scripts\python -m pip install -e .   # the system: drones, cortex, retention, backends
cd ..\hivebench
python -m venv .venv
.\.venv\Scripts\python -m pip install -e .   # suite + studio (console scripts)
```

Linux/macOS: same in each checkout — `python3 -m venv .venv && .venv/bin/python -m pip install -e .`

Flat import names are preserved (`experiments.*`, `testing.*`, `tests.*`,
`harness.*`, `deepseek_harness`); the system under test resolves as `splinter` /
`cortex` / ... from the sibling checkout. Override its location with
`$SPLINTER_HOME`.

## Run the studio (HiveBench Studio)

Two commands are the whole story, run from this checkout: `--setup` copies
`providers.example.json` → `providers.local.json` if missing, probes for a reachable
backend (LM Studio on `:1234`, or auto-starts the local `llama-server` from
`models/gguf`), and prints the next step. The studio serves the splinter over a
FastAPI API; the endpoint contract lives in `harness/app.py`, and the
splinter repo's `docs/INTEGRATE.md` shows how to point external clients at it.

```powershell
.\.venv\Scripts\python -m harness --setup   # copies providers config, probes backend, warms the drone
.\.venv\Scripts\python -m harness           # studio UI on http://127.0.0.1:8765
```

## Sandbox (optional, Docker-backed)

The Studio ships a `/v1/sandbox/exec` endpoint: run a command in an isolated
container when the Docker daemon is up (network disabled, 512 MiB / 1 CPU,
hard timeout), and fall back to a local subprocess otherwise. It backs the
cascade's `code_exec` detector and gives the agent a safe execution surface.
Check availability with `GET /v1/sandbox/status`.

```powershell
curl -X POST http://127.0.0.1:8765/v1/sandbox/exec -H "Content-Type: application/json" -d '{"command":"python -c \"print(42)\"", "timeout":10}'
```

## Hive Cascade (core feature)

Hive Cascade is the local-first answer-quality loop built into HiveBench: **generate → detect → repair → escalate**. It runs entirely on your machine — no answer key is ever exposed to the runtime (the verifier reads a gold-stripped `runtime_spec`).

- **Detect** — deterministic verifier (`consistency` + the task's own tests via `code_exec`), reported as findings.
- **Repair** — one greedy local re-attempt with the finding quoted; truncated/byte-identical attempts are rejected; cleared repairs are re-framed by `canonical()`.
- **Escalate** — human gate (first trigger verbose, later compact) and an optional API fallback.
- **Learned gate** (opt-in) — `intern-0.8b` readout as a third trigger, blind-spot only; off by default by its own pre-registered rule.

Control it from the **Studio UI** at `/cascade` (every runner knob exposed) or the **CLI**:

```powershell
.venv\Scripts\python -m experiments.cascade.run_cascade `
  --tasks experiments/cascade/tasks-bench.json `
  --judge deterministic --repair auto `
  --gate intern-0.8b          # optional third trigger; off by default
```

Every ladder-resolved item is harvested to `escalation.jsonl` next to the report (training-side only), feeding repair SFT and local-tier distillation. See `docs/CASCADE-PILOT.md` and `experiments/cascade/LEARNED-GATE-PLAN.md`.

## Console & UI (HiveBench Studio)

The Studio (`python -m harness`) is the control surface for everything. It serves
a single hive theme (yellow/black) across every page.

### Pages

| Page | What it does |
|---|---|
| `/` → `/runs` | Run bundle index |
| `/server` | The console: chat, models, engines, providers, hub, library, inspector, training, settings |
| `/cascade` | Hive Cascade launch console — every runner knob, advanced collapsed |
| `/runs`, `/view/{run}` | Browse and inspect run bundles (rendered reports) |
| `/stacks` | Stack management (multi-tier model stacks) |
| `/docs` | Live OpenAPI docs for all ~40 `/v1/` endpoints |

### Console features

- **Home tab** — orientation, quick links, and the **backend selector**
  (Strata / Llama.cpp / API — one source of truth for the Studio and Cascade).
- **Settings tab** — a searchable settings index: 32 settings across 9 nested
  categories, each linking to where it lives.
- **Command palette** (Ctrl+K) — every console command, searchable. Ten commands
  ship today: `help`, `new`, `save`, `model`, `mode`, `provider`, `engine`,
  `bench`, `train`, `status`.
- **Model library** — search Hugging Face for GGUF repos, list files, download
  with progress, manage the local folder.
- **Engines** — reusable profiles (model, context, GPU layers, sampling, provider
  reference) instead of re-typing launch settings each time.
- **Provider references** — engine profiles point at a provider by name; raw
  credentials live in exactly one place.
- **Sandbox** (`/v1/sandbox/exec`) — Docker-backed code execution (network off,
  512 MiB / 1 CPU, hard timeout) with a local-subprocess fallback.
- **Escalation lane** — every ladder-resolved item is harvested to
  `escalation.jsonl` next to the report: task, draft, findings, repair, and the
  resolution. Training-side only; the runtime never reads it.
- **Dev server** — run it on a port you and your agent can both open, and
  iterate on the UI together.

### API surface (~40 endpoints under `/v1/`)

| Area | Endpoints |
|---|---|
| Cascade | `/v1/cascade/run`, `/runs`, `/report/{run}`, `/registry`, `/roles`, `/paths`, `/plan`, `/frontier`, `/validate` |
| Models | `/v1/models/local`, `/v1/models/local/path`, `/v1/models/linux`, `/v1/models/hub`, `/v1/models/hub/files/{repo}`, `/v1/models/hub/download`, `/v1/models/hub/downloads` |
| Servers | `/v1/server/start`, `/stop`, `/unload`, `/status`, `/v1/server/metrics` |
| Engines | `/v1/engines`, `/v1/engines/preset`, `/v1/engines/ab/bench` |
| Providers | `/v1/provider/config` |
| Agent | `/v1/agent/stream`, `/status`, `/cancel`, `/v1/agent-presets/*` |
| Commands | `/v1/commands`, `/v1/commands/run` |
| Sandbox | `/v1/sandbox/exec`, `/v1/sandbox/status` |
| Runs | `/v1/runs`, `/v1/report/{run}`, `/v1/protocol/run` |
| Training | `/v1/training/engines`, `/launch`, `/status/{run}`, `/report/{run}` |
| Conversation | `/v1/splinter/*` (turn, stream, reset, mode), `/v1/openai/*` |

Live, always-current OpenAPI docs: `http://127.0.0.1:8765/docs`.

## Running the suite

Offline, no LLM required (CI runs exactly this):

```
pytest tests/unit tests/integration
# or by group:
python tests/run_hive_tests.py --group speed|intelligence|skills|maximum
```

| Group | Measures |
|---|---|
| `speed` | latency / PES thresholds |
| `intelligence` | retrieval / assembly quality |
| `skills` | pipeline / backends |
| `maximum` | full suite (default) |

## Try it live

The live benchmark talks to an OpenAI-compatible backend (e.g. LM Studio on
`localhost:1234`). A quick resumable iteration run, from this checkout:

```powershell
.\.venv\Scripts\python -m experiments.generate_data --live --no-thinking --confidence off --max-convs 3 --max-turns 10
```

## Console scripts

| Script | Entry point |
|---|---|
| `hivebench` | `experiments.generate_data:main` |
| `hivebench-protocol` | `experiments.run_p1_p10:main` |
| `hivebench-probe` | `experiments.model_probe:main` |
| `hivebench-diagnostic` | `experiments.retrieval_diagnostic:main` |
| `hivebench-gate-ab` | `experiments.confirmation_gate_ab:main` |
| `hivebench-compare` | `experiments.run_compare:main` |
| `hivebench-ab` | `experiments.paired_ab:main` |
| `hivebench-harness` | `harness.__main__:main` |

## Tree

| Path | What it is |
|---|---|
| `experiments/` | data generation, probes, A/B batteries, finetune runs (console scripts above) |
| `testing/` | ablation / AB test tooling |
| `tests/` | unit + integration suite for the splinter system — run from here |
| `harness/` | HiveBench Studio sidecar (`python -m harness`) |
| `vendor/deepseek_harness/` | vendored dsh Python SDK (agent bridge; not an installed package) |

## Where we're going

- **S4/S5** — Studio provider row → sidecar (recall battery done; provider row
  pending UI confirm); opencode provider config → sidecar with conversation id
  = project name.
- Live batteries stay live-gated in CI (skipped without a backend); threshold
  benchmarks stay out of CI on purpose — shared runners flake them.

The system under test, its white paper, and the measured-outcome table behind
every claim: the [Splinter Memory](https://github.com/sky-is-green/splinter-memory) repo.

## License

MIT. See [`LICENSE`](LICENSE). Copyright (c) 2026 HiveBench Studio contributors.
