# The Hive Book

A human-readable guide to how Hive works, why it is built this way, and how to
operate it without having to reverse-engineer the code.

This is a **living book**. Each chapter starts as a working map, then gains
measured examples and failure notes. The first complete chapter is
[Working modes: terminal and web](01-working-modes.md); the rest link to the
existing design and measurement documents until their narrative chapters are
written.

## Contents

- [How to read this book](#how-to-read-this-book)
- [1. Working modes: terminal and web](01-working-modes.md)
- [2. Install, paths and first run](#2-install-paths-and-first-run)
- [3. Models, engines and backends](#3-models-engines-and-backends)
- [4. Serve and stacks](#4-serve-and-stacks)
- [5. Chat, memory and the encoder](#5-chat-memory-and-the-encoder)
- [6. The Hive Cascade](#6-the-hive-cascade)
- [7. Senses](#7-senses)
- [8. The harness controller](#8-the-harness-controller)
- [9. State, security and the tool gateway](#9-state-security-and-the-tool-gateway)
- [10. Evidence, evaluation and release](#10-evidence-evaluation-and-release)
- [Appendix: command and file index](#appendix-command-and-file-index)

## How to read this book

Hive is easiest to understand as four layers:

```text
interfaces        terminal / web Studio / API
      │
orchestration     chat, cascade, controller, tools
      │
engines           llama.cpp / Strata / DeepInfer / endpoints
      │
evidence          benchmarks, traces, audit, release gates
```

Read the interface chapter first if you are operating Hive. Read the evidence
and security chapters before you trust it with a long-running job. Read the
research chapters only when you are changing the model or measurement lane.

**Status legend**

- **Complete** — written for this book.
- **Existing** — the linked document is authoritative today.
- **Planned** — the chapter is stubbed in this index and links to the nearest
  current material.

## 1. Working modes: terminal and web

**Status: complete** · [Read the chapter](01-working-modes.md)

What each surface is for, the honest pros and cons, how to switch, what state
they share, and a manual smoke test.

## 2. Install, paths and first run

**Status: existing** · [CLI reference](../CLI.md) · [Install and Studio](../../hivebench/README.md#install)

The short version:

```bash
python -m pip install -e ".[studio,eval,test]"
hive setup
hive doctor
```

`HIVE_HOME` holds models, stacks and backends; `HIVE_STATE_DIR` holds mutable
state, conversations, profiles and audit. The source checkout uses its own
`harness_state/` unless the environment says otherwise.

## 3. Models, engines and backends

**Status: existing** · [Engine notes](../ENGINE.md) · [Engine options](../ENGINE-OPTIONS.md) · [Serve on AMD](../SERVE-AMD.md)

Models are GGUF files in the local library. Backends are the server builds under
`tools/backends/`; the stack schema names a role, a model, a context and a
backend, not a hardcoded model size. `hive doctor` is the first check; the
Studio's Launch console exposes the same choices.

## 4. Serve and stacks

**Status: existing** · [CLI: serve and stacks](../CLI.md#hive-serve) · [CLI: stacks](../CLI.md#hive-stacks)

A stack is an ordered set of tiers. `hive serve` starts or adopts an engine,
records the process, and lets the CLI find it again. The TUI reads those
records; the Studio owns the same model and stack registry.

## 5. Chat, memory and the encoder

**Status: existing** · [Encoder and controller](../ENCODER-CONTROLLER.md) · [Encoder notes](../../splinter/docs/ENCODER-NOTES.md)

The chat can run in curated mode (Splinter selects context) or agent mode
(DSH tools). Memory persists per conversation; the encoder is the small model
that ranks, dedups and detects drift. The controller encoder is a separate,
lazy CPU adapter used for request features and traces; it does not change the
serving engine.

## 6. The Hive Cascade

**Status: existing** · [Cascade overview](../../cascade/README.md) · [Harness controller](../../cascade/HARNESS-CONTROLLER.md) · [Cascade results](../AB-RESULTS.md)

Generate → detect → repair → escalate. A checker can override a model's
opinion; a judge is consulted only where no executable check exists. The
cascade is exposed through `/cascade`, the CLI runner, and the `/v1/cascade/*`
API.

## 7. Senses

**Status: planned** · [Senses contract](../../harness/cascade/senses.py) · [Cascade page](../../hivebench/README.md#the-quality-loop--the-hive-cascade)

Senses is an ingestion contract, not a fake answer path: image, audio, video
and text enter with a content id, modality and storage record. The API returns
`503` until a real multimodal encoder is attached. Image routing is the first
wired path; audio and video are stored but unrouted.

## 8. The harness controller

**Status: existing** · [Encoder and controller](../ENCODER-CONTROLLER.md)

E0 is deterministic hardware fit. It proposes only allowlisted runtime keys,
needs confirmation, records rollback state, and writes a redacted proposal
trace. Optional request features and an explicit CPU encoder vector give the
future E1 head privacy-preserving training data. E1 is not trained yet.

## 9. State, security and the tool gateway

**Status: core and DSH hook complete; MCP/browser adapters pending** · [State store](../STATE-STORE.md) · [Tool gateway](../TOOL-GATEWAY.md)

State is split by scope: settings, grants, profiles, controller traces, audit
and browser profiles. Studio settings now migrate from the legacy
`studio_settings.json` to `settings/studio.json`; the security policy and global
grants live in their own files.

The gateway core enforces reason checks, headless denial, session/global
grants, private-network approval, pending decisions, timeouts and append-only
audit. `hive tools status|policy|grants|pending|decide|revoke|audit`, the TUI
`a` overlay, and the Studio Global panel expose the state. A DSH `PreToolUse`
hook posts the concrete tool call to Hive, waits for the approval decision, and
fails closed if the gateway is unreachable; if the hook cannot be installed or
the selected runtime cannot load it (external preflight or bundled start-up
check), agent turns are refused. The
approve/deny path has passed against both the source-built and published DSH
runtimes. The Docker sandbox uses a pinned image and an explicit, audited
local-fallback approval. DSH web tools classify loopback, private, metadata and
DNS-private destinations as private; a read-only private fetch still needs
approval and grants are scoped to the exact origin. `hive web` adds a
stateless secure fetch with per-hop redirect checks, size/type limits, and
explicit download/upload gates. The optional browser adapter adds a fresh
Chromium context behind a validating local proxy, so redirects and subresources
are checked before a socket opens. MCP metadata/subagent inheritance remains
the release blocker.

## 10. Evidence, evaluation and release

**Status: existing** · [Release gate](../RELEASE-GATE.md) · [AB results](../AB-RESULTS.md) · [CLI](../CLI.md)

Hive treats claims as things to measure. Unit and integration tests cover the
components; the A/B campaign records the model tiers; `HG-1` is the manual
release gate because a person must be able to find, understand and safely
operate the surface. A failed gate is not a release.

## Appendix: command and file index

- [Full CLI reference](../CLI.md)
- [Secure fetch contract](../WEB-FETCH.md)
- [Browser adapter](../BROWSER.md)
- [Studio API map](../../hivebench/API.md)
- [README capabilities and API counts](../../hivebench/README.md)
- [Release gate](../RELEASE-GATE.md)
- [State store](../STATE-STORE.md)
- [Tool gateway](../TOOL-GATEWAY.md)

### Where the code lives

```text
harness/cli.py              command surface, TUI fallback, ui/tools commands
harness/tui.py              curses dashboard
harness/app.py              Studio HTTP app
harness/controller/         E0 controller and lazy encoder
harness/security.py         policy, grants, approval gateway, audit
harness/state_store.py      StatePaths, JsonStore, audit locks
harness/stack/              stack schema, residency and launch
cascade/                    cascade design and role contract
experiments/                benchmarks and measurement drivers
docs/                       design notes and this book
```

The book grows by adding a chapter only when the chapter can point to code,
a command or a measurement the reader can run. A diagram without that is a
brochure, not documentation.
