# Documentation

Start here:

| Need | Read |
|---|---|
| install the Rust runtime and Python reference | [INSTALL.md](INSTALL.md) |
| serve, inspect, and stop stacks | [CLI.md](CLI.md) |
| run the daemon and Studio API | [DAEMON.md](DAEMON.md) |
| migrate from the Python `hive` | [MIGRATION-HIVE.md](MIGRATION-HIVE.md) |
| releases, checksums, signing | [RELEASES.md](RELEASES.md) |
| current measured state | [STATUS.md](STATUS.md) |

## Runtime and engines

| Area | Docs |
|---|---|
| engine behaviour and AMD options | [ENGINE.md](ENGINE.md), [ENGINE-OPTIONS.md](ENGINE-OPTIONS.md) |
| serving on AMD | [SERVE-AMD.md](SERVE-AMD.md) |
| remote GPU lane | [REMOTE-SERVE.md](REMOTE-SERVE.md) |
| encoder and controller | [ENCODER-CONTROLLER.md](ENCODER-CONTROLLER.md), [PHASE3-ENCODER.md](PHASE3-ENCODER.md) |
| persistent state | [STATE-STORE.md](STATE-STORE.md) |

## Safety and tools

| Area | Docs |
|---|---|
| tool approvals and audit | [TOOL-GATEWAY.md](TOOL-GATEWAY.md) |
| activity, Stop, and Cease | [ACTIVITY-CONTROL.md](ACTIVITY-CONTROL.md) |
| secure fetch | [WEB-FETCH.md](WEB-FETCH.md) |
| Chromium browsing | [BROWSER.md](BROWSER.md) |
| dedicated AI user | [AI-USER.md](AI-USER.md) |
| WASM plugins | [PLUGINS.md](PLUGINS.md) |

## Research

| Area | Docs |
|---|---|
| the A/B campaign | [AB-PLAN.md](AB-PLAN.md), [AB-RESULTS.md](AB-RESULTS.md) |
| training and cloud runs | [TRAINING.md](TRAINING.md), [CLOUD-RUNBOOK.md](CLOUD-RUNBOOK.md) |
| model studies | [SCION-DIRECT-AB.md](SCION-DIRECT-AB.md), [SCION-DIRECT-AB-RESULTS.md](SCION-DIRECT-AB-RESULTS.md), [round5/](round5/) |
| prior art | [SLOT-PRIOR-ART.md](SLOT-PRIOR-ART.md) |
| papers | [../splinter/SPLINTER-WHITE-PAPER.md](../splinter/SPLINTER-WHITE-PAPER.md), [../cascade/CASCADE-PAPER.md](../cascade/CASCADE-PAPER.md) |

## Rewrite record

These are historical records for the Rust migration. The runtime is merged and
released as `v0.2.0`; they are not day-to-day operating docs.

| Area | Docs |
|---|---|
| migration plan and gates | [RUST-MIGRATION.md](RUST-MIGRATION.md) |
| caveat ledger and triage | [REWRITE-CAVEATS.md](REWRITE-CAVEATS.md) |
| final review and language fit | [FINAL-REVIEW.md](FINAL-REVIEW.md) |
| UX walkthrough and evidence | [UX-WALKTHROUGH.md](UX-WALKTHROUGH.md) |
| phase baselines | [PHASE0-BASELINE.md](PHASE0-BASELINE.md) |
| release gate | [RELEASE-GATE.md](RELEASE-GATE.md) |

## Interface guides

- [Working modes](book/01-working-modes.md)
- [Book index](book/README.md)
- [The Hive CLI](CLI.md)
