# The Hive encoder and settings controller

**Status:** E0 implemented; learned rungs still experiments, 2026-10-25.

This is the plan for the small model that turns a request, a machine profile and
the current run state into **safe harness settings**. It is separate from the
cascade's task controller in [`../cascade/HARNESS-CONTROLLER.md`](../cascade/HARNESS-CONTROLLER.md):
that one routes and verifies answers; this one configures the harness that runs
them.

## The boundary: sensor, controller, executor

A sentence encoder maps text to a vector. It does not emit settings by itself.
The thing that chooses settings is a **controller**: a small readout over the
encoder's representation plus explicit state.

```text
request ──► encoder ──► representation ─┐
hardware profile ───────────────────────┼─► policy/head ─► proposal ─► guard ─► settings
run state (ctx, memory, latency, health)┘                         │
                                                                  └─► audit log / rollback
```

Package them together if useful, but keep the layers testable:

- **Encoder** — text/code (later image/audio/video) into an embedding space.
  It is also the memory/retrieval encoder.
- **Controller head** — a linear/MLP head over `[embedding; hardware features;
  run features]`. It proposes actions or rankings; it does not write files.
- **Guard** — allowlisted keys, ranges, scope, confirmation and rollback. No
  model output bypasses this layer.
- **Executor** — applies approved changes through the existing controls.

The first version should be **mostly deterministic**. A hardware recommendation
is a constrained fit problem, not a language-model judgement. The encoder earns
its place when it has to read the request: intent, task bucket, expected length,
risk and memory needs.

## What the controller may touch

These are already real settings blocks in
[`harness/controls.py`](../harness/controls.py) and the stack schema. Do not add
a second settings vocabulary.

| area | examples | why an encoder helps | risk |
|---|---|---|---|
| **memory** | `max_context`, `max_tokens`, dedup/drift thresholds, combination depth | distinguishes short chat, long narrative and code context | medium |
| **routing** | C1 bucket, P0–P6 path, model/tier choice | intent and expected work are textual | high |
| **runtime** | ctx, threads, parallel, ngl/gpu layers, KV type, batch/ubatch, speculative depth | workload length and hardware fit | high |
| **quality** | judge, gate, repair, escalation threshold | risk and checkability are task signals | high |
| **launch** | backend, card pin/split, `resident_gib`, VRAM margin, memory guard | hardware capability and displayed VRAM | high |
| **MCP** | read-only tools, enabled tools | conversational trust/permission model | medium |

High-risk settings are **proposals** until a human accepts them. Never let an
encoder turn “I want faster replies” into an unchecked write to a launch flag.

## First run: scan only with permission

Scanning is deterministic probing, not encoder inference. It should be a
separate, auditable step.

### Consent

Ask once, in `hive setup`:

```text
Hive can inspect this machine to recommend safe settings.
  [x] CPU, RAM and swap
  [x] GPUs, VRAM, GTT and display status
  [x] disk space and installed engine backends
  [ ] installed models and saved stacks        (local paths only)
  [ ] recent performance traces                (no prompts)
  [ ] export the profile for support           (off by default)

Nothing leaves this machine unless you export it. You can rescan or delete it
at any time.
```

Allow **no scan** and **partial scan**. The default is local-only, no network,
no reading of user documents. Do not scan model files beyond their configured
directories.

### What gets stored

```jsonc
{
  "schema": 1,
  "profile_id": "local-2026-10-25-a1b2",
  "scanned_at": "2026-10-25T21:00:00Z",
  "consent": {"hardware": true, "models": false, "traces": false, "export": false},
  "cpu": {"model": "...", "logical": 16, "physical": 8},
  "ram": {"total_gb": 30.9, "available_gb": 18.2, "swap_gb": 8.0},
  "gpus": [
    {"index": 0, "name": "AMD Radeon RX 7900 XT", "vram_gb": 21.46,
     "vram_used_gb": 19.32, "gtt_total_gb": 29.4, "display": true},
    {"index": 1, "name": "AMD Radeon RX 7900 XT", "vram_gb": 21.46,
     "vram_used_gb": 0.55, "gtt_total_gb": 29.4, "display": false}
  ],
  "backends": ["rocm"],
  "storage": {"models_free_gb": 23.7},
  "confidence": {"cpu": "high", "gpus": "high", "storage": "high"},
  "source": "sysfs+psutil"
}
```

Use the existing [`harness/hardware.py`](../harness/hardware.py) probes and
`hive doctor --json`; do not duplicate them. Persist under `HIVE_HOME` or the
configured state directory; include a schema version and a human-readable
timestamp.

### Recommendation flow

1. **Facts first.** Detect visible cards, display card, free VRAM, GTT, RAM,
   disk and backends.
2. **Fit constraints.** A model/stack must fit with the memory guard and
   display reserve. Reuse `harness/stack/residency.py` and the existing fit
   logic. If it does not fit, say so; never recommend “just try smaller”.
3. **Workload profile.** The encoder classifies the request (short chat, long
   context, code, vision, batch, risk). It may select a candidate profile, not a
   raw flag value.
4. **Cost/quality choice.** Choose a known-good profile; use measured latencies
   and the cascade table where available.
5. **Explain and propose.** Show the profile, evidence, confidence and what
   changes. Do not auto-apply on first run.
6. **Apply, measure, revise.** Record the actual outcome (first-token/decode
   latency, OOM/guard trip, acceptance, user correction). Revert on failure.

Example proposal:

```jsonc
{
  "schema": 1,
  "scope": "profile:local",
  "reason": "long-context coding on a 20 GiB display card",
  "changes": {
    "runtime_defaults.ctx_size": 16384,
    "runtime_defaults.ngl": 999,
    "runtime_defaults.resident_gib": 4,
    "quality_defaults.gate": "off"
  },
  "confidence": 0.78,
  "evidence": [
    "model weights fit with 1.8 GiB VRAM reserve",
    "ctx 16k leaves 3.1 GiB for KV and expert cache",
    "the request is code, so promotion to the larger tier is allowed"
  ],
  "requires_confirmation": true,
  "revert": "previous settings snapshot"
}
```

The controller may propose only keys in an allowlist with declared ranges and
types. A proposal is rejected if it would exceed the guard, hide a warning or
select an unavailable backend.

## The controller head

Start with a **small classification/ranking head**, not a generative model:

```text
input = [encoder(prompt/context); hardware vector; state vector]
targets = bucket, expected tokens, memory need, risk, a profile rank
loss = cross-entropy + calibration + pairwise preference
```

Recommended first targets, in order:

1. **intent/bucket** — easy/qa/code/reasoning/vision.
2. **expected output length** — for max tokens and timeout.
3. **context pressure** — short/long/needs retrieval.
4. **profile rank** — which measured profile to try first.
5. **risk/checkability** — whether `verify` or a human gate is required.

Do not start with a head that predicts arbitrary flags. A joint “settings vector”
has too little data and too many ways to be plausibly wrong.

### Training data

Use the harness's own logs, with consent and redaction:

- request class and text embedding (not necessarily raw text);
- hardware profile id and visible resource snapshot;
- chosen profile/settings and scope;
- outcome: accepted/rejected, latency, first-token time, OOM/guard trip,
  repair count, escalation, user correction;
- the **counterfactual** where available: a measured alternative profile.

Never train on prompts or traces marked private. Split by **session/prompt**,
not by random token or turn; a memory dataset can leak through adjacent turns.
Record the split manifest with the encoder.

### Calibration and safety

- Calibrate probabilities; a confidence score is only useful if
  `p=0.8` means about 80%.
- A low-confidence controller must fall back to the conservative deterministic
  profile, not guess.
- Wrong-target negative controls: feed a profile/encoder trained on another
  model or machine and report the degradation.
- Any setting that can OOM, change output quality materially, or disable a
  verifier requires confirmation.

## Packaging a custom encoder

An encoder intended for Hive should ship as a small, inspectable artifact:

```jsonc
{
  "schema": 1,
  "name": "hive-controller-encoder",
  "version": "0.1.0",
  "license": "apache-2.0",
  "modalities": ["text", "code"],
  "dimensions": 768,
  "max_tokens": 8192,
  "tasks": ["intent", "context_pressure", "risk"],
  "prompt_prefixes": {"query": "query: ", "document": "passage: "},
  "head": "linear|mlp",
  "thresholds": {"intent": 0.62, "risk": 0.8},
  "calibration": "temperature+Platt",
  "hardware_min": {"ram_gb": 2, "cpu": true},
  "fallback": "sentence-transformers/all-MiniLM-L6-v2",
  "sha256": "...",
  "training": {"data": "local-traces-v1", "split": "session-disjoint"},
  "eval": {"decision_accuracy": null, "ece": null, "hardware": "..."}
}
```

Keep the fallback explicit. If the custom encoder is missing, corrupted or
over-budget, Hive must run the conservative baseline, not fail to start.

Serving choices:

- **local in-process** on CPU for small encoders; never evict the engine for it;
- **served** through an OpenAI-compatible embedding endpoint when the encoder is
  larger or shared;
- **lazy load** only when the controller is invoked; the memory guard applies to
  it as well.

## Building one from scratch

This is the “own use case” path, not a requirement for the first release.

1. **Write the decisions, not the model.** Pick 3–5 actions with measurable
   outcomes: route, context budget, retrieval depth, verify/escalate, profile
   rank. Define the conservative fallback for each.
2. **Collect traces with consent.** Store request features/embeddings, hardware,
   chosen action, outcome, and counterfactuals. Redact prompts unless the user
   explicitly opts in. Keep a data card.
3. **Build the deterministic baseline.** A rules/fit planner must exist before
   the learned head; otherwise there is nothing to beat and no safe fallback.
4. **Split by session.** A held-out prompt must not share a conversation, a
   document or near-duplicate turns with training.
5. **Train an encoder + head.** Start with a small sentence encoder
   (MiniLM/BGE/EmbeddingGemma if it fits) frozen, then a ridge/MLP head; only
   fine-tune the encoder after the head and labels are stable.
6. **Calibrate.** Temperature/Platt on held-out data; report ECE and reliability,
   not just accuracy.
7. **Evaluate the whole loop.** Compare against the deterministic baseline on
   decision accuracy, latency, first-token time, memory, guard trips and user
   corrections. Include a wrong-machine and wrong-model negative control.
8. **Export and package.** Quantize only after the float model passes; write the
   manifest, checksum, license and fallback. Keep the training code and data
   card with the artifact.
9. **Stage the rollout.** Shadow mode first (propose, do not apply), then
   confirmation-only, then scoped auto-apply for low-risk keys. Never start with
   launch flags or memory thresholds.

## Experiment ladder

Do these in order; stop if the earlier rung fails.

| rung | what | learned? | success gate |
|---|---|---|---|
| E0 | hardware-only conservative recommender | no | settings fit; no OOM; user can explain every choice |
| E1 | intent/context classifier with a frozen small encoder | yes | beats rules on held-out requests; calibrated |
| E2 | action-ranking head over encoder + hardware | yes | beats E0 on latency/quality at equal or lower cost |
| E3 | custom Hive encoder trained on local traces | yes | beats E1/E2 on session-held-out data; portable manifest |
| E4 | multimodal senses (image/audio/video) | yes | encoder + a real generator; no fake answer path |

E0 is the first deliverable. It gives the product a useful first-run experience
without any encoder, and it produces the labels and counterfactuals E1–E3 need.

### E0 in this build

The deterministic rung is now wired end to end:

```bash
hive profile scan --consent hardware
hive profile show
hive controller propose --model models/gguf/some-model.gguf --ctx 16384
hive controller propose --model models/gguf/some-model.gguf \
  --request "How do I profile a long coding task?"
hive controller propose --model models/gguf/some-model.gguf \
  --request "..." --embed-request
hive controller apply <proposal-id> --yes
hive controller outcome <proposal-id> accepted --metric latency_ms=42
hive controller revert --yes
```

The controller proposes only these keys, with type/range checks:

```text
runtime_defaults.threads
runtime_defaults.ctx_size
runtime_defaults.ngl
runtime_defaults.resident_gib
```

It never writes the memory, quality, launch, encoder or MCP blocks. A proposal
is saved as a recommendation under `HIVE_STATE_DIR/profiles/`; applying it is a
separate confirmed action that stores the previous present/absent state under
`HIVE_STATE_DIR/settings/controller.json`, writes the studio settings, and
appends a redacted audit event. Reverting removes keys that did not exist before
the apply, rather than leaving `null` values behind.

The Studio's **Global** tab has the same flow: consent checkbox → scan →
optional model size/context → optional request text → propose → review
evidence/warnings → apply with a confirmation dialog → revert.

An optional request goes through the model-free
[`features.py`](../harness/controller/features.py) baseline (intent, context
pressure, risk) and can lower the context cap for a short chat or raise it for
code/reasoning. Only the feature dictionary and proposal are written to the
controller trace, never the raw request text. `--embed-request` is the explicit
opt-in that loads the configured sentence-transformer on the CPU and adds its
vector, model name and dimension to the trace; without it, E0 stays model-free.
The trace is append-only `profiles/controller-traces-YYYYMM.jsonl`;
`controller outcome` records the human/evaluation label and metrics, and
apply/revert append lifecycle events, so later E1 work has privacy-preserving
examples to label.

The encoder boundary is separate from E0. `hive encoder status` inspects the
configured model without loading it; `hive encoder embed "text"` loads the
sentence-transformers backend lazily on the CPU. It is local-only by default;
pass `--allow-download` to permit a missing model to be fetched. The controller
does not use an encoder for its decisions yet, so there is no fake embedding in
this path.

### Idle unload

The controller's lazy encoder has a configurable lifecycle in the `encoder`
settings block:

```jsonc
{
  "controller_idle_unload": true,
  "controller_idle_seconds": 300
}
```

When enabled, the last successful controller embedding arms a daemon timer.
If the encoder is idle for the configured number of seconds, the service drops
its reference to the model; the next request reloads it through the same lazy
factory. The timer never touches the serving engine or its VRAM. Turning the
toggle off keeps the encoder resident for the life of the Studio process.
`0.1` is the floor and `300` is the default. Changes take effect on the next
controller embedding; the timer is cancelled when the toggle is turned off.
`hive controller status` reports whether the controller encoder is loaded and
how long remains before unload.

## Current state and gaps

Already present:

- the five settings blocks and override precedence in
  [`harness/controls.py`](../harness/controls.py);
- hardware probes and fit/residency helpers;
- `hive setup` and `hive doctor` paths;
- the Studio Global controls and the CLI/TUI surfaces;
- the memory encoder path and `EmbeddingGemma` as a candidate, not a dependency;
- `StatePaths`/`JsonStore` atomic, locked, schema-versioned controller state;
- a consent-gated hardware profile store;
- the deterministic E0 recommender and allowlist;
- the model-free request-feature baseline and request-aware context cap;
- privacy-preserving proposal/lifecycle traces for E1 data collection;
- confirmed apply, rollback presence tracking, and the settings audit log;
- the lazy local `SentenceTransformerEncoder` adapter with configurable idle
  unload;
- CLI, API and Studio Global-tab surfaces for the E0 flow;
- migration of Studio settings toward `settings/studio.json` with a legacy
  read path.

Not wired:

- a first-run `hive setup` prompt for the hardware scan;
- a controller head or custom encoder package;
- trace collection/counterfactual labeling;
- a served-encoder adapter for the controller (the Studio's retrieval drone
  already has one);
- a multimodal generator, so senses cannot yet answer an image.

## Non-negotiables

- The encoder never writes settings directly.
- No hardware upload or prompt upload without explicit consent.
- No custom encoder without a fallback.
- No auto-apply of high-risk launch/memory/quality settings on first run.
- Every applied change has a scope, evidence, timestamp and revert path.
- A controller that cannot beat the deterministic baseline is not shipped.
- DSpark/cascade compatibility is target-specific; so is a controller head.
