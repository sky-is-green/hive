# Compaction and engine surgery

A conversation that outgrows its window does not get truncated here: it gets **compacted into a
checkpoint**, and that checkpoint is the moment the suite offers to change *what serves the session*.
The function is modeled on opencode's session compaction
(`packages/core/src/session/compaction.ts`); the surgery offer is this suite's addition, because
sessions here are stateless (the transcript is the state) and the checkpoint is therefore a free
switch point structurally. The tests decide whether the switch sticks.

## The function (`harness/compaction/core.py`)

- **Triggers.** `Trigger.auto()` fires when the estimate crosses the model's window minus a buffer
  (`Settings.buffer`, default 10%); `Trigger.overflow(share)` is the same fire after a provider
  rejected a request as too long; `Trigger.manual()` is the user asking. With `auto` off, only
  manual runs.
- **Split.** The newest messages that fit `Settings.keep` tokens (default 15,000) stay verbatim; the
  older exchanges are summarized. Nothing to compact when everything fits `keep`.
- **Checkpoint.** The summary and the recent window are written as one block:
  `<conversation-checkpoint><summary>…</summary><recent-context>…</recent-context></conversation-checkpoint>`,
  which replaces the covered history in the transcript.
- **Enforcement and observability.** The summary must use the template's headings; a reply that
  ignores them gets one nudge and then fails the compaction (never a half-written checkpoint).
  Events (`compaction.started/completed/failed/skipped`) go to an injected sink, plus an
  `Editor.configure()` surface for plugins, as upstream.
- Models are stubbed everywhere: the summarizer is an injected `Summarizer`, the counter an injected
  callable, so every path is offline-testable.

## The surgery offer (`harness/compaction/surgery.py`)

At a completed checkpoint, `offer_surgery(session, checkpoint, current, candidates, apply, …)` runs
a transaction over the engines that could serve the session:

1. **probe** each candidate (reachable, which model it says it is);
2. **gate** it: deterministic checks plus a canary replay of the last exchanges, scored the way the
   suite scores everything (`GateResult(passed, score, baseline)`; an optional absolute floor);
3. **commit** only when the gate passes — `apply(engine)` is called exactly once, and the stack layer
   owns the actual switch; otherwise the session stays where it was and the checkpoint simply applies.

Nothing here spawns a server or loads a model; probes, gates and the switch are injected.

## Status

Implemented and tested: `harness/compaction/` (core + surgery), `tests/unit/test_compaction.py`
(13 tests; full suite 946 passed, 1 skipped). Not wired yet: the `harness/stack` listener that calls
`offer_surgery` on the compaction event, and the measurement that makes the gate meaningful — the
35B mid-conversation hand-off between engines, where the continuation drifts by the engines'
numeric difference and the canary score decides whether the switch is worth committing.
