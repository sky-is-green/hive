# Human gate before API escalation

## Why

A local rejection should not jump straight to a paid API call.  The person at
the terminal often knows what the local tier missed — a missing fact, an
ambiguous constraint, a tool that is unavailable, or simply that the task is
outside the local model's family.  Asking once, at the last moment, has two
effects:

1. It gives the user a chance to solve the problem locally, with one
   correction or one hint.
2. It records *why* local could not solve it, which is exactly the routing
   signal the cascade has been missing.  The choice itself is a gold-free
   label: `solve` means local was recoverable; `escalate` means the API was
   genuinely needed.

This is a gate, not a new model tier.  It sits after the verifier rejects and
after the automatic repair round (when one is configured) fails to clear the
finding, immediately before the hosted fallback.  The two top-level outcomes
stay simple:

- **Solve locally** — correct the answer, give the model a hint, or keep the
  local answer when the task's checker already passes.
- **Escalate to API** — send the task to the configured frontier endpoint.

## Flow

```
local generation -> verifier -> reject
                                  |
                                  v
                    automatic repair (one round,
                     the finding quoted, T=0)
                        /               \
                  cleared             still flagged
                     |                     |
             ship canonical        human gate (interactive)
                                     /                \
                                 solve                escalate
                              /        \                  |
                     corrected answer   hint -> one       API answer
                              \        /   local retry        |
                               checker                     checker
                                  |                           |
                              ship local                   ship API
```

The first trigger is deliberately verbose.  It prints:

- the task id, bucket, and task prompt;
- the local answer;
- the automatic repair attempt and what it still trips on (when one ran);
- the verifier's decision, calibrated confidence, and reason;
- the task checker result (`PASSED`, `FAILED`, or `not available`);
- the API endpoint and model that would be called; and
- the two choices.

Later triggers are compact, because the user has already seen the mechanism.
The answer as recorded (up to the report field limit) remains in `report.json`.

## The solve path

After `[s]olve`:

- a plain line is treated as a corrected answer;
- a line prefixed `hint:` is passed to the local model for one retry, and the
  retry is checked again;
- `keep` is offered only when the task checker already passes, i.e. the
  verifier is the thing that is wrong;
- a failed correction returns to the gate with the new evidence.

The solve path records an optional reason (`what was missing?` / `what did you
change?`).  It never silently accepts a failing answer.

## The escalate path

After `[e]scalate`, the user may give a short reason.  The reason is stored in
the report and appended to the API prompt as reviewer context, so the
frontier model knows what the local tier could not resolve.  The API call is
paid and logged like any other fallback.

## Non-interactive runs

`--human-gate auto` (the default) enables the prompt only when **all** of the
following hold:

- `--api-fallback` is configured;
- the run is not `--router-mode measure`; and
- stdin is a TTY.

That keeps benchmark and batch runs non-interactive.  `--human-gate on` forces
the prompt and warns if there is no fallback or if the run is a measurement;
`--human-gate off` preserves the old automatic behavior.  EOF at the prompt
escalates with an explicit reason rather than hanging forever.

## What is recorded

Each gated task gets a `human_gate` object in `report.json`:

```json
{
  "asked": true,
  "first": true,
  "choice": "escalate",
  "mode": "escalate",
  "reason": "the local model lacks the current release date",
  "hint": "",
  "human_answer": null,
  "attempts": 1,
  "accepted_locally": false,
  "api_target": "https://.../v1",
  "api_model": "..."
}
```

When the repair leg ran, the same task record carries a `repair` object (the
quoted finding, the attempt, whether it cleared, and the canonical re-framing
when it did); the summary adds `repairs_attempted/cleared/still_flagged`.  The
report summary adds `human_gate.asked/solve_attempts/solved/kept/
escalated/first_triggered/reasons`, plus a `human_corrected` count.  A
`HUMAN` telemetry row records the decision and reason alongside the existing
model rows.  These rows are the raw material for the deferred C2 controller:
they are human labels about recoverability, not gold answers.

## Offline workload preview

Before changing the live gate, use `experiments/cascade/eval_partner_transfer.py`
to apply a frozen judge head to a held-out organic/QA set.  It writes:

- `human-gate-candidates.jsonl` — every candidate with its probability, bucket,
  origin, prompt and local answer, ranked by the judge;
- `human-gate-preview.md` — the **real** `HumanGate._show` rendering for the
  first few triggers (verbose first, compact later), with an explicit note that
  it is offline and that evaluation labels are never shown to the runtime gate;
- `BONSAI-TRANSFER.md` / `eval-*.json` — judge accuracy plus threshold
  sensitivity (rejected counts and false rejects at 0.50/0.58/0.60/0.65/0.70).

The preview is a workload and visibility test, not a live run.  The live gate
still triggers only on a verifier reject; the judge probability is a stand-in
for that trigger in the offline analysis.  Thresholds are chosen from the
gold-free workload and the user's tolerance for false rejects, never from the
evaluation labels at runtime.

The first Bonsai baseline (CPU, 2026-10-18) is in
`experiments/cascade/results/partner-bonsai-organic-20261018/`.  On 252
held-out candidates the frozen head rejected 127 at p<0.58 — every rejection
was a wrong answer and no correct answer was rejected, but it caught only
1 of 9 organic wrongs and missed 13 wrongs overall (recall 90.7%, carried by
J-Bench/QA).  `human-gate-missed.jsonl` lists those misses.  The lesson for
the gate: precision is not enough; recall on organic free-form tasks has to
come from the 0.8B verifier or the human/API tier.

Pair it with `run_partner_probe.sh` for a guarded, pure-CPU extraction so the
176B serve can stay up.

## Scope and caveats

- The gate is provider-agnostic.  It displays whatever `--api-fallback` names;
  it does not choose a provider or hide the price.
- The checker is the task's runtime checker.  In the bench corpus some
  checkers are gold-dependent and are for evaluation only; the gate treats
  them as evidence, not as a deployment verifier.  The gold-free runtime split
  (`code_exec`, `constraint_check`, `schema_ast`, `consistency`) is unchanged.
- The gate does not load a second model.  It reuses the local tier for a hint
  retry, so it adds no VRAM context.  Heavy partner runs still need a resource
  guard; the 35B partner extraction run consumed unexpected ROCm memory
  despite `-ngl 0` and has been stopped.
- The router's pre-generation `gate -> api` path is not covered yet; this gate
  covers the local-first verifier-rejection path, which is the default.
