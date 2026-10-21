# The harness controller: one loop, any driver

**The claim.** The stack the A/B campaign measured is not a ladder with a
fixed direction. It is a **set of tiers, a verification loop and a
measured table**, and a *controller* decides who handles what. The same
controller can be a small local model, a frontier API model, or the
person at the terminal - the tools are the same and the table is the
same. What changes is the driver, not the system.

This document is the **drop-in contract**: the tools, the invariants,
the table and the log. Anyone can point their own controller at these
tools, and the measurements in `docs/AB-RESULTS.md` remain the reference
for what each tier is worth. `docs/CONTROLLER-PRIOR-ART.md` is the survey
of how other harnesses implement the same role, and which of their
patterns this contract borrows.

## Why the controller is a model *and* a table

The campaign's sharpest result is that a judgement model is not enough:
the 176B's head sends up 3 of 7 errors and gives **four of the seven
`p >= 0.97`** - both arithmetic and both code answers. A controller
that only *reads* answers cannot see a plausible wrong one. What catches
those is execution. So the controller's competence is not "knowing the
answer"; it is **routing and verifying**:

1. **Checkability sorts before competence.** If a domain admits a check
   (run the code, recompute the arithmetic), the check decides - not the
   strongest model's opinion.
2. **Competence sorts before cost**, per bucket, from the table.
3. **Cost sorts last**, but it is what makes the cascade worth running.

## The tools (the contract)

Four tools, all HTTP. A controller that can call these is compatible
with the harness, whatever model it is.

```jsonc
// 1. ask - any tier in the registry answers
{
  "name": "ask",
  "arguments": {
    "model": "t0-35b | t1-bonsai-27b | api-frontier",
    "prompt": "string",
    "max_tokens": 1024
  }
}
// -> { "answer": "string", "model": "...", "ms": 0,
//      "tokens": { "prompt": 0, "completion": 0 } }

// 2. verify - the harness picks the method, the evidence is the verdict
{
  "name": "verify",
  "arguments": { "task": "string", "answer": "string" }
}
// -> { "verdict": "pass | fail | unknown",
//      "method": "execution | grounding | none",
//      "evidence": "stdout/exit | the cited sources | why-unknown",
//      "check": "the program, or the claim set that was grounded",
//      "mutated_ok": true }   // the check failed a corrupted answer
//
// execution: a program decides (run the code, recompute the arithmetic).
// grounding: claim-level support against the retrieved context, each
// claim cited; an uncited claim is not a pass.
// The harness chooses; the controller never writes the verdict itself.

// 3. score - the judge head (the 176B's judgements, or a distilled copy)
{
  "name": "score",
  "arguments": { "task": "string", "answer": "string" }
}
// -> { "p_correct": 0.0, "model": "...", "ms": 0 }

// 4. ask_human - the last local tier
{
  "name": "ask_human",
  "arguments": { "question": "string", "answer": "string",
                 "evidence": "string" }
}
// -> { "decision": "solve | escalate | reject", "hint": "string" }
```

## The invariants

These are not prompt suggestions; they are what makes the system
falsifiable, and the harness enforces them - the tool interfaces and the
policy layer, not the prompt. The controller is told them so it can
plan; it does not get to opt out. (OpenCode's two-plane design is the
precedent: permissions say what an agent may do, policies say what it
cannot do at all, and neither is a sentence in a prompt.)

- **A `pass` is authoritative.** The checker's program ran and printed a
  pass. The controller does not override it with a judgement.
- **A `fail` is evidence, not a verdict.** The controller reads the
  evidence and re-checks once (a bad check is a real failure mode). A
  second `fail` rejects the answer.
- **`unknown` is honest.** A check that cannot be *falsified against a
  mutated answer* (`mutated_ok=false`) is `unknown`, never `pass`. A
  grounding check that cannot cite is also `unknown`.
- **Two verifiers, one interface.** `execution` for what a program can
  settle (code, arithmetic); `grounding` for what only evidence can
  settle (claims against sources). They are the same tool because the
  controller should not have to know which one it needs - but the log
  records the method, so the routing table can be measured per method.
- **`score` never overrides execution.** A high `p_correct` on a checked
  answer is ignored; a `score` is only consulted where `verify` is
  `unknown`.
- **Escalate with evidence.** An API call carries the task, the local
  answer, and the check/judge findings. The API is paid for the residue,
  not for the whole task.
- **Every decision is logged** (see below). The controller's own
  confidence is not evidence and is not logged as one.
- **The loop is bounded.** After a configured number of steps the
  harness removes the tools and requires a final answer - `accepted`,
  `escalated` or `rejected`. An unbounded controller is a bug, not a
  feature (OpenCode's `steps` is the same bound).

## The routing table (measured, 2026-10-21)

| the tier | what it is for | the measured number | the cost |
|---|---|---|---|
| t0 - the 35B | the bulk | 21/28 (0.750) | the cheapest local |
| t1 - Bonsai 2 27B | the repairable errors | 26/28 (0.929); **+5 net** over the 35B | 8.9 s/item |
| the judge (176B head) | the plausible errors | AUC 0.864; 3 of 7 errors, 0 of 21 correct | 2.5 s/verdict |
| the checker (4-9B + sandbox) | the executable errors | **not yet measured** | a sandbox call |
| the API | what nothing local settles | not yet measured (the key is expired) | per token |

The memory decisions (dedup, drift) are a **separate lane**: the
controller does not choose them per task, the membrane does. Their
numbers are in `docs/AB-RESULTS.md`; the drift corpus is being rebuilt
(2C-septies) and is not part of this table.

## The loop

```text
answer  = ask(t0, task)
v       = verify(task, answer)
if v.verdict == "pass":                      return finish(answer, v)
if v.verdict == "fail":                      answer = ask(t1, task) -> re-verify
if v.verdict == "unknown":
    p = score(task, answer)
    if p >= threshold:                       return finish(answer, v)
    answer = ask(t1, task)
    v = verify(task, answer)
    if v.verdict == "pass":                  return finish(answer, v)
    h = ask_human(task, answer, v.evidence)
    if h.decision == "escalate":             return finish(ask(api, task, evidence), ...)
    return finish(answer, v)
```

The policy is data, not code: the threshold, the per-bucket routing and
the model registry are configuration the controller reads. A different
operator can change the table without changing the loop.

## The three drivers

- **Local** (the Studio's Global tab, or a local controller model): the
  loop above, the 35B as the bulk, the escalation to Bonsai, the human
  gate before the API.
- **API**: the frontier model is the controller. It calls the *same*
  tools - `ask(t0, ...)` for bulk work, `verify` for every answer it
  intends to accept, `score` where a check is impossible. The API pays
  for the plan and the residue; the locals carry the bulk and the
  verification.
- **Human**: the person runs the same loop by hand; the tools are the
  Studio's buttons.

A controller is drop-in if it implements the four calls and honours the
invariants. The harness does not care whether the controller is 0.8B or
a frontier model, local or remote.

## The log (the training data)

One JSONL record per decision:

```jsonc
{ "task_id": "...", "driver": "local|api|human",
  "steps": [
    {"tool": "ask", "model": "t0", "answer_sha": "...", "ms": 0},
    {"tool": "verify", "verdict": "fail", "check_sha": "...", "mutated_ok": true},
    {"tool": "ask", "model": "t1", "answer_sha": "...", "ms": 0},
    {"tool": "verify", "verdict": "pass"}
  ],
  "final": "accepted|escalated|rejected",
  "outcome": "correct|wrong|unknown"   // filled by the evaluation pass, never by the loop
}
```

The separation is the campaign's rule: the runtime never sees the answer
key; the outcome is attached afterwards. The log is what a controller
model is trained on - and what proves the table still holds.

## The controller's prompt (reference)

The policy is short enough to be a system prompt; everything below the
table is measured, so a controller can start from this text and replace
the table as its own log grows.

```text
You are the controller of a local-first harness. You do not answer the
task yourself; you route it and you verify it.

Tools:
- ask(model, prompt): t0 is the 35B (the bulk); t1 is the 27B (repair);
  api-frontier is the paid tier (the residue only).
- verify(task, answer): runs an executable check. pass | fail | unknown.
  A pass is authoritative. A fail is evidence - re-check once, then
  reject. unknown means no falsifiable check exists.
- score(task, answer): the judge's p_correct. It is consulted only when
  verify is unknown. It never overrides a check.
- ask_human(question, answer, evidence): the last local tier.

Order: t0 -> verify; on fail, t1 -> verify; on unknown, score ->
t1 -> verify; then the human; then the API. Never accept an answer that
failed a check. Never reject one that passed. Send the evidence with
every escalation. Log every step.
```

The table itself is data, not prose: the model registry, the threshold
and the per-bucket routing are configuration. A different operator can
swap the models without touching the loop, which is what makes the
system drop-in in both directions - their controller on this harness, or
this controller on their tiers.

## What is not built

- The **checker service** (a sandbox + the 4-9B/0.8B agent). The
  mutation self-test is specified, not implemented.
- The **controller model** itself. The loop is written and the tools are
  specified; the first controller can be a 4B with the tool schema, or a
  frontier API.
- The **API driver** (the paid key is expired).
- The **distilled judge**: the 176B's judgements are portable in
  principle; the head's blind spot (plausible answers) is inherited, so
  it stays a tool for the `unknown` branch, never the gate.

## Why this is the right seam

The campaign measured the tiers one at a time and found that the
*consequence* of a decision is where the value is: the judge is cheap and
good, but the escalation it triggered was worth zero; the encoder's
decision can be moved, but a false merge deletes a fact. The controller
is the component that owns the consequences - it chooses which action to
take on a judgement. Putting the loop behind a stable tool surface is
what makes it improveable: every run logs a decision, every decision can
be scored afterwards, and the table is re-derived from the log.
