# Cascade accounting — verifier tier vs model-only gate (frozen organic-66)

The gate was being scored as if it were the only verifier. The cascade the
project is actually building has three tiers: a local **task verifier** where
the task admits one, the learned **gate** for open-ended tasks, and the hosted
**API** on rejection. This accounting measures that structure on the frozen
organic slice.

## The slice is almost entirely verifiable

| family | items | wrongs | programmatic checker |
|---|---|---|---|
| gsm8k | 32 | 3 | yes (numeric answer) |
| humaneval | 16 | 4 | yes (execute tests) |
| math | 13 | 2 | yes (final-answer check) |
| truthfulqa | 5 | 0 | no |
| **total** | **66** | **9** | 61/66 checkable (92%) |

**All nine measured organic wrongs are on checkable tasks.** The two universal
misses — `math-0013:hs1` (mod-6 error) and `humaneval-0001:hs1` (missing
odd-digit guard) — are exactly the kind a checker catches by construction.

## Policies on the 66 (API calls = wrongs + corrects escalated)

| policy | wrongs pass | wrongs escalated | corrects escalated | API calls |
|---|---|---|---|---|
| model-only L0.5 @0.55 | 2/9 | 7/9 | 10/57 | 17/66 |
| model-only MLP @0.50 | 4/9 | 5/9 | 7/57 | 12/66 |
| model-only AND (both <0.5) | 4/9 | 5/9 | 2/57 | 7/66 |
| verifier + L0.5 on open-ended | **0/9** | **9/9** | 1/57 | 10/66 |
| verifier + MLP on open-ended | **0/9** | **9/9** | **0/57** | **9/66** |
| verifier + accept all open-ended | 0/9 | 9/9 | 0/57 | 9/66 |

Verifier-first logic: checkable task → the checker decides (pass = accept,
fail = escalate); open-ended (truthfulqa) → the model gate; API only on
rejection.

Per-family, the model-only L0.5 @0.55 is weakest exactly where the checker is
strongest:

| family | caught | corrects escalated |
|---|---|---|
| gsm8k | 3/3 | 2/11 |
| humaneval | 3/4 | 2/12 |
| math | 1/2 | **5/11** |
| truthfulqa | 0/0 | 1/5 |

## What is and is not being claimed

- **The 0/9 is definitional, not a model win.** The checkers produced the
  labels, so using them as the tier-1 verifier is circular *on this benchmark*.
  The deployment claim is conditional: where a trusted verifier exists, it
  dominates the learned gate, and the learned gate is only needed for
  open-ended tasks.
- **The open-ended side is unmeasured.** The frozen test has five truthfulqa
  items, all correct: no catches to score. The four round-2 truthfulqa train
  wrongs are inside the heads' training data, so they are not a valid test.
  Tuning the open-ended tier needs a held-out open-ended slice (family-held-out
  refit or a new harvest), not more threshold work on these five.
- **Verifier costs and failure modes are task-specific.** Running 176B-generated
  code needs a sandbox; math needs a symbolic/numeric checker; gsm8k needs
  answer parsing. A wrong verifier becomes the cascade's error. These are
  deployment engineering, not free wins.
- **The model-only fallback is still the best learned gate:** L0.5 @0.55
  catches 7/9 at 10/57 correct escalations; MLP 5/9 at 7/57; AND 5/9 at 2/57.
  Do not use the gate alone on math (5/11 FR for 1/2 caught).

## Implication

The "organic gap" that blocked shipping is a **verifiable-task gap**: the
learned readout was being asked to replace a checker on tasks that have one.
The cascade should run the checker first where available, and reserve the
learned gate for open-ended work — whose performance is the actual open
question. Next experiments: (a) different-question probes on the same 0.8B as a
model-only second opinion; (b) a held-out open-ended evaluation before any
gate-for-open-ended claim.

## Artifacts

- `cascade-accounting.json` — policy numbers, family composition, checkable
  share, per-policy API-call counts.
