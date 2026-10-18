# The arithmetic-consistency verifier (2026-10-18)

A gold-free verifier that finds the equations an answer asserts about itself, evaluates both sides,
and flags the ones that disagree.  It is the `consistency` class from `bench_registry.json`
("disagreement / arithmetic mismatch", `gold_free_runtime: true`), which was defined but never
implemented.

```sh
.venv/bin/python experiments/cascade/consistency_check.py \
    --out experiments/cascade/results/scion176b-harvest-20261018/consistency-check.json
```

## Why it exists

Window B left two organic items that *every* learned detector accepts with high confidence:
`gsm8k-0033:hs1` and `math-0013:hs1`.  Both are the same failure — the reasoning chain is right and
one arithmetic step is wrong:

| item | the answer asserts | actually | detector readings |
| --- | --- | --- | --- |
| `gsm8k-0033:hs1` | `5 + 10 + 10 + 2.5 + 2.5 = 25` | 30 (= gold) | L0.5 0.525 · MLP **1.000** · self-harvest 0.787 |
| `math-0013:hs1` | `2007 = 6 x 334 + 1` | 6×334 = 2004, remainder 3 (= gold digit 2) | L0.5 **0.999** · self-community **1.000** · self-harvest 0.774 |

A 176B has no trouble adding 5+10+10+2.5+2.5.  It emitted the wrong token.  This is not a
representation limit and no amount of probe capacity or natural-error harvest data addresses it —
but re-evaluating the arithmetic the model itself wrote does, with no gold answer and no model.

## Result

Equations are only read from inside math spans (`$$…$$`, `\[…\]`, `\(…\)`, `$…$`), never from prose.

| slice | n | wrong | equations checked | flagged | of which wrong | precision | recall | false-flag rate on corrects |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **organic-66** (natural errors) | 66 | 9 | 70 | **2** | **2** | **1.000** | 0.222 | **0.000** |
| harvest-train-871 | 871 | 461 | 791 | 38 | 19 | 0.500 | 0.041 | 0.046 |
| harvest-937 | 937 | 470 | 861 | 40 | 21 | 0.525 | 0.045 | 0.041 |

On the organic slice it flags **exactly the two items nothing else catches, and nothing else** —
precision 1.000, zero false positives.

The lower precision on the full 937 is informative rather than disappointing.  91% of that corpus's
wrongs are mechanical perturbations (num-flip, boxed-edit, code-edit, choice-flip), which change the
*terms* while leaving the arithmetic internally consistent — so this verifier correctly ignores them
and fires only on genuine arithmetic slips.  It is measuring a different thing than the perturbation
detectors, which is why it is complementary.

## What a gold-free stack catches on organic-66

| detector | flags | catches | false-rejects |
| --- | --- | --- | --- |
| `code_exec` (humaneval tests) | 4 | 4/9 | 0/57 |
| `consistency` (this) | 2 | 2/9 | 0/57 |
| **both, gold-free** | **6** | **6/9** | **0/57** |
| learned gate (for reference) | — | 7/9 at 5 CR | 5/57 |
| checker *with* gold (eval only) | 9 | 9/9 | 0/57 |

So six of the nine organic wrongs are caught by two deterministic, gold-free, zero-false-positive
mechanisms that cost one regex and one arithmetic evaluation.  The three that remain are
`gsm8k-0009:hs2`, `gsm8k-0015:hs2` and `math-0020:hs2` — all three *are* caught by the 0.8B heads, so
they are the part the learned gate is actually for.

This reframes the gate's role.  It was measured against a checker that already solves the test set
(9/9 with gold, 0 false rejects); the honest comparison is the 6/9 gold-free deterministic stack,
which the gate does not beat on this slice — it adds 3 more catches at the cost of 5 correct
rejections on 57.

## Caveats

- Recall is low by construction (2/9, 0.045): this detects arithmetic slips only.  It is a
  high-precision trigger, not a general detector.
- Two guardrails were needed to get precision up, and both are heuristics: equations are read only
  from math spans, a run into a letter (`2.5x = 15`) is variable notation and is skipped, and a span
  mixing `\text{…}` or the word "remainder" states a convention rather than an equality.  Before
  trusting 2007/6 = 334-with-remainder-3 as an error, those guards are what stop it.
- The eval() is guarded to digits and arithmetic operators only, but a production version should use
  an AST walk rather than `eval` on model output.
- n = 9 on the organic slice.  Precision 1.000 on 2 flags is 2/2, not a reliable estimate; the 937
  numbers (40 flags, 21 wrong) are the better-powered ones.

## Files

- `consistency_check.py` — the verifier and its measurement.
- `../results/scion176b-harvest-20261018/consistency-check.json` — per-slice numbers and every
  flagged equation.