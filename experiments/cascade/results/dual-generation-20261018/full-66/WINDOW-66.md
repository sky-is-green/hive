# WINDOW 66 — both tiers on strata, 66 organic items, and the cascade does not pay

**Date** 2026-10-18 15:11–15:20 · 473 s wall · `--max-tokens 1024` · one cap hit (Bonsai), no missing finals
Engines: `strata-q36` (Scion-35B-A3B, `qwen35moe`) and `strata-dense` (Bonsai-2 27B, dense `qwen35`), one card each, fired at both per item.

## Accuracy on the 66 (gold, `numeric_gold` — eval only)

| model | gold-correct | on the 57 the 176B solved | tokens median | ms median |
|---|---|---|---|---|
| Bonsai-2 27B dense | **58/66 (0.879)** | 50/57 (0.877) | 338 | 6,417 |
| 176B (the labels) | 57/66 (0.864) | 57/57 by construction | 242 | 4,430 |
| Scion-35B-A3B | 46/66 (0.697) | 42/57 (0.737) | 180 | 1,745 |

**The smaller model is the slower one.** Bonsai writes 338-token chains at 65 tok/s; the 176B writes
242 at ~55 tok/s effective (MoE + its MTP speculation); the 35B writes 180 at 116 tok/s. "Smallest
first" is a trap on this hardware: the tier-1 candidate costs 1.5× the 176B's latency per item.

Two caveats that keep this honest. The bench is **176B-enriched by construction** — 9 organic wrongs
plus perturbed variants — so 58 vs 57 is not a general accuracy claim. And the 35B's answers were
generated at temperature 0, so its 20 wrongs are the model's real errors, not sampling noise.

## The deployable gate, and what it misses

The only gold-free runtime instrument applicable here is the arithmetic verifier
(`consistency_check.py`). Accept = not flagged. Nothing else — `checker_pass` in these rows is the
gold comparator and is eval-only.

| tier | accepted (of 66) | of those, wrong | flagged | of those, correct | rejected correct |
|---|---|---|---|---|---|
| 35B | 63 | **17** | 3 | 0 | 0 |
| Bonsai | 66 | **8** | 0 | 0 | 0 |

Neither tier rejects a correct answer (the invariant the project set). But on the 57 the 176B solved,
the 35B passes 12 of its 15 wrongs silently and Bonsai passes all 7. **The gate is not why these
models are usable or not; it barely fires.**

## The flows, costed (on the 57 the 176B solved)

| flow | tier-1 accepts | escalates | cost | vs alone |
|---|---|---|---|---|
| 35B → 3 (5%) → Bonsai | 54/57 | 3 | **210 s** | 35B alone 96 s, Bonsai alone 409 s, **176B alone 295 s** |
| Bonsai → 35B | 57/57 | 0 | 409 s | — |

With the swap charged 32 s per escalation. The cheapest flow (35B-first) lands ~77–79% accurate on
this subset versus 100% for the 176B (by construction) and 88% for Bonsai-alone — it is cheaper than
the 176B and clearly worse. **No arrangement of these two tiers beats the 176B on accuracy at equal
or better latency; the cascade as specified does not pay on organic math.**

## Agreement, for the arbiter that was supposed to come next

46/66 agreed, 20 disagreed, 17 of the disagreements on items the 176B solved. Final answers match
the 176B's on 55/65 for Bonsai and 40/65 for the 35B — so the "let the small model disagree, then
escalate" design has no shortage of signal, only no gate that reads it.

## The instrument finding this run produced

Running the verifier across three models' answers to the same items showed its 1.000 precision was
**one model's style, not a property of the verifier**: 5 false flags on Bonsai (fragments of variable
equations, a prime factorization read as `40 = 2`, long-division quotients) and near-zero coverage on
the 35B (every number unit-labelled, and spans containing `\text{` were skipped outright). Four fixes
later (see `results/consistency-v2-20261018/`) its precision is 1.000 on all three sets and it now
catches the 35B's `24 × $125 = $2,880`. This is the more reusable result of the window: a gold-free
verifier has to be validated across answer styles, not just across items.

## Files

`full-66/dual-generation.jsonl` (per item, per model: text, final, gold, finish reason, tokens, ms),
`dual-generation-summary.json`, `cascade-analysis.json` (the gate recomputed from the text, so a
verifier fix re-scores it), this note. The per-item jsonl and the engine/server logs stay local,
gitignored like the run logs.
