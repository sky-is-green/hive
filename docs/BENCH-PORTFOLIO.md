# Bench portfolio and the router boundary

"Download all the benches" is the right instinct — more task families give the
router more to work with — but **only the gold-free subset adds runtime
correction**. The rest are evaluation data and family priors. This note fixes
that boundary, lists the starter pack, and defines the ingest/eval pipeline.

## What a bench can buy

A router can correct a failure only if it can **detect** it without already
knowing the answer. That splits benches three ways:

| bench use | examples | what lands in the router |
|---|---|---|
| **gold-free runtime verifier** | code tests, IFEval constraints, JSON schema, tool-call AST, consistency/recompute | reject / repair / escalate |
| **gate training + family prior** | any labelled bench (GSM8K, MATH, AIME, MMLU-Pro, …) | correctness predictor; "this family is weak" routing |
| **judge only** | MT-Bench, Arena-Hard, open-ended QA | hosted API judge; no local correction |

The 2×2 that drives routing:

| | failure detectable at runtime | not detectable |
|---|---|---|
| **model capable** | verifier passes → accept | gate/judge (open-ended quality) |
| **model weak** | verifier fails → repair or escalate | **route the whole family to the API** (bench prior) |

The bottom-right cell is the important one: benches can tell us the 176B is weak
on a family (AIME, repo-level code, long-context) even when no item-level check
exists. The router routes that family to the API by default; the gate is not
asked to detect individual errors it cannot see.

## Gold-free vs gold-dependent (the correction filter)

- **Code:** HumanEval/MBPP/LiveCodeBench ship tests with the task. If a
  deployment task also supplies tests, `code_exec` is a genuine runtime
  verifier with actionable feedback (failing test + traceback). Without tests,
  there is no check; route by prior.
- **Instruction following:** IFEval's constraints (`instruction_id_list` +
  `kwargs`) are task-intrinsic. This is the most underrated router graft: some
  violations are deterministically repairable (format, length, punctuation),
  the rest are escalation triggers.
- **Math/GSM8K/AIME:** the bench checker compares to a **gold answer**. At
  runtime the gold is usually absent. Available substitutes: task-supplied
  expected answer, an independent solver, or consistency/recomputation. Treat
  the final-answer check as conditional, not automatic.
- **MCQ/knowledge (MMLU-Pro, TruthfulQA-MC1):** answer-key checks are
  gold-dependent. They are excellent for family priors and gate training, but
  they give the router **no runtime correction path** — without the key it
  cannot know the letter is wrong.
- **Open-ended:** no programmatic gold; the judge tier (hosted API) is the only
  honest verifier.

This is the correction to "graft every bench checker onto the router": a
checker that needs the gold is an evaluation tool, not a runtime verifier. The
deployable grafts are `code_exec`, `constraint_check`, `schema_ast`,
`consistency`, and `judge`.

## Downloaded starter pack (hard + checkable first)

Local data lives in `data/benches/` (gitignored); `data/benches/manifest.json`
and `experiments/cascade/bench_registry.json` carry source URL, license, rows,
bytes and sha256.

| bench | rows | verifier | runtime | use | license |
|---|---|---|---|---|---|
| HumanEval | 164 | code_exec | tests supplied | verifier + negatives | MIT |
| MBPP | 974 | code_exec | tests supplied | verifier + negatives | CC-BY-4.0 |
| IFEval | 541 | constraint_check | gold-free | verifier + repair | Apache-2.0 |
| GSM8K test/train | 1,319 / 7,473 | numeric_gold | conditional | train + eval | MIT |
| AIME | 90 | numeric_gold | conditional | eval + hard negatives | Apache-2.0 |
| LiveCodeBench | 400 | code_exec | tests supplied | verifier + negatives | CC (card) |
| MMLU-Pro test | 12,032 | choice_gold | eval only | family prior + gate train | MIT |

Existing in-repo corpora (already wired): `cascade-bench-v1` (100 frozen tasks:
GSM8K/MATH/HumanEval/TruthfulQA), `cascade-community-v1` (664 tasks), and the
full-HumanEval code set (164). The new pack adds hard code (LiveCodeBench),
hard math (AIME), the constraint class (IFEval), and a large MCQ family prior
(MMLU-Pro) without touching the frozen J-Bench.

### Queued, not bulk-downloaded

| bench | why | status |
|---|---|---|
| HumanEval+ / MBPP+ (EvalPlus) | stronger tests → more organic code negatives | locate source |
| BFCL | tool-call AST/executable checks, gold-free | queued |
| MT-Bench / Arena-Hard | open-ended judge tier + family-held-out gate eval | needs API provider |
| GPQA | hard knowledge family prior | gated |
| SWE-bench-lite | repo-level code verification | heavy; later |
| RULER / LongBench | long-context family prior | only if 256K workload needs it |

Deferred: saturated easy MCQ (few wrongs, gold-dependent, no runtime path),
safety/toxicity (different objective), vision (flashnext is text-only).

## Why not literally all benches now

1. **The bottleneck is answers + labels, not files.** A bench gives tasks; the
   176B must answer them and a verifier must label them. That is the expensive,
   serve-window-bound step. Downloading 100 benches without that pipeline is
   hoarding.
2. **Most checkers are gold-dependent.** They do not become runtime corrections
   (see above). Wiring them into the router creates a false sense of safety.
3. **Saturation and contamination.** Easy knowledge benches give almost no
   wrongs; hard benches may be in the 176B's training data. Prefer time-sliced
   (LiveCodeBench) or held-out/private subsets.
4. **Licensing and size.** Work drive has 330 GB free, but "all" is unbounded
   (repo-based benches, gated sets), license-mixed and duplicated. Curate with a
   manifest and keep data out of git.
5. **Router bloat.** A branch per benchmark is a test harness, not a product.
   The LOFO result is the warning: per-family behaviour does not transfer for
   free. Build **verifier classes** and family priors, and hold out whole
   benches.

## The graft that matters: verify → repair → escalate

Verifiers are more valuable as **feedback** than as rejectors:

```
176B answer
  → gold-free verifier (code tests / constraints / schema / consistency)
      pass → accept
      fail → feed the failure back to the 176B, local retry (1–2×)
              pass → accept (repair wins)
              fail → hosted API (escalate)
  no verifier → gate score / family prior → accept or API
```

This is the genuine "recompute again, trying a different way": the second pass
gets **new evidence** (the failing test, the violated constraint), not another
opinion about the same state. It is also cheap — one more local generation per
failure — and it uses exactly the benches that ship gold-free checks.

## Pipeline and guardrails

1. **Ingest:** map each bench into the `tasks-*.json` schema
   (`id, benchmark, bucket, prompt, checker, split, meta`), adding `family` and
   `verifier_class`. Extend `experiments/cascade/bench_tasks.py` builders.
2. **Answer:** generate 176B candidates in agreed serve windows; record model,
   temperature, seed and timing. The 0.8B feature extraction stays ~0.5 s/item.
3. **Label:** run the verifier; for gold-dependent benches the label is for
   training/eval only. Keep the gold out of the router's runtime input.
4. **Split:** freeze test benches; family-held-out folds (leave-one-bench-out)
   for transfer claims; never train on a test split.
5. **Sandbox:** code execution needs no-network, CPU/memory/time limits; a
   checker bug is a router error, so test the checkers themselves.
6. **Decide:** report verifier-first catch/FR/API-call counts, repair success,
   gate prefilter value, and family-level routing priors — not just gate AUC.

## Next step

Finish the ingest for the starter pack (MBPP, IFEval, AIME, LiveCodeBench,
MMLU-Pro), pick a serve window, and run the first hard-subset harvest:
HumanEval 164 + MBPP 200 + IFEval 200 + AIME 90 + LiveCodeBench 200. That is
~850 answers — a few hours of 176B time — and enough to measure whether the
repair loop and the gold-free verifiers earn their place before adding the long
tail.
