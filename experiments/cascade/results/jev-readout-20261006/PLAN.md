# JEV-readout test plan (2026-10-06)

Question: for the cascade verifier role, what is the **smallest footprint** that
answers "is this candidate correct?" well enough — and does it have to be an
independent model at all?

Frozen set: `judge-bench.jsonl` (378 items: 97 pos / 281 neg; split 68/164
train, 29/87 test; sha256 `3289dcb8...`; all items checker-verified).  The test
split is frozen; fitting uses train only.  Prior art: `PRIOR-ART.md`.

## Experiments

| # | test | readout | status |
|---|---|---|---|
| T1a | generator judges J-Bench (self) | prompted (yes/no logits, both orders) | queued (Scion-35B session) |
| T1b | generator judges J-Bench (self) | final-layer linear probe | queued (bridge hidden dump) |
| T1c | independent model judges J-Bench | prompted + probe | running (Qwen3.5-4B / 0.8B; Intern/JEV later) |
| T1d | self-preference | Scion's own vs Strata-answer accept rates | queued (needs Scion answers on the test split) |
| T2 | size/label curve | probe across 0.8B / 1.7B / 4B / 9B, labels 20/100/300 | 0.8B smoke: AUC 0.973 (final), 0.898 (mid) on 120 items |
| T3 | ternary survival (novel) | probe fitted on bf16 states -> ternary states; refit | blocked on mid-layer extraction for ternary GGUF |
| T4 | calibration + cascade end-to-end | threshold/no-leak point, then `run_cascade.py` | after a candidate passes |

## Gates

- G1: if T1a/T1b (generator self) match T1c within 3 points BA and no worse
  false-accepts -> no independent model needed.
- G2: else the smallest independent base + probe that clears the bar wins
  (literature predicts ~4B-class).
- G3: if a sub-1B model must be quantised for residency, T3 decides whether the
  readout survives.

## Results (2026-10-06, frozen test split unless noted)

Headline table — see `COMPARISON.md` for the live version:

| readout | kind | test AUC | best BA | note |
|---|---|---|---|---|
| Intern-4B | trained judge | 0.979 | 0.937 | the quality bar (purpose-trained) |
| **Scion-35B final-layer probe** | probe | **0.999** | **0.994** | the generator itself; no-release threshold 0.95 -> acc 0.974 |
| Qwen3.5-4B final-layer probe | probe | 0.960 | 0.937 | base model + fitted head matches the trained 4B |
| Scion-35B prompted self-judge | prompted | 0.903 (test) | 0.925 | instruct model; order-flip 7.4%; better than trained 0.8B/1.7B |
| Qwen3.5-0.8B mid-layer probe | probe | 0.924 | 0.856 | beats every small trained judge |
| Intern-2B | trained judge | 0.932 | 0.856 | |
| Tiny-Jev-1.7B | trained judge | 0.890 | 0.862 | |
| Intern-0.8B | trained judge | 0.774 | 0.759 | |
| Qwen3.5-4B prompted | prompted | 0.494 | 0.546 | base model: 82.8% option-order flips (label bias) |

Other measurements:

- **Cross-type transfer** (4B probe, depth 0.75, fit on number-flips only):
  boxed 0.998, code 0.874 — a general correctness signal, not perturbation
  detection.
- **Natural negatives**: Scion generated its own answers for all 100 bench
  tasks (`scion-answers.json`): **72/100 correct -> 28 natural errors**, five
  times the natural negatives in the recorded runs (3).
- **Prompted self-preference (paired 100 tasks)**: Scion's prompted judgment of
  **its own answers is near-chance as a discriminator (AUC 0.559, 21/28 wrong
  accepted)**, while the same prompt on Strata's answers scores AUC 0.371 on
  only 3 available negatives (uninformative).  The probe-vs-prompted gap on
  the same model is the story: 0.999 vs 0.559.
- **Frozen-head source-swap audit (done, `source-audit.json`)**: the head
  fitted on Strata-answer train items, applied frozen to:
  - Scion's **own** answers: AUC **0.971** over the same 100 tasks including
    **28 natural errors** (mean score 1.04 correct vs 0.22 wrong) — the
    correctness signal generalizes beyond perturbations;
  - Strata's answers: AUC 0.990 (97 correct / 3 wrong).
  Uncontrolled self-preference deltas: correct +0.078, wrong +0.018.
- **Self-bias / attribution (done, `attribution-audit.json` + `attribution-fix.json`)**:
  the uncontrolled source swap (different texts) showed +0.078 correct /
  +0.018 wrong — style, not authorship.  The controlled test (identical text,
  own vs other) first appeared to show **+0.178 correct / +0.098 wrong** for a
  head fitted on the J-Bench wording — but that head is **format-locked**:
  re-fitting it *in the audit's own wording* removes the effect entirely.
  Baseline (own-attribution-only fit): delta_correct **-0.003**, delta_wrong
  **-0.013**; balanced (both attributions): **+0.007 / -0.013** — over 60
  held-out candidates (51 correct / 9 wrong).  In-domain, the head is
  **attribution-neutral**; the prompted judge still shows a real **+0.099** on
  its own wrong answers.  Deployment rule: fit the head on the serving prompt
  format (AnyJev's per-layout heads).  Caveat: wrong-answer deltas rest on 9
  held-out negatives.

### T2 size / label map (probe at depth 0.75, test AUC)

| base | 20 labels | 100 labels | full train (262) |
|---|---|---|---|
| Qwen3.5-0.8B | 0.784 | 0.831 | 0.924 |
| Qwen3-1.7B | — | — | **0.988** |
| Qwen3.5-4B | 0.959 | 0.962 | 0.946 |

The 1.7B is the surprise: its mid-layer probe beats the 4B and matches the
purpose-trained Intern-4B (0.979), with a usable no-false-accept point
(thr 0.94 -> acc 0.948).  The 4B is label-saturated at ~20 (0.959 vs Tiny-Jev
0.890 / Intern-2B 0.932).  The 0.8B keeps scaling with labels.



### T3 ternary survival (final layer, same prompts and items)

| body | PPL | refit head (test AUC / BA) | f16 head frozen |
|---|---|---|---|
| Clef f16 | 12.59 | **0.952 / 0.914** | — |
| Clef QAT (2.1 bpw) | 23.25 | **0.712 / 0.684** | 0.502 |
| Clef h128 GPTQ | 120.6 | **0.648 / 0.667** | 0.510 |

Cross-body transfer: the f16 head is at chance on both ternary bodies
(0.502 / 0.510); QAT->f16 transfers at 0.866, h128->f16 at 0.241.  So at 2-bit
the decision signal *partially* survives (QAT > h128, tracking PPL) but the
**direction does not transfer**: each body needs its own head, and refit
recovery at the final layer is 0.65-0.71 vs 0.95 for f16.  Prior work reports
4-bit probes keep 0.90-1.00 AUROC; 2-bit is where readout fidelity breaks
materially.  Mid-layer extraction for ternary bodies (fork graph hook) is the
open follow-up; the QAT body is the better platform (0.712).

### L1/L2 correction + L3 monitor (2026-10-06)

Selection/retry arms (100 tasks x 4 samples, temp 0.7; `correction-eval.json`):

| arm | correct | acc | tokens | tokens/correct |
|---|---|---|---|---|
| B0 single sample | 76/100 | 0.760 | 37,234 | 490 |
| B1 majority of 4 | 80/100 | 0.800 | 36,747 | 459 |
| **T1 best-of-4 head** | **81/100** | **0.810** | 37,576 | **464** |
| T2@0.5 head retry | 80/100 | 0.800 | 54,927 | 687 |
| **T2@0.87 head retry** | **83/100** | **0.830** | 61,344 | 739 |

Selection gives **+5 points at the same token budget** and beats majority vote
at equal tokens; retry gives +7 points at +50-60% tokens.

**L3 online monitor: negative.**  Per-position hidden states from the same 399
streams (`test-mtp-probe`) predict final correctness at **AUC 0.55-0.62 at
every horizon** (start 64-512, midpoint, last tokens; `monitor-diag.json`),
and token-level tail statistics (switch rate, stability) combine to only
**0.67** (`monitor-token-diag.json`).  Early-abort gates at start horizons
burst false-cancels (e.g. 59/80 correct streams at 128 tokens;
`monitor-probe.json`).  **The decision signal is not on the generation path**:
it materialises when the candidate is read back in a verification prompt,
where the same model scores 0.97-0.999.  Verification is a second, cheap
*read* (one prefill, zero generated tokens), not a state continuation -- which
is why a lite two-pass works while "monitor while writing" does not, with a
linear head.  A gradient-trained trajectory monitor (Gnosis-style) is the
remaining option if the online path is ever pursued.



### Retry & re-read (2026-10-06, second block)

**Re-read: fit the head on the serving format.**  A head refit on
trace-inclusive prompts (reasoning tail + answer) beats the frozen
answer-only head for pool selection: best-of-3/4 = **25/30** vs 22/30
(oracle 27/30); best-of-2 = 23/30.  Per-sample AUC is unchanged (0.84-0.85);
the gain is within-pool ranking from in-domain fitting.  Frontier
(`selection-frontier.json`): best-of-1 21/30 (576 tok/correct), best-of-2
23/30 (1,027), best-of-3 **25/30 (1,441)**, best-of-4 25/30 (1,901).

**Retry: only full blind re-derivation works.**  Conditional on a wrong first
answer (24 tasks, all splits):

| retry kind | fixed |
|---|---|
| blind full resample (any of 3 later samples) | **12/24** |
| hinted full retry (shown the flagged answer) | 9/24 |
| answer-only retry (reasoning kept, final step redone) | **1/24** |

Test-split arms: single 21/30, gated answer-retry 21/30 (+440 tokens).
Showing the model its own wrong answer *anchors* it (first-shot fix 4/24 vs
blind 7/24), and the error is not localized in the final step -- it is
committed upstream, so a tail patch reproduces it.  If you retry, re-derive
fully and blindly.

**Product loop:** generate 3 samples, rank with the in-domain head (plus
canonical agreement), accept; retry only when the whole pool is flagged low.
Steering/monitoring remains dead (L3); selection is the cheap lever.

### Bonsai dense test (2026-10-07, trained 2-bit ternary dense)

PrismML `Ternary-Bonsai-2-27B` Q2_0 (7,206,168,928 B, sha `3907dc16…`),
served by the moe-corr fork (dim 5120; the in-tree Prism build predates the
codec and rejects type 142).  4 samples × 100 tasks at the card's thinking
sampling (temp 1.0 / top_p .95 / top_k 20 / min_p .05, xhigh default), checker
labels, train-fit heads.

| measurement | Bonsai | Scion ref |
|---|---|---|
| J-Bench probe test AUC / BA | **0.988 / 0.983** | 0.999 / 0.994 |
| prompted J-Bench test AUC | 0.952 | 0.903 |
| own answers (first sample) | 93/100 correct, 7 errors | 72/100, 28 errors |
| prompted self-check catches | 2/7 (keeps 91/93) | 7/28 |
| head catches (J-Bench fit / in-domain) | **6/7 / 6/7** | 23/28 |
| all-400 per-sample AUC (in-domain) | 0.999 | — |
| full-set selection single → best-of-2/4 | 93 → **96 = oracle** | 76 → 81 (oracle 83) |
| gated sequential | **96/100 @ 1.12 samples** (102.6k tok) | 83/100 @ 1.47 (61.3k) |
| held-out 30 selection | 28/30 single = 28/30 oracle (degenerate) | 21/30 → 25/30 (oracle 27) |
| blind vs hinted (7 wrong-first) | 3/7 vs 2/7 (anchor: math-0021 3/3 vs 0/2) | 12/24 vs 9/24 |
| in-format balanced attribution | +0.005 correct / +0.012 wrong | +0.007 / −0.013 |

Errors are long: failed first answers average 3,040 tokens vs 510 correct.

Verdict: the readout transfers to the trained-low-bit dense body (0.988 ≫ the
0.90 bar; dense vs MoE immaterial); the Clef 2-bit collapse (0.712 / 0.648) was
the post-hoc quant regime, not the bit-width.  Selection reproduces the
mechanism (the head reaches the pool oracle) but Bonsai is near-ceiling: on the
frozen 30 test tasks single already equals the oracle, so held-out gain is
unmeasurable.  Report: `BONSAI-HEAD-REPORT.md`; artifacts: `probe-bonsai27b.json`,
`bonsai-eval.json`, `bonsai-summary.json`, `bonsai-2-27b-decision-head.json`,
`bonsai-indomain-head.json`, `run-bonsai.sh`, `run-attribution-bonsai.sh`.

## Status and next steps

- Done: J-Bench; the full readout/judge/probe matrix; Scion-35B session;
  source-swap audit incl. natural errors; T2 size/label map; T3 ternary
  survival; controlled attribution + balanced refit; exported
  `scion-35b-a3b-decision-head.json` for the released model with the
  before/after report (`SCION-HEAD-REPORT.md`): prompted **7/28 -> head
  23/28** own-error catches at zero false rejects (thr 0.5); L1/L2 correction
  arms (selection +5 pts at equal tokens); L3 monitor negative.
- Done: Bonsai dense test (2026-10-07): J-Bench probe **0.988 AUC** on the
  trained 2-bit ternary dense body; prompted self-check 2/7 vs head 6/7 of its
  own errors; selection reaches the pool oracle on the full bench (93→96) but
  the frozen 30-task split is degenerate; in-format attribution-neutral;
  `BONSAI-HEAD-REPORT.md`.  Generalization verdict: trained-low-bit readout
  survives across dense+MoE; post-hoc 2-bit was the Clef failure mode.
- Next: community-benchmark arms (A baseline / B think-harder / C head-gated
  matched budget / D best-of-3; GSM8K-300, MATH-200, HumanEval-164), the
  MTP-style sidecar prototype, and the whitepaper draft.  A mainstream-model
  benchmark row remains the optional appendix.



