# Decision-head research — handoff (2026-10-06, new session start)

**This replaces the Clef/dense ternary lane.**  The community-quant thread
(Clef-Flash rotation/GPTQ/QAT) is canned: its artifact is not postable (free
generation degrades; a plain Q2_K beats it) and the decision signal is
recoverable far more cheaply by reading the model's own states.  The new path
is: **decision heads — fitted readouts on a model's hidden states — as the
judge/chooser inside a sampling loop, tested across low-bit models.**

Everything below is measured, committed and reproducible from this directory
(`hivebench/experiments/cascade/results/jev-readout-20261006/`).

---

## 1. The thesis in one paragraph

A model's hidden state already contains a linearly-readable verdict on whether
a candidate answer is correct.  A closed-form linear head over that state
(≈66 KB, fit on 20–300 labelled examples, no fine-tuning, no second model)
matches or beats purpose-trained decision models, is attribution-neutral when
fitted in the serving format, and turns sampling into accuracy when used as
the chooser in a generate → re-read → select loop.  Asked in words, the same
model catches 7/28 of its own errors; the head catches 23/28.  Knowledge lives
in the model; the head listens; the controller acts.

## 2. Headline results

### 2a. Frozen judge benchmark (J-Bench)

`judge-bench.jsonl` — 378 items (97 positives / 281 checker-verified
negatives), splits 68/164 train + 29/87 test, sha256 `3289dcb8…`; built by
`experiments/cascade/judge_bench.py` from cascade-bench-v1 (100 tasks,
GSM8K/MATH/HumanEval/TruthfulQA, programmatic checkers).  Test split frozen.

### 2b. Readout matrix (test split, AUC / best BA)

| readout | model | AUC | BA |
|---|---|---|---|
| **probe, final layer** | Scion-35B-A3B (2-bit ternary MoE) | **0.999** | **0.994** |
| probe, depth 0.75 | Qwen3-1.7B (instruct) | **0.988** | 0.954 |
| probe, final | Qwen3.5-4B (instruct) | 0.960 | 0.937 |
| Intern-Decision-4B (trained judge) | — | 0.979 | 0.937 |
| probe, depth 0.75 | Qwen3.5-0.8B (instruct) | 0.924 | 0.856 |
| Intern-Decision-2B (trained) | — | 0.932 | 0.856 |
| Tiny-Jev-1.7B (trained) | — | 0.890 | 0.862 |
| Intern-Decision-0.8B (trained) | — | 0.774 | 0.759 |
| **prompted yes/no** | Qwen3.5-4B (instruct) | **0.494** | 0.546 |
| prompted self-judge | Scion-35B-A3B | 0.903 | 0.925 |

Label budgets: 4B saturates at ~20 labels (0.959); 0.8B scales
0.784 → 0.831 → 0.924 at 20/100/262.  Cross-type transfer (fit on number-flips
only): boxed 0.998, code 0.874 — a general correctness signal, not
perturbation detection.

### 2c. The generator checking its own answers (Scion-35B-A3B)

Same model, same 100 generated answers (72 correct / 28 wrong):

| | catches own errors | keeps correct | AUC |
|---|---|---|---|
| prompted self-check | **7 / 28** | 72/72 | 0.559 |
| head (thr 0.5) | **23 / 28** | 72/72 | 0.971 |
| head (thr 0.87 calibrated) | 26 / 28 | 67/72 | 0.971 |

Frozen head on natural (unperturbed) errors: AUC 0.971, mean score 1.04
correct vs 0.22 wrong (`source-audit.json`).  Full before/after write-up:
`SCION-HEAD-REPORT.md`.

### 2d. Self-bias / attribution audits

- Uncontrolled source swap (different texts): +0.078 correct / +0.018 wrong —
  style, not authorship.
- Controlled attribution (identical text, own vs other wording): an
  **out-of-format** head shows +0.178/+0.098, but **in-format refits are
  attribution-neutral** (baseline −0.003/−0.013; balanced +0.007/−0.013 over
  60 held-out candidates).  The head is format-locked; fit it on the serving
  prompt.  The prompted judge keeps a real +0.099 preference on its own wrong
  answers (`attribution-audit.json`, `attribution-fix.json`).

### 2e. Quantization survival (T3, final layer, Clef bodies)

| body | PPL | refit head AUC | f16 head frozen |
|---|---|---|---|
| Clef f16 (Qwen3.5-9B dense) | 12.59 | 0.952 | — |
| Clef QAT (2.1 bpw ternary) | 23.25 | 0.712 | 0.502 |
| Clef h128 GPTQ | 120.6 | 0.648 | 0.510 |

Signal partially survives 2-bit (QAT > h128, tracks PPL) but heads do not
transfer across bodies.  Literature: 4-bit probes stay 0.90–1.00; 2-bit is
where readout fidelity breaks down.

### 2f. The loop: selection, retry, re-read

Correction arms (100 tasks × 4 samples, temp 0.7; `correction-eval.json`):

| arm | correct | acc | tokens/correct |
|---|---|---|---|
| single | 76/100 | 0.760 | 490 |
| majority of 4 | 80/100 | 0.800 | 1,854\* |
| best-of-4 head (frozen) | 81/100 | 0.810 | 1,831 |
| retry @0.5 | 80/100 | 0.800 | 687 |
| retry @0.87 | 83/100 | 0.830 | 739 |

\* Honest pool cost (148,333/80).  Earlier revisions charged the majority vote
only the picked candidate (459); T1's 464 was the same undercount.  Both arms
pay the 4-sample pool; the retry rows are sequential-policy costs.

Held-out (30 test tasks) with the **in-domain** head refit on trace-inclusive
prompts (`selection-frontier.json`):

| policy | correct | acc | tokens/correct |
|---|---|---|---|
| best-of-1 | 21/30 | 0.700 | 576 |
| best-of-2 | 23/30 | 0.767 | 1,027 |
| **best-of-3** | **25/30** | **0.833** | **1,441** |
| best-of-4 | 25/30 | 0.833 | 1,901 |
| gated (stop when confident) | 23/30 | 0.767 | 646 (mean 1.17 samples) |
| oracle (pool contains one) | 27/30 | 0.900 | — |

Retry kind, conditional on a wrong first answer (24 tasks):

| retry kind | fixed |
|---|---|
| **blind full re-derivation** | **12/24** |
| hinted (shown the flagged answer) | 9/24 (anchors: first-shot 4 vs 7 blind) |
| answer-only (reasoning kept, final step redone) | **1/24** |

Wrong answers cost ~60 % more tokens (511 vs 320) and lock in earlier — errors
are committed upstream; patching the tail reproduces them; showing the model
its error anchors it.  If you retry, re-derive fully and blindly.

### 2g. Mid-generation monitor: corrected — verifier, not canceller

The original "negative" was a file-sort misalignment (`win_%02d` string sort),
and the dumps later needed an adapter fix; both are documented in
`MONITOR-FIX.md`.  Corrected trajectory GRU: read-back .832/.825,
generation-context .785/.792/.812/.932 (Scion); Qwen context .866/.829/.863/
.994 — the signal is real in context.  What dies is the use-case: an
early-cancel simulation on held-out tasks (validation-calibrated thresholds)
shows an oracle ceiling of only 21–34% token savings at k=64–256, with
realistic points at 10–19% savings against 3–15% false cancels.
**The monitor is an end-of-trace verifier/selector, not a generation-time
canceller.**  `MONITOR-FIX.md`, `monitor-context-scion.json`,
`cancel-sim-*.json`, `traj-select-*.json`.

### 2h. Delivered artifact

`scion-35b-a3b-decision-head.json` (66 KB) — model id, prompt format, dim
(2048), ridge weights + standardisation + Platt, training provenance (bench
sha), test metrics (AUC 0.9988, BA 0.994, threshold 0.87).  Apply:
`p = sigmoid(a·(((h−mu)/sd)·w + b) + c)` on the last-prompt-token state from
`/embedding` (`--embeddings --pooling last --embd-normalize -1`).

### 2i. Bonsai dense test (2026-10-07): trained-low-bit dense transfer

PrismML `Ternary-Bonsai-2-27B` (Q2_0, 2.13 bpw, trained ternary dense;
7,206,168,928 B, sha `3907dc16…`), served by the moe-corr fork (dim 5120; the
in-tree Prism build predates the codec and rejects type 142).  4 samples × 100
tasks at the card's thinking sampling (temp 1.0 / top_p .95 / top_k 20 /
min_p .05), checker labels, train-fit heads.

- J-Bench probe test: **AUC 0.988 / BA 0.983** (Scion 0.999; Clef QAT 0.712,
  h128 0.648) — the readout survives trained 2-bit ternary in a dense body.
  The Clef collapse was post-hoc quant, not bit-width or architecture.
- Own answers: 93/100 first-sample correct (7 natural errors; failures are
  long — 3,040 vs 510 tokens).  Prompted self-check catches **2/7** (keeps
  91/93); head catches **6/7** (J-Bench fit 92/93 kept; in-domain 93/93);
  all-400 per-sample AUC 0.999 in-domain / 0.868 J-Bench-fit.
- Selection (full 100): single 93 → best-of-2/4 **96 = oracle**; gated head
  96/100 @ 1.12 samples (102.6k tok); J-Bench-head gated 96 @ 1.06 (78.1k).
  Held-out 30: single 28 = oracle 28 (both errors have no correct sample
  in-pool) — no measurable held-out selection gain.
- Retry (7 wrong-first): blind 3/7 vs hinted 2/7; anchor case `math-0021`
  (blind 3/3, hinted 0/2).  In-format balanced attribution +0.005 / +0.012
  (neutral).  Full report: `BONSAI-HEAD-REPORT.md`.

### 2j. Community arms — Scion (2026-10-07)

664-task set (`tasks-community.json`: GSM8K-300 / MATH-200 / HumanEval-164;
465 train / 199 test; sha `83baf6f3…`), built offline from the cached datasets.
Scion 3 samples/task (temp 0.7, non-thinking) + 1 think-harder (thinking,
2,048).  Held-out 199: single 142, think-harder 150 (2.2× tokens), gated 152
(in-domain) / 154 (J-Bench) at 1.3–1.5×, best-of-3 149 / **156** at ~2.7×,
majority 154 at equal tokens, oracle 169.  Frozen J-Bench head ≫ in-domain
refit on this mix (all-sample AUC 0.86 vs 0.66; keeps 140/142 vs 120/142) —
in-domain fitting on the community train mix overfits (train AUC 1.0).
Headroom is in MATH/GSM8K; code is hard.  Report:
`SCION-COMMUNITY-REPORT.md`.

Bonsai community (head-only, card sampling) completed 2026-10-08: single 168 →
gated/best-of-3 173–176 (frozen J-Bench head), majority 168, oracle 178; the
veto validated (frozen 173 with **5 fixes / 0 breaks**; raw-margin variant
174, 6/0), pooled Scion+Bonsai override evidence **14 fixes / 2 breaks over
398 held-out tasks, exact p=.004**.  Raw answer-slot margin readout added
(Scion test AUC .880, Bonsai .858; no universal winner vs the fitted head, so
per-model validation picks).  Report: `BONSAI-COMMUNITY-REPORT.md`.

---

## 3. Where everything lives

Repo: `hivebench` (branch `hive_cascade`), HEAD `d5e5825`.
Results dir: `experiments/cascade/results/jev-readout-20261006/`

| file | role |
|---|---|
| `HANDOFF.md` (this) | session start |
| `PRIOR-ART.md` | literature scan (AnyJev, LLM-as-Jev, Kadavath, Gnosis, SAPLIR/Aiersilan, Lu, Panickssery) |
| `PLAN.md` | experiment ladder + results ledger |
| `WHITEPAPER-OUTLINE.md` | paper skeleton with claims/caveats |
| `SCION-HEAD-REPORT.md` | before/after on the released Scion model |
| `judge-bench.jsonl` + `judge-bench.manifest.json` | frozen benchmark (sha256) |
| `COMPARISON.md` | unified readout/judge table |
| `correction-eval.json`, `retry-policy.json`, `selection-rerank.json`, `selection-frontier.json` | loop results |
| `hinted-eval.json`, `answer-retry-eval.json`, `reread-variant.json` | retry/re-read results |
| `source-audit.json`, `attribution-audit.json`, `attribution-fix.json` | self-bias audits |
| `monitor-*.json` | mid-generation negative |
| `transfer-*.json`, `probe-clef-*.json` | ternary survival |
| `scion-35b-a3b-decision-head.json`, `scion-answers.json` | exported head; 100 own answers |
| `BONSAI-HEAD-REPORT.md` | before/after on Ternary-Bonsai-2-27B (dense 2-bit) |
| `probe-bonsai27b.json`, `bonsai-eval.json`, `bonsai-summary.json` | Bonsai results |
| `bonsai-2-27b-decision-head.json`, `bonsai-indomain-head.json` | Bonsai heads (164 KB, dim 5120) |
| `attribution-bonsai-audit.json`, `attribution-bonsai-fix.json` | Bonsai attribution |
| `readout-bonsai27b-prompted.json`, `readout-bonsai-own-prompted.json` | Bonsai prompted baselines |
| `run-bonsai.sh`, `run-attribution-bonsai.sh` | Bonsai session chains |
| `run-*.sh` | one-shot chains (small sweep, judges, scion session, source audit, attribution, ternary, monitor, correction, hinted, answer) |
| probe-state `.npz`, `streams/scion-bench-monitor/` (2.5 GB dump) | local caches, gitignored; regenerable |

Scripts (in `experiments/cascade/`): `judge_bench.py`, `readout_judge.py`,
`probe_judge.py`, `probe_scion.py`, `score_judges.py`, `compare_readouts.py`,
`gen_answers.py`, `export_head.py`, `source_audit.py`, `attribution_audit.py`,
`attribution_fix.py`, `correction_eval.py`, `retry_policy.py`,
`selection_rerank.py`, `hinted_retry.py`, `answer_retry.py`,
`reread_variants.py`, `monitor_probe.py`, `monitor_diag.py`,
`monitor_token_diag.py`, `build_monitor_windows.py`, `transfer_eval.py`.

**Artifacts on disk (outside the repo):**
- Scion model: `work/scion/Scion-35B-A3B-PQ2_0-corr.gguf`, 11,336,908,288 B,
  sha256 `545d83a93c6ad147…` (identical to `SkyIsNotGreen/Scion-35B-A3B`).
- Bonsai model: `~/Downloads/Ternary-Bonsai-2-27B-PQ2_0.gguf`,
  7,206,168,928 B, sha256
  `3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1` — served
  by the moe-corr fork (it carries `GGML_TYPE_PQ2_0 = 142`); the in-tree
  Prism build rejects the file.
- Fork: `~/llama.cpp` (`moe-corr-runtime`; `llama-server`,
  `test-mtp-probe`, build `0.2.0-dev build 10721`).
- Prism build: `work/prism-ml-llama.cpp/build-hip/bin/` — predates the
  Bonsai-2 ternary codec (build 11248; no type 142), so it is **not** the
  runtime for the Q2_0 file.
- venv: `<work>/.venv-rocm/bin/python` (transformers 5.5.0,
  torch 2.14+rocm).
- HF cache: `Qwen3.5-0.8B`, `Qwen3-1.7B`, `Qwen3.5-4B` (all **instruct**),
  `empero-ai/Qwen3.8-35B-A3B-Distill` (Scion's base), OLMoE-1B-7B, judges
  (Intern 0.8/2/4B, Tiny-Jev-1.7B, JEV-9B), `prism-ml/Ternary-Bonsai-*`
  (cache dirs exist but are **empty** — never downloaded).

**Serving recipe used throughout:**
`llama-server -m <gguf> -ngl 99 -c 8192 --host 127.0.0.1 --port 8897
--embeddings --pooling last --embd-normalize -1 -fa auto`
Then `/apply-template` (thinking disabled via
`chat_template_kwargs: {"enable_thinking": false}`) → `/embedding` for states,
`/v1/chat/completions` for generation, `/completion` with `n_probs` for logit
readouts.

**Repo hygiene:** `scion` was restructured by a parallel session (ops docs
removed; `dense/` is documented in `scion/README.md` + `scion/dense/README.md`
only).  Don't resurrect the old handoffs.  The local claims file
`.claims/JEV-READOUT.json` (gitignored) covers commits to the results dir and
the new scripts.

---

## 4. Planned next: the Bonsai dense test

**Done (2026-10-07)** — verdict in §2i and `BONSAI-HEAD-REPORT.md`: the head
transfers to the trained 2-bit ternary dense body (J-Bench 0.988); selection
reaches the pool oracle on the full bench but the frozen 30-task split is
degenerate; in-format attribution-neutral.  Community arms: Scion done (§2j,
`SCION-COMMUNITY-REPORT.md`), Bonsai head-only running; the sidecar and
whitepaper carry forward.

Question: does the decision-head loop transfer from our trained-corrections
MoE to **PrismML's Ternary-Bonsai-2-27B** — the *trained-low-bit dense*
counterpart this line of work descends from?  If it works on both, the
dense+MoE, trained-low-bit cases are covered; a mainstream-Qwen benchmark row
becomes an appendix rather than a prerequisite (readout-level evidence on
standard Qwens 0.8B/1.7B/4B already exists).

1. **Fetch the model** (needs operator approval; cache is empty):
   `prism-ml/Ternary-Bonsai-2-27B-gguf`, Q2 file only, to
   `work/models/bonsai-2-27b/` (external drive for scratch).
2. **Runtime check:** confirm which build serves the Prism Q2_0 g128 codec with
   `/apply-template`, `/embedding` (our fork vs
   `prism-ml-llama.cpp/build-hip`); run one `/embedding` smoke and record dim.
3. **Same protocol as Scion:** 100 cascade-bench tasks, 4 samples/task at the
   card's recommended sampling, checker labels; fit the head on the 70 train
   tasks *in the serving format*; evaluate on the 30 test tasks —
   best-of-1..4 frontier, gated sequential, blind-vs-hinted retry, and the
   natural-error catch rate on Bonsai's own answers.
4. **Success criteria:** head AUC ≥ 0.90 on candidate verification; selection
   gain over single ≈ the Scion reference (76→~81+, oracle-gap capture);
   in-format attribution-neutral.  If yes → generalize-thesis complete; write
   the paper.  If no → isolate whether it is codec (embedding quality under
   g128), architecture (dense vs MoE), or training regime.
5. **Then:** the community-benchmark arms (A baseline / B think-harder /
   C head-gated matched budget / D best-of-3; GSM8K-300, MATH-200,
   HumanEval-164) and the MTP-style sidecar prototype + whitepaper draft.

---

## 5. Standing constraints

- **Never `git push`; commit locally.**  hivebench commits run the claims gate
  (`.claims/JEV-READOUT.json` already covers this directory + scripts).
- **Ask the operator before downloads, heavy runs, and any rental.**
- One heavy track at a time; non-display GPU only (`HIP_VISIBLE_DEVICES=1`,
  check `0000:07:00.0/power/runtime_status == active`); long runs under
  `systemd-run --user --unit=…` (app-scope jobs die); `/tmp` is a 16 GB tmpfs —
  scratch to the external drive or `/tmp/opencode`.
- Never post to HF; never post the dead sidecar corrections.
- The box wedged the non-display GPU once during a mixed generation+embedding
  session (`INCIDENT-GPU-HANG.md`); keep sessions checkpointed and expect a
  reboot to recover if it happens again.

---

## 6. Open questions (for the paper / next session)

- ~~Does Bonsai dense behave like Scion MoE?~~ Answered 2026-10-07: yes at the
  readout level (J-Bench 0.988 vs 0.999), so the signal is not MoE- or
  codec-specific; but its near-ceiling accuracy leaves no held-out selection
  headroom (single 28/30 = oracle), so the *selection* claim needs a generator
  with recoverable errors.
- Can selection close more of the oracle gap?  Bonsai closed 100 % of the
  full-bench gap (93 → 96 = oracle) with the gated policy at 1.12 samples, but
  the frozen 30-task split had nothing to close.  The open question is now
  *when* the loop pays — as a function of generator error rate and pool
  diversity — not whether the head can rank.
- Label budget on the target model vs the 4B=20 / 0.8B=262 datapoints.
- Sidecar packaging (head tensors + format hash loaded by the engine) vs the
  current client-side JSON.
- Mainstream-model benchmark row: needed for the community-benchmark claim,
  deferrable for the method claim.

---

## 7. Close (2026-10-08)

All planned tests are complete: veto replication (Scion + Bonsai), the raw
answer-slot-logit baseline, the early-cancel simulation, trajectory-based
selection, sidecar GPU parity, and the embedded-adapter fix (both dump tools).
Final scorecard and verdict: `CONCLUSION.md`.  Monitor record:
`MONITOR-FIX.md`.  Nothing published or pushed; the fork changes are committed
locally; every artifact needed to regenerate the results is in this directory.
