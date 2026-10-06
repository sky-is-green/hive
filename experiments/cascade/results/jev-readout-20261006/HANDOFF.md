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
| majority of 4 | 80/100 | 0.800 | 459 |
| best-of-4 head (frozen) | 81/100 | 0.810 | 464 |
| retry @0.5 | 80/100 | 0.800 | 687 |
| retry @0.87 | 83/100 | 0.830 | 739 |

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

### 2g. Mid-generation monitor: negative

Per-position hidden states of 399 generation streams predict final correctness
at AUC 0.55–0.62 (any position: start, midpoint, last token); token-level tail
statistics combine to 0.67; abort gates flag most correct streams.  **The
verdict is not on the generation path; it materialises when the candidate is
read back in a verification prompt.**  `monitor-probe.json`,
`monitor-diag.json`, `monitor-token-diag.json`.

### 2h. Delivered artifact

`scion-35b-a3b-decision-head.json` (66 KB) — model id, prompt format, dim
(2048), ridge weights + standardisation + Platt, training provenance (bench
sha), test metrics (AUC 0.9988, BA 0.994, threshold 0.87).  Apply:
`p = sigmoid(a·(((h−mu)/sd)·w + b) + c)` on the last-prompt-token state from
`/embedding` (`--embeddings --pooling last --embd-normalize -1`).

---

## 3. Where everything lives

Repo: `hivebench` (branch `hive_cascade`), HEAD `a2937a5`.
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
- Fork: `/home/penis/llama.cpp` (`moe-corr-runtime`; `llama-server`,
  `test-mtp-probe`, build `0.2.0-dev build 10721`).
- Prism build (serves Prism codecs): `work/prism-ml-llama.cpp/build-hip/bin/`.
- venv: `/home/penis/Desktop/work/.venv-rocm/bin/python` (transformers 5.5.0,
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

- Does Bonsai dense behave like Scion MoE (planned above)?
- Can selection close more of the oracle gap (80 % open on the frozen head;
  in-domain refit closed 60 %; agreement ties the head on MC-style items)?
- Label budget on the target model vs the 4B=20 / 0.8B=262 datapoints.
- Sidecar packaging (head tensors + format hash loaded by the engine) vs the
  current client-side JSON.
- Mainstream-model benchmark row: needed for the community-benchmark claim,
  deferrable for the method claim.
