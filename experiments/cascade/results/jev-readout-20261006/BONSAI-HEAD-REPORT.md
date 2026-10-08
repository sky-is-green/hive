# Ternary-Bonsai-2-27B + decision head — before / after (2026-10-07)

**Model:** `prism-ml/Ternary-Bonsai-2-27B-gguf`
(`Ternary-Bonsai-2-27B-PQ2_0.gguf`, 7,206,168,928 B, sha256
`3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1`) — the
trained 2-bit ternary **dense** model (Qwen3.8-27B backbone, hybrid attention,
2.13 bpw, Hadamard-rotated g128 weights), the trained-low-bit counterpart of
the Scion-35B-A3B MoE.
**Runtime:** moe-corr fork (`/home/penis/llama.cpp`, llama.cpp 0.2.0-dev build
10721), `llama-server --embeddings --pooling last --embd-normalize -1`; hidden
dim **5120**.  The in-tree Prism build (`work/prism-ml-llama.cpp`, build 11248)
predates the codec and rejects the file (`tensor 'output.weight' has invalid
ggml type 142`); the fork loads it and generates coherently.
**Heads:** `bonsai-2-27b-decision-head.json` (J-Bench train fit) and
`bonsai-indomain-head.json` (train-task sample fit, answer candidate), 164 KB
each — closed-form ridge + Platt on the last-prompt-token state.  No
fine-tuning, no second model, no second inference pass.
**Protocol:** 100 cascade-bench-v1 tasks (70 train / 30 test), 4 samples/task at
the card's thinking sampling (temp 1.0, top_p 0.95, top_k 20, min_p 0.05,
xhigh default, 4096 cap), checker-labelled; judge-prompt states through
`/embedding`; heads fitted on train only.

## Before / after: the model checking its own answers

Same model, same 100 first answers (93 correct / 7 wrong), same question
("Is the candidate answer correct?").

| readout | catches its own errors | keeps correct | AUC |
|---|---|---|---|
| **before** (prompted self-check) | **2 / 7** | 91 / 93 | 0.791 |
| after (head, J-Bench fit, thr 0.5) | **6 / 7** | 92 / 93 | 0.966 |
| after (head, in-domain fit, thr 0.5) | **6 / 7** | 93 / 93 | 1.000* |

\* First-sample AUC; the in-domain head is in-sample on the 70 train tasks
(held-out test tasks: 1/2 errors caught, 28/28 kept, per-sample AUC 0.992).
The J-Bench-fit head is out-of-distribution for Bonsai's natural answers (fit
on Strata candidates + perturbations) and still catches 6/7 at one false
reject.  Over all 400 samples: in-domain AUC **0.999** (34/39 errors caught,
361/361 correct kept); J-Bench-fit 0.868 (26/39, 360/361).

Errors are long: first answers that fail average **3,040 tokens** vs 510 for
correct answers (reasoning 9.4k vs 1.2k chars) — Bonsai's failures are the
tasks it over-thinks.

## Before / after: verifying another model's answers (J-Bench test, 116 items)

| readout | AUC | best BA | acc@0.5 | FA / FR @0.5 |
|---|---|---|---|---|
| prompted | 0.952 | 0.960 | 0.810 | 21 / 1 |
| **head** | **0.988** | **0.983** | 0.948 | 6 / 0 |

The prompted readout is unusually strong on this model (0.952 vs Scion's
0.903), and the fitted head still separates above it.

## The selection loop (100 tasks × 4 samples, answer-format head)

| policy | correct | acc | tokens | tok/correct |
|---|---|---|---|---|
| single (B0) | 93/100 | 0.930 | 68,698 | 739 |
| majority of 4 (B1) | 96/100 | 0.960 | 286,545 | 2,985\* |
| best-of-4 by head (T1) | **96/100** | 0.960 | 286,545 | 2,985 |
| **gated head, thr 0.3** | **96/100** | 0.960 | 102,633 | 1,069 (1.12 samples) |
| gated J-Bench head | 96/100 | 0.960 | 78,057 | 813 (1.06 samples) |
| oracle best-of-4 | 96/100 | 0.960 | — | — |

\* Earlier revisions charged the vote only the picked candidate (64,441 →
671 t/c); a vote needs the whole pool.  With honest accounting B1 and T1 cost
the same; the gated arms are the efficiency story.

Full-set frontier: best-of-2/3/4 all 96/100 — the head reaches the oracle on
the first extra sample.  On the frozen 30-task test split the head ranks pools
to 28/30, but single is already 28/30 and the oracle is 28/30 (both test errors
have no correct sample in their pools), so **no selection gain is measurable
on the held-out split**; the +3 (93→96) is on the full bench, whose 70 train
tasks the head was fitted on.  Scion reference (held-out 30): single 21/30 →
best-of-3 25/30 (+4), oracle 27/30, gated 23/30.

Reading: the chooser still closes the available gap, but Bonsai leaves far less
of it — its single-shot answers are at the pool oracle for 28/30 held-out tasks
and 96/100 overall.

## Retry: blind vs hinted (7 wrong-first tasks: 5 train / 2 test)

| retry | any fix | first retry fix |
|---|---|---|
| blind full re-derivation (samples 1–3) | **3 / 7** | 3 / 7 |
| hinted (shown the flagged answer) | 2 / 7 | 2 / 7 |

The anchor effect reproduces on the one task that could show it: `math-0021`
was fixed by all three blind resamples and by neither hinted retry.  The two
held-out errors were not fixable by either arm.  n=7 — directional only, same
verdict as Scion: if you retry, re-derive fully and blindly.

## Attribution audit (in-format neutrality)

Controlled test: every candidate is presented twice with identical text, the
only difference being "Your own answer" vs "Another model's answer".  The
J-Bench-fit head (serving format) applied to the audit wording shows deltas of
**+0.001 on correct** and +0.076 on wrong candidates.  In-format refits over
the 60 held-out candidates (57 correct / 3 wrong): own-only fit +0.062 / +0.115;
**balanced fit (both attributions) +0.005 / +0.012** — neutral, matching the
Scion result (+0.007 / −0.013).  The prompted judge keeps a +0.070 preference
on wrong candidates.  Caveat: wrong-answer deltas rest on 3 held-out negatives
(Bonsai is 93/100 correct); correct-candidate deltas rest on 57.

## Codec vs architecture vs training regime

| body | kind | bits | J-Bench test AUC |
|---|---|---|---|
| Scion-35B-A3B | trained ternary MoE | 2.61 | 0.999 |
| **Ternary-Bonsai-2-27B** | **trained ternary dense** | **2.13** | **0.988** |
| Clef f16 (Qwen3.5-9B) | dense f16 | 16 | 0.952 |
| Clef QAT | post-hoc ternary | 2.1 | 0.712 |
| Clef h128 GPTQ | post-hoc 2-bit | ~2 | 0.648 |

The 2-bit readout failure seen in the Clef lane was neither the bit-width nor
the architecture: a trained-ternary dense body keeps the verdict at 0.988.  The
differentiator is the training regime / codec quality — ternary-aware training
(Bonsai, Scion) vs post-hoc quantisation (Clef QAT/GPTQ), tracking their PPL
ordering and the 4-bit probe literature (0.90–1.00 AUROC).

## Reproduce

1. Verify the artifact: `sha256sum Ternary-Bonsai-2-27B-PQ2_0.gguf` ==
   `3907dc16…62ec1` (7,206,168,928 B).
2. Serve with the fork:
   `llama-server -m Ternary-Bonsai-2-27B-PQ2_0.gguf -ngl 99 -c 8192 \`
   `  --embeddings --pooling last --embd-normalize -1 -fa auto --jinja`.
3. Full session: `run-bonsai.sh` (J-Bench probe → 4 samples/task → hinted →
   embeddings → eval → export → prompted baselines); `run-attribution-bonsai.sh`
   for the controlled attribution audit.
4. Fit/apply: `probe_scion.py --name bonsai27b`, then
   `p = sigmoid(a·(ridge((h−mu)/sd)+b)+c)` from
   `bonsai-2-27b-decision-head.json` (J-Bench threshold 0.58; at 0.5 it catches
   6/7 natural errors with one false reject).  The in-domain head
   (`bonsai-indomain-head.json`) keeps 93/93 at 0.5.

Every number regenerates from the committed JSONs (`probe-bonsai27b.json`,
`bonsai-eval.json`, `bonsai-summary.json`, `readout-bonsai27b-prompted.json`,
`readout-bonsai-own-prompted.json`, `attribution-bonsai-*.json`) and the frozen
bench (`judge-bench.jsonl`, sha256 `3289dcb8…`).

## Caveats

- One benchmark source (cascade-bench-v1) and one task family (candidate
  verification); statements are about this verifier task.
- Bonsai is near-ceiling on this bench: 93/100 first samples correct, 7 natural
  errors — catch/retry rates rest on small n (7 errors; 2 held-out).
- The full-set selection gain is partly in-sample (70 train tasks); the
  held-out split has no recoverable errors to demonstrate selection on.
- The head is prompt-format-locked and body-specific; refit per serving format
  and per quantised body (per Scion / AnyJev).
- A 164 KB head is larger than Scion's 66 KB because the state is 5120-d.

**Follow-up (2026-10-08):** community arms (664 tasks, card sampling), the
veto replication (173, 5 fixes / 0 breaks), and the raw answer-slot margin
readout are in `BONSAI-COMMUNITY-REPORT.md`.
