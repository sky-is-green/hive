# Scion-35B-A3B + decision head — before / after (2026-10-06)

**Model:** `SkyIsNotGreen/Scion-35B-A3B`
(`Scion-35B-A3B-PQ2_0-corr.gguf`, 11,336,908,288 B, sha256
`545d83a93c6ad147…`) — the released 2.61 bpw ternary MoE.
**Head:** `scion-35b-a3b-decision-head.json` (66 KB) — closed-form ridge +
Platt on the final-layer, last-prompt-token hidden state
(`llama-server --embeddings --pooling last --embd-normalize -1`), fitted on
the J-Bench train split (232 items, bench sha256 `3289dcb8…`).
No fine-tuning, no second model, no second inference pass.

## Before / after: the model checking its own answers

Same model, same 100 generated answers (72 correct / 28 wrong), same question
("Is the candidate answer correct?").  Before = prompted yes/no readout;
after = fitted head.  Threshold 0.5 unless noted.

| readout | catches its own errors | keeps correct answers | AUC |
|---|---|---|---|
| **before** (prompted) | **7 / 28** | 72 / 72 | 0.559 |
| **after** (head, thr 0.5) | **23 / 28** | 72 / 72 | 0.971 |
| after (head, thr 0.87, calibrated) | **26 / 28** | 67 / 72 | 0.971 |

3.3x more of its own errors caught at threshold 0.5 with **zero** false
rejects; 26/28 at the calibrated point for five rejected-correct answers.

## Before / after: verifying another model's answers (J-Bench test, 116 items)

| readout | AUC | best BA | acc@0.5 | FA / FR @0.5 |
|---|---|---|---|---|
| prompted | 0.903 | 0.925 | 0.862 | 15 / 1 |
| **head** | **0.9988** | **0.994** | 0.931 | 8 / 0 |

## Reproduce

1. Build the frozen bench: `python experiments/cascade/judge_bench.py`
   (verify `sha256` in `judge-bench.manifest.json`), or use the committed
   `judge-bench.jsonl`.
2. Serve the released model with the fork runtime:
   `llama-server -m Scion-35B-A3B-PQ2_0-corr.gguf -ngl 99 -c 8192 \`
   `  --embeddings --pooling last --embd-normalize -1 --port 8897`
3. Generate its own answers (checker-labelled):
   `python experiments/cascade/gen_answers.py --server http://127.0.0.1:8897`
   (reproduces 72/100 on a greedy decode).
4. Fit and export the head:
   `python experiments/cascade/probe_scion.py --server ... --name scion35b`
   then `python experiments/cascade/export_head.py`.
5. Apply: embed the judge prompt, then
   `p = sigmoid(a * (ridge((h - mu)/sd) + b) + c)`, threshold at 0.5 (or the
   exported `test.threshold`).

Every number regenerates from the committed JSONs
(`readout-scion35b-prompted.json`, `readout-scion-own-prompted.json`,
`source-audit.json`, `probe-scion35b.json`, `scion-35b-a3b-decision-head.json`)
and the frozen bench.

## Caveats

- J-Bench negatives are checker-verified perturbations plus the natural errors
  from the source audit; the frozen test split is 116 items.
- The head is prompt-format-locked and body-specific (see `PLAN.md`): fit it on
  the serving prompt, refit per quantized body.
- The controlled attribution audit shows the in-format head is
  attribution-neutral (own-minus-other delta ~0); the prompted readout shows
  +0.099 self-preference on its own wrong answers.
- One task family (candidate-answer verification) and one benchmark source
  (cascade-bench-v1); results are statements about this task, not a general
  law.
