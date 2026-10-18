# bonsai-2-27b-transfer

## What was measured

The Bonsai 27B final-token hidden state under the same judge prompt used for
the frozen J-Bench head. The head was **not refit** for the primary numbers;
this is a transfer baseline. The extra extraction contains held-out organic/QA
items plus four train examples needed by the generic probe runner; those four
are only a numerical parity check, not a refit. Evaluation labels are from the
reference readout NPZ and are used for scoring only.

Runtime: pure CPU (HIP/ROCR/CUDA hidden, `--device none --no-op-offload`),
`-t 8 -c 2048 -np 1`, nice 19, taskset 0-7. The 176B serve on 8081 stayed up;
see Stability.

## Head sanity

- Frozen head: `bonsai-2-27b-decision-head`, training n=262
- J-Bench test: expected AUC 0.9881093935790726, observed 0.9881093935790726 (n=116)
- CPU/GPU state parity on 4 re-embedded train rows: mean cosine 0.999957, min 0.999953, max |diff| 0.085247

## Generator context (prior; not this CPU run)

Source: `experiments/cascade/results/jev-readout-20261006/bonsai-eval.json` — prior GPU J-Bench generation eval; not this CPU run.

| arm | correct | n | acc |
|---|---:|---:|---:|
| B0 | 93 | 100 | 0.930 |
| B1 | 96 | 100 | 0.960 |
| T1 | 96 | 100 | 0.960 |

Those J-Bench generation numbers are in-family and from the earlier GPU
run. This run measures judge transfer only; it does not measure Bonsai
generation.

### Combined test (182 items: J-Bench + organic)

| metric | value |
|---|---:|
| n | 182 |
| correct | 86 |
| wrong | 96 |
| auc | 0.9525 |
| ece | 0.1359 |
| caught_at_0.5 | 81 |
| false_rejects_at_0.5 | 0 |
| false_accepts_at_0.5 | 15 |
| best_ba | 0.9479 @ 0.60 |
| at_0.5 decisions correct | 167 / 182 |
| at_0.5 false accepts | 15 |
| at_0.5 false rejects | 0 |
| at_head threshold | 0.58 |
| at_head decisions correct | 171 / 182 |
| at_head false accepts | 11 |
| at_head false rejects | 0 |

### J-Bench test only

| metric | value |
|---|---:|
| n | 116 |
| correct | 29 |
| wrong | 87 |
| auc | 0.9881 |
| ece | 0.1484 |
| caught_at_0.5 | 81 |
| false_rejects_at_0.5 | 0 |
| false_accepts_at_0.5 | 6 |
| best_ba | 0.9828 @ 0.58 |
| at_0.5 decisions correct | 110 / 116 |
| at_0.5 false accepts | 6 |
| at_0.5 false rejects | 0 |
| at_head threshold | 0.58 |
| at_head decisions correct | 113 / 116 |
| at_head false accepts | 3 |
| at_head false rejects | 0 |

### Organic test (57 correct / 9 wrong)

| metric | value |
|---|---:|
| n | 66 |
| correct | 57 |
| wrong | 9 |
| auc | 0.6238 |
| ece | 0.1141 |
| caught_at_0.5 | 0 |
| false_rejects_at_0.5 | 0 |
| false_accepts_at_0.5 | 9 |
| best_ba | 0.6579 @ 0.81 |
| at_0.5 decisions correct | 57 / 66 |
| at_0.5 false accepts | 9 |
| at_0.5 false rejects | 0 |
| at_head threshold | 0.58 |
| at_head decisions correct | 58 / 66 |
| at_head false accepts | 8 |
| at_head false rejects | 0 |

### QA all 129 (reference; includes 38 frozen-head train rows)

| metric | value |
|---|---:|
| n | 129 |
| correct | 44 |
| wrong | 85 |
| auc | 0.9976 |
| ece | 0.0954 |
| caught_at_0.5 | 83 |
| false_rejects_at_0.5 | 0 |
| false_accepts_at_0.5 | 2 |
| best_ba | 0.9882 @ 0.46 |
| at_0.5 decisions correct | 127 / 129 |
| at_0.5 false accepts | 2 |
| at_0.5 false rejects | 0 |
| at_head threshold | 0.58 |
| at_head decisions correct | 127 / 129 |
| at_head false accepts | 2 |
| at_head false rejects | 0 |

### QA organic-only all 129 (reference)

| metric | value |
|---|---:|
| n | 54 |
| correct | 44 |
| wrong | 10 |
| auc | 0.9795 |
| ece | 0.0625 |
| caught_at_0.5 | 8 |
| false_rejects_at_0.5 | 0 |
| false_accepts_at_0.5 | 2 |
| best_ba | 0.9386 @ 0.75 |
| at_0.5 decisions correct | 52 / 54 |
| at_0.5 false accepts | 2 |
| at_0.5 false rejects | 0 |
| at_head threshold | 0.58 |
| at_head decisions correct | 52 / 54 |
| at_head false accepts | 2 |
| at_head false rejects | 0 |

### QA clean 91 (excludes 38 frozen-head train rows)

| metric | value |
|---|---:|
| n | 91 |
| correct | 35 |
| wrong | 56 |
| auc | 0.9954 |
| ece | 0.0975 |
| caught_at_0.5 | 54 |
| false_rejects_at_0.5 | 0 |
| false_accepts_at_0.5 | 2 |
| best_ba | 0.9821 @ 0.46 |
| at_0.5 decisions correct | 89 / 91 |
| at_0.5 false accepts | 2 |
| at_0.5 false rejects | 0 |
| at_head threshold | 0.58 |
| at_head decisions correct | 89 / 91 |
| at_head false accepts | 2 |
| at_head false rejects | 0 |

### QA organic-only clean 91

| metric | value |
|---|---:|
| n | 43 |
| correct | 35 |
| wrong | 8 |
| auc | 0.9679 |
| ece | 0.0734 |
| caught_at_0.5 | 6 |
| false_rejects_at_0.5 | 0 |
| false_accepts_at_0.5 | 2 |
| best_ba | 0.9232 @ 0.75 |
| at_0.5 decisions correct | 41 / 43 |
| at_0.5 false accepts | 2 |
| at_0.5 false rejects | 0 |
| at_head threshold | 0.58 |
| at_head decisions correct | 41 / 43 |
| at_head false accepts | 2 |
| at_head false rejects | 0 |

### QA harvest-only 75 (out-of-family)

| metric | value |
|---|---:|
| n | 75 |
| correct | 31 |
| wrong | 44 |
| auc | 0.9949 |
| ece | 0.0960 |
| caught_at_0.5 | 42 |
| false_rejects_at_0.5 | 0 |
| false_accepts_at_0.5 | 2 |
| best_ba | 0.9773 @ 0.44 |
| at_0.5 decisions correct | 73 / 75 |
| at_0.5 false accepts | 2 |
| at_0.5 false rejects | 0 |
| at_head threshold | 0.58 |
| at_head decisions correct | 73 / 75 |
| at_head false accepts | 2 |
| at_head false rejects | 0 |

### QA organic-only harvest 75

| metric | value |
|---|---:|
| n | 39 |
| correct | 31 |
| wrong | 8 |
| auc | 0.9718 |
| ece | 0.0830 |
| caught_at_0.5 | 6 |
| false_rejects_at_0.5 | 0 |
| false_accepts_at_0.5 | 2 |
| best_ba | 0.9214 @ 0.75 |
| at_0.5 decisions correct | 37 / 39 |
| at_0.5 false accepts | 2 |
| at_0.5 false rejects | 0 |
| at_head threshold | 0.58 |
| at_head decisions correct | 37 / 39 |
| at_head false accepts | 2 |
| at_head false rejects | 0 |

### QA control: J-Bench non-QA refit

This control fits a fresh ridge/Platt head on the
224 J-Bench training items whose bucket is not QA
(all 38 QA training rows removed), then scores the 129 QA items. It is not
the full 1,186-item combined LOFO protocol. It is a weak control: its
all-129 AUC is real but its 0.5 calibration accepts most wrong answers,
and its organic-only AUC is at chance. The frozen head's clean 91 number
above is the stronger QA transfer result.

### QA transfer with J-Bench non-QA train

| metric | value |
|---|---:|
| n | 129 |
| correct | 44 |
| wrong | 85 |
| auc | 0.8909 |
| ece | 0.5348 |
| caught_at_0.5 | 10 |
| false_rejects_at_0.5 | 0 |
| false_accepts_at_0.5 | 75 |
| best_ba | 0.8765 @ 0.97 |
| at_0.5 decisions correct | 54 / 129 |
| at_0.5 false accepts | 75 |
| at_0.5 false rejects | 0 |

### QA control organic-only

| metric | value |
|---|---:|
| n | 54 |
| correct | 44 |
| wrong | 10 |
| auc | 0.4318 |
| ece | 0.1781 |
| caught_at_0.5 | 0 |
| false_rejects_at_0.5 | 0 |
| false_accepts_at_0.5 | 10 |
| best_ba | 0.6273 @ 0.98 |
| at_0.5 decisions correct | 44 / 54 |
| at_0.5 false accepts | 10 |
| at_0.5 false rejects | 0 |

### QA control harvest-only

| metric | value |
|---|---:|
| n | 75 |
| correct | 31 |
| wrong | 44 |
| auc | 0.8666 |
| ece | 0.4737 |
| caught_at_0.5 | 6 |
| false_rejects_at_0.5 | 0 |
| false_accepts_at_0.5 | 38 |
| best_ba | 0.8523 @ 0.97 |
| at_0.5 decisions correct | 37 / 75 |
| at_0.5 false accepts | 38 |
| at_0.5 false rejects | 0 |

## Human-gate workload (simulated)

- Gate candidates: 252 (112 correct / 140 wrong)
- Rejected at p < 0.58: 127 (50.4%)
- Rejected items with eval label correct / wrong: 0 / 127
- Wrong answers missed (p >= threshold): 13 of 140 (recall 90.7%)
- Uncertain band [0.58, 0.73): 3

| threshold | rejected / ask rate | eval-correct rejects | eval-wrong rejects |
|---:|---:|---:|---:|
| 0.50 | 123 (48.8%) | 0 | 123 |
| 0.58 | 127 (50.4%) | 0 | 127 |
| 0.60 | 128 (50.8%) | 0 | 128 |
| 0.65 | 130 (51.6%) | 2 | 128 |
| 0.70 | 130 (51.6%) | 2 | 128 |

| source | candidates | wrong | rejected wrong | missed wrong |
|---|---:|---:|---:|---:|
| bench | 116 | 87 | 84 | 3 |
| harvest | 136 | 53 | 43 | 10 |

| bucket | candidates | wrong | rejected wrong | missed wrong |
|---|---:|---:|---:|---:|
| code | 39 | 21 | 16 | 5 |
| qa | 91 | 56 | 54 | 2 |
| reasoning | 122 | 63 | 57 | 6 |

The gate only asks after a verifier reject; this simulation uses the Bonsai
head's rejection as that trigger. It does not spend API tokens and does not
use labels at runtime. The precision is perfect on rejected items (0 correct
rejected), but recall is weak on organic/free-form wrong answers: the head
accepted most of the organic wrongs. See `human-gate-preview.md` for the
rendered first triggers, `human-gate-candidates.jsonl` for the ranked list,
and `human-gate-missed.jsonl` for the wrong answers the gate would miss.

## Stability

| metric | value |
|---|---:|
| samples | 501 |
| duration | 2506.137 s |
| max load1 | 9.95 |
| min MemAvailable | 13.478 GiB |
| max swap used | 3.801 GiB |
| max partner RSS | 13.576 GiB |
| 176B health failures | 0 |
| min GPU free | card0: 2799 MiB, card1: 1886 MiB |

## Reading this result

- **In-family (J-Bench) and QA-choice verification are strong.** The frozen
  head reproduces its J-Bench test AUC exactly; on the clean 91 QA rows
  (38 training rows excluded) it catches all 48 choice-flip wrongs and
  6 of the 8 organic QA wrongs at p<0.5.
- **Organic/free-form verification is weak.** On the 66 held-out organic
  tasks the head catches only 1 of 9 wrong answers at p<0.58 (0 at p<0.5)
  and rejects no correct answer. That is below the 0.8B organic reference
  (AUC 0.844; 5/9 wrong caught). The combined-test AUC is carried by the
  in-family J-Bench half; do not read it as general organic accuracy.
- **Do not use Bonsai as the sole organic verifier.** Use it for structured
  QA/J-Bench verification, or test it as a generator/repairer (the prior
  in-family generation eval was strong), while the 0.8B verifier and the
  human gate cover organic free-form tasks.
- **Human-gate consequence.** A Bonsai-prefiltered gate would ask on ~50%
  of held-out candidates and every ask would be a wrong answer, but it would
  silently accept most organic wrongs. The gate recall, not just precision,
  has to be tuned with the organic numbers in view.
- The QA control excludes every QA row from training but still uses only
  224 J-Bench non-QA rows; a true
  combined LOFO head would need the 1,186 non-QA harvest embeddings, which
  were not extracted in this run.
- Gold labels are evaluation-only; the runtime gate remains gold-free.
