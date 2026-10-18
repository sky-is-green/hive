# Different questions to the same 0.8B — model-only second opinion

The cascade idea's cheapest second pass: keep the model, the state, and the
deployment noul interface; change only the question. Five probes on the frozen
organic-66, scored as P(question true) (inverted for the error question), unions
against the frozen L0.5 @0.55 gate.

Config: 66 items × 5 questions = 330 forwards, **180 s** (median 547 ms per
question), serve `/health` 200 before and after, CPU bf16.

## Per-question result

| question | AUC | caught @0.5 | corrects rejected | wrongs' score range |
|---|---|---|---|---|
| holistic | 0.613 | 1/9 | 10/57 | 0.477–0.811 |
| arithmetic | 0.607 | 1/9 | 10/57 | 0.443–0.797 |
| edge cases | 0.350 | 5/9 | 42/57 | 0.421–0.579 |
| error (inverted) | 0.566 | **8/9** | 39/57 | 0.443–0.500 |
| requirements | 0.596 | 1/9 | 8/57 | 0.477–0.684 |

Union with L0.5 @0.55 (reject if either rejects):

| union | caught | corrects rejected |
|---|---|---|
| L0.5 alone (reference) | 7/9 | 10/57 |
| + holistic | 7/9 | 19/57 |
| + arithmetic | 7/9 | 18/57 |
| + edge cases | 7/9 | 43/57 |
| + error | **8/9** | **43/57** |
| + requirements | 7/9 | 17/57 |

## Reading

- **The model's own readout is the bottleneck, not the question wording.** Every
  probe lands at AUC 0.35–0.61; the trained L0.5 head on the same forward pass is
  0.827. Changing the question does not recover what the head gets from the
  hidden state.
- The **error** question catches 8/9 (including `humaneval-0001:hs1`, one of the
  two universal head misses) but only by calling almost everything erroneous:
  39/57 corrects rejected, and every wrong's score sits in 0.443–0.500. It is a
  trigger-happy detector, not a usable second opinion. It misses `math-0013:hs1`.
- The **edge-case** question is anti-correlated as posed (wrongs score higher
  than corrects): the model does not evaluate that dimension in the direction
  the wording asks.
- No union beats L0.5 @0.55 at a comparable cost; the extra catch costs 33
  correct escalations.

## Caveat

n = 9 organic wrongs; one prompt per question; no per-question calibration or
temperature tuning. A calibrated threshold could move a point along the trade-
off, but the AUCs are too low for calibration to close the gap to the trained
head.

## Conclusion

The "recompute again, trying a different way" tier cannot be another question to
the same 0.8B. For a genuinely different second opinion, change the **evidence
source** — task verifier/execution where available, or the hosted API — not the
prompt wording.

## Artifacts

- `alt-questions.json` — per-item scores for all five questions, metrics, unions,
  timings, serve health.
- `experiments/cascade/alt_questions.py` — the probe.
- `alt.log` — ignored.
