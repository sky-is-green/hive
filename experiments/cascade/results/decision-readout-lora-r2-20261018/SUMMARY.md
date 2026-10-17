# Adapted-feature refit (post-LoRA) — exploratory

Question: the LoRA adapter's own verdict logits failed on organic errors, but
did it change the **features** in a way a trained head can use? Extract the same
1,315 items (bench 378 + harvest-r2 937) with the epoch-2 adapter, refit the
same heads, compare head-for-head with the frozen r2 refit.

Config: `--adapter adapter-intern-decision-0.8b-lora-r2-epoch2` (the operating-
point-dominant epoch), deployment prompt path unchanged, 24 layers, depths
0.5/0.75/1.0. 1,315 items in **810 s** (0.62 s/item, median 559 ms), MemAvailable
min 17.89 GiB, serve `/health` 200 before and after. Raw A/B readout recorded
alongside the states.

## Frozen vs adapted features

| variant | bench AUC (acc) | organic AUC | organic caught |
|---|---|---|---|
| **frozen r2** | | | |
| raw A/B | 0.771 (74/116) | 0.603 | 0/9 |
| concat | 0.924 (94/116) | 0.688 | 3/9 |
| MLP-concat | 0.991 (111/116) | **0.844** | **5/9** |
| L0.5 | 0.815 (90/116) | 0.827 | **6/9** |
| L0.75 | 0.952 (101/116) | 0.819 | 3/9 |
| L1 | 0.880 (86/116) | 0.770 | 4/9 |
| **adapted** | | | |
| raw A/B | 0.993 (115/116) | 0.673 | 2/9 |
| concat | 0.992 (111/116) | 0.735 | 2/9 |
| MLP-concat | **0.998 (113/116)** | 0.801 | 3/9 |
| L0.5 | 0.992 (99/116) | 0.743 | 2/9 |
| L0.75 | 0.992 (107/116) | 0.776 | 2/9 |
| L1 | 0.996 (109/116) | 0.772 | 2/9 |

**The adapter trades organic for bench.** Every adapted variant improves bench
(raw 0.771 → 0.993, L1 0.880 → 0.996, MLP 0.991 → 0.998) and every one loses
organic (MLP 0.844 → 0.801 and 5/9 → 3/9; L0.5 0.827 → 0.743 and 6/9 → 2/9).
The nine wrongs show the shift: eight of nine are ≥ 0.92 under the adapted
heads; only `humaneval-0001:hs2`, `humaneval-0015:hs1`, `gsm8k-0015:hs2` stay
below 0.5 under the MLP, and the frozen L0.5's extra catches
(`gsm8k-0009:hs2`, `gsm8k-0033:hs1`, `humaneval-0015:hs2`, `math-0020:hs2`) are
all pushed to ~0.8–1.0. This is the same style-prior/shortcut direction seen in
the logits, now visible in the features.

## Cross-space combinations

| rule | caught | corrects escalated |
|---|---|---|
| frozen L0.5 @0.55 (reference) | 7/9 | 10/57 |
| adapted MLP OR frozen L0.5 @0.55 | 7/9 | 13/57 |
| adapted MLP AND frozen L0.5 @0.55 | 3/9 | 0/57 |
| adapted OR frozen MLP @0.5 | 5/9 | 8/57 |
| adapted AND frozen MLP @0.5 | 3/9 | 2/57 |

The adapted head contributes no catch the frozen L0.5 does not already make; it
only adds correct rejections in the union. It is not a useful second opinion.

## Protocol flaw disclosed

The LoRA trainer/evaluator used the literal **yes/no** token ids (9405/2083),
but the deployment interface (`inference.py`) maps the criteria to answer
symbols **A/B** (`ANSWER_SYMBOLS[:2]`) and reads P(B). The adapter therefore
optimized a different token pair than the judge reads: the deployment raw A/B
readout on the adapted model is organic AUC 0.673 (2/9), while the yes/no eval
reported 0.746 (3/9). The pre-registered LoRA FAIL therefore applies to the
yes/no objective, not cleanly to the deployment objective.

That does not rescue the feature result: this refit uses the deployment heads on
adapted states and still loses organic AUC. The data-mix failure mode
(90.5% of organic train answers are correct) is unchanged by the token pair, so
a corrected A/B LoRA is not the recommended next spend.

## Conclusion

- The LoRA adapter is not a gate candidate and not a complementary head. Frozen
  features remain the best organic representation; the same-evidence ceiling
  (7/9 caught, two universal misses) is unchanged.
- The representational line is closed. The cascade next uses the frozen heads as
  tier 1 and adds **different evidence** (task verifier/execution where
  available; a same-model critique/re-solve pass for open-ended tasks) before
  the hosted API.

## Artifacts

- `readout-states-intern-decision-0.8b-lora-r2.npz` (gitignored) — adapted
  states, raw A/B readout, 1,315 items.
- `extract-intern-decision-0.8b-lora-r2.json` — extraction config, adapter path,
  timing, serve health, memory.
- `readout-intern-decision-0.8b-lora-r2.json` — adapted-head fit (all variants,
  by-source bench/organic, thresholds, ECE).
- `extract.log` — ignored.
