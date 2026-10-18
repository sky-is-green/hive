# Fusion v1 — the purpose-trained gate, honestly fit (2026-10-18)

The offline step `FUSION-PROBE.md` pre-registered: fit the fusion on the r2 **train** split with
**out-of-fold** base probabilities, validate once on test-182, report per family, with and without
the 176B's own-head signal.  No serve contact, no extraction, no new model runs — every number is
recomputed from the committed NPZs.

```sh
.venv/bin/python experiments/cascade/fusion_v1.py \
    --outdir experiments/cascade/results/fusion-v1-20261018
```

## Method

- **Out-of-fold base features.**  The deployed 0.8B heads were fit on all 1,133 train items, so
  their train probabilities are perfectly in-sample (AUC **1.000** — see `oof_sanity`) and useless
  as second-stage features.  Base probabilities are refit per fold with `GroupKFold(5)` **by
  `task_id`** using the exact helpers from `decision_readout.py` (ridge+Platt per depth, and the
  64-unit MLP on the concat), so the OOF feature is task-disjoint exactly as the test condition is
  (the r2 test tasks are disjoint from the train tasks — 0 overlap, verified).
- **Test side.**  Deployed heads, unmodified.  They never saw a test task.
- **Fusion.**  One logistic regression on standardized logits, fit on the 1,133 OOF train rows,
  one pass over test-182.  No test-side tuning, no test labels anywhere in the fit.

| base signal | train OOF AUC | train in-sample AUC | test-182 AUC |
| --- | --- | --- | --- |
| 0.8B L0.5 | 0.822 | 1.000 | 0.839 |
| 0.8B MLP-concat | 0.968 | 1.000 | 0.977 |
| 0.8B ridge-concat | 0.895 | 1.000 | 0.907 |

## 1. The honest no-self fusion — the deployable gate

Fit on 1,133 OOF train rows, evaluated once on test-182.  "catch/CR at caps 0/3/5/9/14" is the
max number of the 9 organic wrongs caught with at most k of the 57 organic corrects rejected.

| features | test-182 | bench-116 | organic-66 | catch/CR @ 0 / 3 / 5 / 9 / 14 |
| --- | --- | --- | --- | --- |
| L0.5 | 0.839 | 0.815 | 0.827 | 0/0 · 4/2 · 4/2 · 7/9 · 7/9 |
| MLP | 0.977 | 0.991 | 0.844 | 0/0 · 5/1 · 5/1 · 6/9 · 8/13 |
| **L0.5 + MLP** | **0.978** | 0.991 | **0.852** | 1/0 · 5/1 · 5/1 · 6/9 · 8/12 |
| L0.5 + MLP + p_raw | 0.972 | 0.987 | 0.827 | 2/0 · 5/1 · 5/1 · 5/1 · 6/13 |
| L0.75 + L0.5 + MLP + p_raw | 0.973 | 0.989 | 0.825 | 2/0 · 5/1 · 5/1 · 5/1 · 6/12 |
| all six base features | 0.971 | 0.986 | 0.819 | 2/0 · 5/1 · 5/1 · 5/1 · 7/13 |

Per family (primary `L0.5+MLP`):

| slice | n | wrong | AUC | catch@0.5 | CR@0.5 |
| --- | --- | --- | --- | --- | --- |
| organic-66 | 66 | 9 | 0.852 | 5 | 7 |
| · gsm8k | 32 | 3 | 0.713 | 1 | 2 |
| · humaneval | 16 | 4 | 0.875 | 3 | 4 |
| · math | 13 | 2 | 0.909 | 1 | 1 |
| · truthfulqa | 5 | 0 | n/a | 0 | 0 |

Two results, both negative for the earlier framing:

1. **The previously reported best feature set does not survive honest fitting.**  `L0.5+MLP+p_raw`
   (the row that carried 0.858 organic in `FUSION-PROBE.md` / `SELF-HEAD.md`) drops to **0.827**
   honest, *below* MLP alone.  The raw answer-slot margin carries the third-largest coefficient
   and has organic AUC 0.603 on its own; the LOO table was rewarding it for noise.  More features
   hurt monotonically (all-six: 0.819).
2. **Reweighting the existing heads is not worth much.**  The best honest fusion beats the best
   single 0.8B head by +0.008 organic AUC (0.852 vs 0.844) — one item, inside n=9 noise — and the
   frontier is the same: 6/9 at 9 correct-rejections for both.  The gate's value is not in
   reweighting `L0.5`/`MLP`/`p_raw`; if the purpose-trained fusion is worth anything it has to come
   from a **new** feature.

## 2. The 176B self signal: where it exists, and how much it is worth

The two own heads exist for **all 182 test items** — bench-116 from the port states, organic-66
from Window A — but only for **332 of the 1,133 train items** (bench-262 + the 70 harvest-qa train
items; the organic harvest train rows are missing).  So with-self variants are fit on that subset.

Same-subset ablation — the base features and the fit rows are identical, only the self column is
added, so the delta is the self signal's contribution and nothing else:

| fit rows | features | test-182 | bench-116 | organic-66 | catch/CR @ 0 / 3 / 5 / 9 / 14 |
| --- | --- | --- | --- | --- | --- |
| 1,133 (all train) | L0.5+MLP+p_raw | 0.972 | 0.987 | 0.827 | 2/0 · 5/1 · 5/1 · 5/1 · 6/13 |
| 332 (self-available) | L0.5+MLP+p_raw | 0.973 | 0.989 | 0.825 | 2/0 · 5/1 · 5/1 · 5/1 · 6/12 |
| 332 | **+ self-community** | 0.978 | 0.993 | **0.858** | 0/0 · 6/3 · 6/3 · 6/3 · 7/10 |
| 332 | + self-jbench | 0.983 | 0.998 | 0.834 | 1/0 · 2/2 · 5/5 · 5/5 · 7/14 |
| 262 (bench only) | L0.5+MLP+p_raw | 0.975 | 0.990 | 0.834 | 2/0 · 5/1 · 5/1 · 5/1 · 6/11 |
| 262 | **+ self-community** | 0.978 | 0.992 | **0.867** | 0/0 · 6/3 · 6/3 · 7/7 · 7/7 |
| 262 | + self-jbench | 0.980 | 0.996 | 0.805 | 1/0 · 2/1 · 4/4 · 5/7 · 6/11 |
| 70 (harvest-qa only) | L0.5+MLP+p_raw | 0.969 | 0.985 | 0.809 | 2/0 · 4/1 · 4/1 · 5/6 · 6/14 |
| 70 | **+ self-community** | 0.977 | 0.992 | 0.838 | 2/0 · 5/1 · 5/1 · 6/7 · 6/7 |

- **`self-community` adds ~+0.03 organic AUC in all three fit subsets**, and it is the only feature
  in the whole study that improves the *frontier* rather than just the AUC (6/9 at only 3 correct
  rejections, against 5/9 at 1 for every base-only variant).
- **`self-jbench` is a trap**, and the coefficients say why: it is nearly perfect on the bench rows
  it is fit against (coefficient 3.03 vs 0.83 for community), so the fusion over-trusts it and it
  then fails on organic (0.671 alone).  It buys bench-116 (0.991 → 0.998) and costs organic.

## 3. How much of that is in-domain — the number that decides Window B

`176b-own-community` was fit on the port's community corpus, whose **train** split covers 465
tasks / 1,395 items.  Splitting every test item by whether its task is in that train split:

| slice | n | wrong | note |
| --- | --- | --- | --- |
| organic-66 in-domain | 40 | 7 | tasks seen by the community head |
| **organic-66 clean** | **26** | **2** | both are `humaneval-0001` |
| bench-116 in-domain | 64 | 48 | |
| bench-116 clean | 52 | 39 | |

61% of organic-66 and 7 of its 9 wrongs are in-domain for the self-community head, so **the
honest cross-task organic evidence available offline is two items**.  On that 2-wrong clean slice
the self signal does something nothing else in the stack does:

| item | L0.5 | MLP | self-community |
| --- | --- | --- | --- |
| `humaneval-0001:hs1` | 0.760 | 0.877 | **0.341** |
| `humaneval-0001:hs2` | 0.005 | 0.032 | 0.171 |

`humaneval-0001:hs1` is one of the two "universal misses" that sat above 0.87 for *every* signal
measured in the previous three reports; the generator's own head puts it at 0.341, and the fusion
with self catches 2/2 on the clean slice (AUC 0.979) where the base-only fusion catches 1/2
(0.958).  That is n=2 — suggestive, not conclusive — but it is the only observation in the whole
cascade that points at a *new evidence source* rather than a better weighting of the existing ones.

## 4. Exploratory reference (not honest, for continuity only)

Leave-one-out fusion **on test-182 itself**, the scheme used in the two previous reports:

| features | test-182 | organic-66 | catch/CR @ 0 / 3 / 5 / 9 / 14 |
| --- | --- | --- | --- |
| L0.5+MLP+p_raw | 0.975 | 0.844 | 0/0 · 5/2 · 5/2 · 5/2 · 8/13 |
| + self-community | 0.975 | 0.858 | 0/0 · 6/3 · 6/3 · 7/6 · 7/6 |
| + self-jbench | 0.986 | 0.877 | 1/0 · 5/3 · 6/5 · 6/5 · 7/14 |
| + self-both | 0.984 | 0.856 | 1/0 · 4/2 · 6/4 · 6/4 · 7/13 |

LOO overstates by roughly 0.03–0.04 organic (self-jbench 0.877 LOO vs 0.834 honestly fit) and is
the reason the `p_raw` row looked good.  Treat the two previous fusion tables as upper bounds.

## 5. Where this leaves the gate

- **Deployable now, no window:** `L0.5+MLP` on OOF train — test-182 0.978, bench-116 0.991,
  organic-66 0.852, frontier 6/9 at 9 correct-rejections (16%).  It is honest and it is free, but
  it is not materially better than the single MLP head already deployed, so switching to it is a
  simplification, not a win.
- **The one lever with new evidence:** `self-community` (+0.03 organic across three fit subsets,
  and the only instrument that touches `humaneval-0001:hs1`).  It is free at serve time.  Its
  organic number is 61% in-domain, so the cross-task value of the generator's own head is
  **currently unmeasured, and unmeasurable offline**.
- **That is the Window B case.**  It is the only step that (a) puts the 176B signal on the organic
  train side so the fusion can fit a trustworthy weight instead of inheriting one from bench, and
  (b) produces a harvest-refit head whose test tasks are disjoint from its train tasks — a clean
  cross-task number for the one instrument that showed a catch the 0.8B stack cannot make.  It will
  not lift organic AUC far above 0.85-0.87 on 9 wrongs; its value is confirming or killing the self
  signal, not a number.  **Ask before running it.**
- Pending and untouched by this step: the API provider for the escalation tier, human-gate coverage
  of the router's pre-generation `gate -> API` path, bench ingest builders, 9B Qwen ternary.

## Caveats

- 5-fold OOF means the base features on train come from heads fit on 80% of the data; the test-side
  features come from heads fit on 100%.  If anything the OOF features are the pessimistic version,
  so the fusion weights are conservative.
- The self variants are fit on ≤332 rows that are ~79% bench (262/332) and never contain an organic
  harvest train row, so they inherit the bench-calibrated reading of the self head.  That is the
  exact gap Window B closes.
- `self-community`'s corpus overlap is with **tasks**, not with our labels: its positives/negatives
  are the 176B's own community answers, none of which are harvest organic-66 items.
- n = 9 organic wrongs.  One item is ±0.11 AUC.  The `p_raw` and `all-base` regressions are
  directionally consistent across six feature sets, which is why they are reported as a real effect
  rather than noise; the 0.852 vs 0.844 gap is not.

## Files

- `fusion_v1.py` — this analysis (rerunnable, ~12 s, CPU only).
- `fusion-v1.json` — every number above, machine-readable, including per-slice metrics for each
  fit, per-item probabilities for the 9 organic wrongs, and the OOF sanity table.