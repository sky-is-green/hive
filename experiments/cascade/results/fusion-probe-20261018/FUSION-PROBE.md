# Fusion probe and the generator's own head — cheap avenues before any extraction (2026-10-18)

The Bonsai judge baseline closed at organic-66 AUC 0.6238 with 1/9 wrongs caught, and the open
question was whether a long (7 h CPU) in-domain extraction + refit of the 27B partner could fix
it.  This step explores every signal already on disk first: no new model runs on GPU, no serve
change, no extraction, pure recomputation.

Two rerunnable scripts:

```sh
.venv/bin/python experiments/cascade/fusion_probe.py --outdir experiments/cascade/results/fusion-probe-20261018
.venv/bin/python experiments/cascade/own_head_probe.py --out experiments/cascade/results/fusion-probe-20261018/bench-176b-own-head.json
```

Slices on the r2 authoritative splits/labels: `bench-116` (87 wrongs / 29 corrects), `organic-66`
(the 176B's own harvest answers; 9 wrongs / 57 corrects), `qa-harvest-75` (44 / 31, of which 70/75
are in the 0.8B heads' training data — secondary slice), `test-182` = bench-116 + organic-66.

## 1. A natural-error-trained reader does not transfer across generators

The Bonsai community head was fit in-domain on Bonsai's *own* natural answers (GSM8K/MATH/HumanEval
traces, 1,232 pos / 163 neg; AUC 0.884 on its own held-out tasks).  Its judge prompt is
byte-identical to the harvest extraction, so it applies directly to the 140 harvest states already
on disk.  On organic-66:

| head (Bonsai reading the 176B's answers) | AUC | caught at 0.5 | corrects rejected at 0.5 | max catch at ≤0 corrects rejected |
| --- | --- | --- | --- | --- |
| frozen J-Bench (committed baseline) | 0.624 | 0/9 | 0/57 | 2/9 |
| community in-domain (natural errors) | 0.630 | 0/9 | 3/57 | 0/9 |

Training the reader on natural errors of *its own* answers does not rescue the cross-generator
setting: both heads sit at chance on the 176B's organic errors.  (On bench-116 the community head
is 0.727 vs the frozen head's 0.988 — each head is good only on its own distribution.)  This is the
per-body/per-data rule of the JEV line, and it removes the main premise of the 7 h extraction: the
same reader on the same candidates, fit on Bonsai-reading-176B errors, would still have to find a
discriminative direction that two independent fits say is not there.

## 2. What the existing signals say per slice

| signal | bench-116 AUC | qa-75 AUC | organic-66 AUC | test-182 AUC |
| --- | --- | --- | --- | --- |
| bonsai-frozen | 0.988 | 0.995 | 0.624 | 0.953 |
| bonsai-community | 0.727 | 0.657 | 0.630 | 0.738 |
| 080-L0.5 | 0.815 | 0.968 | 0.827 | 0.839 |
| 080-L0.75 | 0.952 | 1.000 | 0.819 | 0.947 |
| 080-L1 | 0.880 | 0.968 | 0.770 | 0.879 |
| 080-ridge-concat | 0.924 | 0.968 | 0.688 | 0.907 |
| 080-MLP-concat | **0.991** | **1.000** | **0.844** | **0.977** |
| 080-p_raw (answer-slot margin) | 0.771 | 0.585 | 0.603 | 0.798 |

The 0.8B MLP head is the best single signal on both the bench and organic slices; the raw
answer-slot margin, strong on the study's own samples (0.858 Bonsai / 0.880 Scion), does not
detect organic errors here (0.603).  The 27B partner is a bench/QA instrument only.

## 3. Fusion under leave-one-out on the held-out test-182

Logistic regression on logits, standardized per fold, leave-one-out; "catch/CR at caps 0/3/5/9/14"
is the max number of the 9 organic wrongs caught with at most k of the 57 organic corrects rejected.

| features | test-182 AUC | organic-66 AUC | catch/CR at caps 0/3/5/9/14 |
| --- | --- | --- | --- |
| community alone | 0.723 | 0.610 | 0/0 0/0 0/0 2/8 3/10 |
| L0.5 alone | 0.832 | 0.813 | 0/0 4/2 4/2 6/8 7/10 |
| MLP alone | 0.976 | 0.838 | 0/0 5/1 5/1 6/9 8/13 |
| L0.5+MLP | 0.974 | 0.836 | 0/0 5/2 5/2 5/2 8/13 |
| community+L0.5+MLP | 0.977 | 0.846 | 0/0 5/2 5/2 7/8 7/8 |
| community+L0.5+MLP+p_raw | 0.978 | **0.858** | 0/0 5/2 **7/5** 7/5 7/5 |
| all 8 features | 0.985 | 0.850 | 2/0 5/2 5/2 6/7 7/10 |

Reading: the best fusion (0.858 organic AUC, 7/9 caught at 5 corrects rejected) is a tradeoff
against MLP alone (5/9 at 1, 8/9 at 13) and L0.5 alone (6/9 at 9); with 9 wrongs every difference
is within noise, and the Bonsai features do not improve anything (the 8-feature model is *worse*
on organic than the 4-feature one).  The two universal misses (gsm8k-0033:hs1, humaneval-0001:hs1)
stay above 0.87 for every probability signal — a representation limit, not a weighting one.

## 4. The generator's own head, offline on the frozen J-Bench

The 176B's judge states for the 378 frozen J-Bench items were pulled through the Strata engine
during the decision-head port and are on disk (`scion/decision-heads-work/overnight/probe-states-main176b.npz`,
2560-dim).  Applying the 176B's own heads to them, no serve contact (`bench-176b-own-head.json`):

| signal | train-262 AUC | bench-116 AUC | FA/FR @0.5 | FA @stored threshold |
| --- | --- | --- | --- | --- |
| 176b-own-jbench | 1.0000 | **0.9941** | 6/0 | 3 (at 0.60) |
| 176b-own-community | 0.6936 | 0.6984 | 64/2 | — |
| bonsai-frozen-partner | 1.0000 | 0.9881 | 6/0 | 3 (at 0.58) |
| 080-MLP | 1.0000 | 0.9913 | 2/3 | — |
| 080-L0.5 | 1.0000 | 0.8149 | 19/7 | — |
| 080-p_raw | 0.7995 | 0.7711 | 35/7 | — |

The 176B's own head reproduces the port's number (0.9941) and edges the partner (0.9881); the 0.8B
MLP is a close third with a better FA/FR balance.  The bench is saturated — the discriminator is the
*organic* slice, and the 176B's own head is the one instrument not yet measured there.  Its track
record in its own domain (the port's community arms, verified from the raw states: per-sample test
AUC 0.8516, FA 71 / FR 41 / 597 at 0.5, BA 0.768; first-samples 40/65 natural errors caught keeping
87.3% of corrects; within-pool 0.965; veto 14 fixes / 3 breaks over 199 tasks) says it will land in
the same 0.83-0.85 band as the 0.8B, not above it — but it is free at serve time (one engine pass,
no second model) and it is the body that produced the answer.

Caveat: 16 of our 39 harvest test+QA tasks are in the community head's *train* task split, so its
preview on those items is partly in-task; only a harvest refit would be clean, and that needs the
full states.

## 5. Conclusions

- The dense partner as an organic judge is closed.  Two independent Bonsai heads are at chance on
  the 176B's organic errors; the extraction would buy a third reader of the same candidates, not a
  new evidence source.
- The bench is saturated for every bench-fitted head (0.988-0.994); natural-error detection tops out
  at ~0.83-0.85 per-sample across instruments measured so far.  A purpose-trained model is worth
  building as a *fusion gate* over the signals we have (the user's "weigh the probabilities"), not
  as another body reading the same text.
- Fusion v1 (offline): train on the r2 train split (1,133 items) with out-of-fold base
  probabilities, validate once on test-182, report per family.  The deployed heads were fit on that
  split, so out-of-fold features are required for an honest fit.
- The one missing free instrument is the 176B's own head on organic-66.  It needs the decision-head
  server (the port's `strata-dh` build, `/apply-template` + `/embedding`), which means a serve-down
  window — ask first.

## 6. Windows, measured on the same engine

Embedding rate from the port's own log: 1,992 community items (two vectors each) in ~4,237 s ≈
2.8 s per judge prompt.

- **Window A (~10-12 min):** stop `scion-flashnext`, start the dh server (the `dh-overnight`
  systemd-run pattern, auto-restart at the end), embed the 136 unique harvest test/QA judge prompts
  (~6.5 min), restart.  Gives the two 176B heads' verdicts on organic-66 + qa-75.
- **Window B (~50-55 min):** the same for all 937 harvest items → a clean harvest-refit head
  (ridge+Platt on 871 train items, tested on 66) and the 176B signal for every gate candidate, so
  the fusion v1 can use it as a feature.
