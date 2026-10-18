# Window A — the generator's own heads on the harvest organic-66 and qa-75

2026-10-18.  One serve-down window (7 min 13 s total) to answer the one free-instrument question
left open by `experiments/cascade/results/fusion-probe-20261018/FUSION-PROBE.md`: what do the
176B's **own** on-disk heads say about its natural errors, measured on the same 66 organic items
and 75 QA items the partner baseline used.

Method: the established port pattern (`run_self_head_window.sh`): stop `scion-flashnext`, start the
`strata-dh` decision-head server on 8897 (engine load 36 s), embed the 140-item harvest transfer
subset with the byte-identical judge prompt (`probe_scion.py`), stop, restart `scion-flashnext`
(healthy in 21 s).  Probe rc=0.  States (gitignored, work drive):

    experiments/cascade/results/scion176b-self-organic-20261018/probe-states-scion176b-transfer.npz

- Embed rate: 374 s / 140 prompts ≈ **2.7 s/prompt**.
- The 4 bench seed rows embedded in both this window and the port's overnight window are
  **bit-identical** (mean and min cosine 1.000000) — the two windows' states are comparable.
- Ignore the `probe-scion176b-transfer.json` ridge it printed (AUC 0.780): it is fit on the 4 seed
  rows only and is not an instrument.  The instruments are the two frozen heads below.

Heads applied (2560-dim Jevons-style readouts, `eval_self_head.py`):

- `176b-own-jbench` — the frozen J-Bench head (`scion/decision-heads/scion-176b-a6b-decision-head.json`,
  bench AUC 0.9941).
- `176b-own-community` — the same body's in-domain natural-error fit
  (`scion/decision-heads-work/overnight/scion176b-community-indomain-head.json`, port per-sample test AUC 0.8516).

## Organic-66 (9 wrong / 57 correct)

| signal | AUC | catch@0.5 | CR@0.5 | catch/CR at caps 0 / 3 / 5 / 9 / 14 |
|---|---|---|---|---|
| **176b-own-community** | **0.803** | 4/9 | 2/57 | 1/0 · 4/1 · 5/4 · 6/8 · 7/12 |
| 176b-own-jbench | 0.671 | 0/9 | 0/57 | 2/0 · 3/2 · 3/2 · 4/6 · 4/6 |
| 0.8B mlp-concat (ref) | 0.844 | 5/9 | 7/57 | 0/0 · 5/1 · 5/1 · 6/9 · 8/13 |
| 0.8B L0.5 (ref) | 0.827 | 6/9 | 9/57 | 0/0 · 4/2 · 4/2 · 7/9 · 7/9 |
| Bonsai frozen (ref) | 0.624 | 0/9 | 0/57 | 2/0 · 3/1 · 3/1 · 4/6 · 4/6 |
| Bonsai community (ref) | 0.630 | 0/9 | 3/57 | 0/0 · 0/0 · 0/0 · 3/8 · 3/8 |

## qa-harvest-75 (44 wrong / 31 correct) and the combined 140

| signal | qa-75 AUC | qa-75 catch@0.5 / CR | window-140 AUC | window-140 catch@0.5 / CR |
|---|---|---|---|---|
| 176b-own-jbench | **0.979** | 42/44 / 1/31 | 0.926 | 45/56 / 1/84 |
| 176b-own-community | 0.504 | 8/44 / 6/31 | 0.676 | 12/56 / 7/84 |

## Fusion, exploratory LOO on the held-out test-182

Same LOO scheme as the previous report (features = logits of each signal, leave-one-out logistic
regression on the 182 test items themselves — exploratory, not an honest fit).  The 176B signal
comes from the port states for bench-116 and this window for organic-66.

| features | test-182 AUC | organic-66 AUC | catch/CR at caps 0 / 3 / 5 / 9 / 14 |
|---|---|---|---|
| self-community | 0.729 | 0.795 | 1/0 · 4/1 · 5/5 · 6/8 · 7/12 |
| self-jbench | 0.959 | 0.645 | 2/0 · 2/0 · 3/4 · 4/6 · 4/6 |
| self-community + L0.5 + MLP + p_raw | 0.975 | **0.858** | 0/0 · 6/3 · 6/3 · 7/6 · 7/6 |
| self-both + L0.5 + MLP + p_raw | 0.984 | 0.856 | 1/0 · 4/2 · 6/4 · 6/4 · 7/13 |
| *previous best*: bonsai-community + L0.5 + MLP + p_raw | 0.978 | 0.858 | 0/0 · 5/2 · 7/5 · 7/5 · 7/5 |

## Reading this result

1. **The generator's own readout is a real organic instrument.**  The in-domain own head lands at
   0.803 on organic-66 — a different league from the two Bonsai cross-family heads (0.624 / 0.630)
   and only ~4 points below the 0.8B MLP (0.844).  It is free at serve time: one engine pass, no
   second body, no extra memory.
2. **The two own heads are complementary, not redundant.**  The J-Bench-frozen head is a QA/structured
   verifier (qa-75 0.979, catching 42/44 at 0.5) and organic-weak (0.671); the community head is the
   mirror image (organic 0.803, qa 0.504).  Both exist on disk already; a gate can route by task form
   and take whichever applies.
3. **In the fusion, the self signal neither helps nor hurts** at this sample size: self + L0.5 + MLP
   + p_raw reproduces the previous best organic number exactly (0.858) and test-182 within 0.003
   (0.975 vs 0.978), *without needing the Bonsai body in the stack*.  With n=9 organic wrongs these
   differences are noise; the finding is that the third body can be dropped, not that the numbers
   moved.
4. The two universal misses (`gsm8k-0033:hs1`, `humaneval-0001:hs1`) remain above 0.87 for every
   signal measured so far; nothing here changes that.

## Caveats

- `176b-own-community` is in-domain by design: its training corpus overlaps some harvest tasks
  (16/39 harvest test+QA tasks appear in its train task split).  Read 0.803 as "in-domain fit",
  not cross-domain evidence.  The Bonsai community head had the identical setup and stayed at
  chance, so the separation is the body, not the training recipe — but a clean cross-task number
  would need the Window B refit (871 train / 66 test harvest split).
- The fusion rows are LOO on the test set itself, exploratory.  The honest fusion v1 (out-of-fold
  features on the r2 train split, single validation on test-182) is the next step; it needs Window B
  to have the 176B signal on the train side.
- qa-75 is a secondary slice (70/75 items sit in the 0.8B heads' training data, as noted in the
  previous report); the own-head qa numbers are unaffected by that particular caveat.

## Files

- `run_self_head_window.sh` — the window (rerunnable; Window B = same script, full harvest bench).
- `eval_self_head.py` — heads on window states, slice metrics, exploratory fusion.
- `self-head-eval.json` — all numbers above, machine-readable.
- `probe-states-scion176b-transfer.npz` (gitignored) — the 140×2560 states.
- `window.log`, `engine.log`, `server.log` (gitignored) — timings and engine record.
