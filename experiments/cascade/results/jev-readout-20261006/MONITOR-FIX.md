# Monitor record — sort bug, adapter bug, and the final envelope (2026-10-08)

The trajectory-monitor line needed two corrections before its numbers could be
trusted.  This file is the record of both, the final numbers, and what the
monitor can and cannot do.

## 1. File-sort bug (2026-10-06)

`test-mtp-probe` names dumps `win_%02d_h.bin`; the probes used string `sorted()`
so windows after `win_09` were misaligned with their labels — agreement of the
dump's argmax with the recorded next token was **0.16 with string order vs
0.83 with numeric order**.  Fixed at the source in `monitor_probe.py`,
`monitor_diag.py`, `monitor_token_diag.py`, `hidden_probe.py`; invalid
artifacts preserved as `monitor-*-misaligned.json`.

## 2. Adapter bug (2026-10-08)

The `-corr` GGUFs embed a correction LoRA adapter (240 tensors).  `llama-server`
applies it per batch via `llama_set_adapters_lora`; `test-mtp-probe` and
`test-decision-head` did not, so every Scion dump was produced from the
**uncorrected quant**.  Fixed in both tools (`llama_set_adapters_lora(ctx,
nullptr, 0, nullptr)` after context creation); `test-decision-head` then
reproduced the stored engine scores to Δ≈1e-7.  Qwen and Bonsai carry no
embedded adapter and were unaffected.  Old artifacts preserved as
`*-noadapter.json` / `dump-noadapter/`; the corrected dumps are canonical.

## 3. Final numbers (Scion-35B-A3B correction pool, trace-relative positions)

Correction applied 2026-10-08; Qwen numbers from the original run.

| readout (GRU over 8 snapshots) | noadapter | **corrected** |
|---|---|---|
| read-back start64 / start256 | .803 / .870 | **.832 / .825** |
| context start64 / start128 / start256 / last8 | .772 / .790 / .817 / .949 | **.785 / .792 / .812 / .932** |

Qwen3.8-27B IQ4_XS context: start64 **.866**, start128 .829, start256 .863,
last8 **.994**.  Static single-position probes stay weak (.43–.71) in every
configuration; the trajectory is the signal.

## 4. Early-cancel envelope (`cancel_sim.py`)

Simulated cancel-at-horizon on held-out tasks (thresholds calibrated on the
validation split; low single-digit false-cancel targets):

| | Scion | Qwen |
|---|---|---|
| k=64, conservative | caught 14%, FC 7.5%, **savings 11.8%** | caught 24%, FC 3.3%, **savings 9.4%** |
| k=64, moderate | caught 23%, FC 15%, 19.2% | caught 30%, FC 9.8%, 15.9% |
| k=64, oracle | 33.7% | 27.6% |
| k=256, conservative | caught 19%, FC 5%, 8.1% | caught 27%, FC 16%, 10.2% |

**Verdict: early cancellation is not worth building.**  Even a perfect
predictor saves only 21–34% at these horizons, realistic points save 10–19%
while falsely cancelling 3–15% of correct samples, and validation-calibrated
thresholds systematically under-cancel on test (ranking transfers, thresholds
do not).

## 5. Trajectory selection (`traj_select.py`)

Using the end-of-trace (last8) trajectory score to pick among a task's samples,
held-out correction pool:

| arm | Scion | Qwen |
|---|---|---|
| single | 21/30 | 25/30 |
| majority | 23/30 | 27/30 |
| frozen judge-prompt head | 22/30 | — |
| raw answer-slot margin | **24/30** | — |
| trajectory last8 | 22/30 | **28/30 = oracle** |
| trajectory+raw | 24/30 | — |
| oracle | 27/30 | 28/30 |

The richer channel does not beat the cheapest one where selection matters
(Scion has real headroom: 6 tasks; Qwen's split is nearly saturated: 3).

## 6. Conclusion

The monitor is an **end-of-trace verifier/selector**, not a generation-time
canceller.  Its best use is verification of finished traces (last8 AUC
.93–.99), with early-exit ruled out by the envelope above.  The efficiency
lever remains head-gated sampling, measured separately (community reports).

Artifacts: `monitor-nonlinear.json`, `monitor-probe.json`, `monitor-diag.json`,
`monitor-token-diag.json`, `monitor-context-scion.json` (+ `-noadapter`,
`-misaligned`), `monitor-context-qwen.json`, `cancel-sim-scion.json`
(== `-v2.json`), `cancel-sim-qwen.json`, `traj-select-scion.json`,
`traj-select-qwen.json`.
