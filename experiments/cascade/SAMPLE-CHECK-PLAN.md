# Sample-and-check plan (post-M4)

## Goal

Produce labels on the *silent* distribution — the drafts the deterministic
detectors accept but are wrong, plus the fragile corrects — so the learned
gate can be re-audited and (if it separates) retrained or restarted (M5)
under a pre-registered rule, the same discipline that made M4 decidable.

## Transform

Escalation gold seeds the loop.  For every ladder-resolved item harvested by
the escalation lane (`escalation_lane.py`), sample `k` fresh local drafts at
T=0.7, checker-label each (labels are eval-only; the runtime stays
gold-free), extract the 0.8B readout states, and append to a labeled corpus.

- Script: `experiments/cascade/sample_check.py`
- Inputs: run dirs with `run/report.json` + `tasks-organic66.json`.
- Hard tasks: rows where `escalated or flagged` (and `scion_ok is not None`,
  i.e. a local attempt happened) — the mined hard tasks.
- Per task: `k` drafts (default 4) at temperature 0.7, max_tokens 1024,
  reasoning_effort none.
- Per draft: `check(task, draft)` label, then `LearnedGate.states()` extraction
  (L0.5/L0.75/L1).
- Output: `<out>/corpus.jsonl` + `<out>/corpus_states.npz` + per-run
  `<out>/<run>/sample_check.json`.

## Pre-registered acceptance (frozen before any run)

A head (path B or path A) may be promoted from opt-in to default only if, on
the pooled labeled corpus:

1. **≥ 1 fresh silent-wrong catch** at the frozen 0.5 cut, **and**
2. **≤ 2 false flags** on silent corrects, **and**
3. **no false-flag damage** (every false flag's draft was actually correct —
   by construction, since labels come from the checker), **and**
4. **a genuine separation** — the wrongs' p max must sit below the corrects'
   10th percentile for at least one head.  A better threshold on the same
   features does not count (M4 showed that ceiling is 3 catches / 0 flags for
   L0.75 and 4 / 1 for mlp-concat, but only per-run, not pooled).

No per-run or per-head threshold tuning.  The frozen 0.5 cut and the frozen
head choice (L0.75) are the baseline; a promoted head must beat them under
the rule above on the *pooled* corpus, not on a single run.

## Exit criteria for "labels are enough"

- Corpus ≥ 200 silent drafts (≈ 40 wrongs) across ≥ 3 runs.
- At least one head shows a gap in the pooled p ranges.
- Only then: decide M5 (serve restart for path A) or a path-B retrain.

## Audit

`experiments/cascade/audit_labeled.py` reports per-head separation, cap-2
ceiling, and the verdict (separates / overlaps) on the pooled corpus.

## Consumers (downstream of this corpus)

1. Gate re-audit / retrain (this plan).
2. Repair SFT on `(prompt, draft, finding, fixed)` triples from the
   escalation lane.
3. Local-tier (176B) distillation of API/human answers.

## Status

Built: `sample_check.py`, `audit_labeled.py`.  First corpus run in progress.
M5 still held.