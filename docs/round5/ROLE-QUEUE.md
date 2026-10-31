# Role queue and gate

**Date:** 2026-10-31
**Baseline:** `multirole-cp-v1` (controller + planner), live on `:8156`
**Queue:** `experiments/role_queue.json`
**Gate:** `experiments/role_gate.py`

The queue adds roles in batches. Each batch is trained from the current best
checkpoint, evaluated on **every protected role** plus the new roles, and passes
only if the gate holds. A failed batch does not advance; the previous checkpoint
stays live.

## Batches

| batch | roles | status |
|---|---|---|
| `placement-critic` | placement, critic | **failed gate** — 3- and 4-role tests degraded controller/planner; see `MULTIROLE-LIMIT.md` |
| `placement-only` | placement | **queued as a separate adapter** |
| `critic-only` | critic | **queued as a separate adapter** |
| `agency-dispatcher` | agency, dispatcher | queued |
| `merge-incident` | merge/CI coordinator, incident/rollback | queued |
| — | router, judge, face | **not queued** — semantic roles stay Scion/d1 |

**Queue policy change:** one 1.7B LoRA adapter holds **two** constraint roles.
Adding a third role to the same adapter fails the protected-role gate even with
more replay. New roles are trained as **separate adapters** unless a larger base
or an adapter router is used.

## Gate rules

Protected roles are always re-evaluated, even when they are not in the batch.

| check | threshold |
|---|---|
| controller original 9 | 9/9 |
| controller wider 60 | no more than 2 tasks lost from baseline (≥57/60) |
| planner original 17 HF | ≥16/17 |
| planner original 17 GGUF | ≥15/17 |
| planner wider 77 | no more than 2 tasks lost from baseline (≥67/77 HF, ≥64/77 GGUF) |
| new role HF | ≥0.90 |
| new role GGUF | ≥0.90 |
| HF↔GGUF gap | ≤0.05 |
| missing eval file | fail |

The baseline is `../train-runs/multirole-cp-v1`. The gate compares candidate
evals against it and writes `gate-report.json`; exit code 0 means the batch may
advance to a **canary**, not to the live endpoint.

Batches with `"mode": "separate-adapter"` skip the protected-role checks: that
adapter contains only its own role, so controller/planner regression is not
applicable. Those batches only need the new role(s) to clear 0.90 with a small
HF↔GGUF gap.

## Workflow per batch

1. Build the held-out suite and checker-validated corpus for the new roles.
2. Mix the new rows with replay from all earlier roles.
3. Train from the current best checkpoint (low LR continuation).
4. Evaluate the new roles **and** controller/planner, HF and GGUF.
5. Run `role_gate.py`.
6. If pass: candidate becomes the new baseline, then a live canary with Scion
   fallback. If fail: stop the queue and investigate the lost role; do not
   overwrite the live endpoint.
7. Package to a new HF repo only with explicit name/visibility approval.

## Commands

```sh
python experiments/role_gate.py \
  --batch placement-critic \
  --candidate-dir ../train-runs/multirole-pc-v1 \
  --out ../train-runs/multirole-pc-v1/gate-report.json
```

Validate the gate itself against the baseline:

```sh
python experiments/role_gate.py --roles controller,planner \
  --candidate-dir ../train-runs/multirole-cp-v1
```

## Current baseline numbers

| role | HF | GGUF | original suite |
|---|---:|---:|---:|
| controller | 59/60 (0.983) | 59/60 (0.983) | 9/9 |
| planner | 69/77 (0.896) | 66/77 (0.857) | 16/17 HF, 15/17 GGUF |

## Degradation policy

A lost original task is a hard fail for controller (9/9) and planner (16/17 HF
or 15/17 GGUF). A wider-suite loss of up to 2 tasks is tolerated as decoding
noise. If a role loses more, first increase its replay share; if it still
fails, split the role into its own adapter rather than forcing one model.

The queue stops at the first failed gate. This is deliberate: a batch that
breaks an earlier role is not a candidate, even if the new role scores well.
