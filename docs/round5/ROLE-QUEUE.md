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
| `placement-only` | placement | **passed** — 43/46 (0.935) HF and GGUF, gate pass |
| `clerk` | mechanics + RAG (**Hive Reader 1.7B**) | **passed** — mechanics 40/40 HF / 39/40 GGUF, RAG 40/40 both |
| `auditor` | placement + critic (**Hive Auditor 1.7B**) | **queued as a separate adapter** |
| `agency-dispatcher` | agency, dispatcher (**Hive Operator 1.7B**) | queued |
| `merge-incident` | merge/CI coordinator, incident/rollback | queued |
| — | router, judge, face | **not queued** — semantic roles stay Scion/d1 |

**Queue policy change:** one 1.7B LoRA adapter holds **two** constraint roles.
Adding a third role to the same adapter fails the protected-role gate even with
more replay. New roles are trained as **separate adapters** unless a larger base
or an adapter router is used.

**Separate-adapter progress:** placement-only is trained from
`LFM3-1.7B-Base`, 1,200 placement rows, 3 epochs (225 steps). It scores
**43/46 (0.935)** on its held-out suite in both HF and GGUF and passes the
separate-adapter gate. It is running on `:8158` as `placement-1.7b-sft`
(candidate, not wired into a role). Next is critic-only.

## Hive Reader 1.7B (mechanics + RAG)

Mechanics and RAG are combined into one separate adapter because both are
document-in / structured-answer-out and neither needs the Conductor’s scheduling
capacity.

- Held-out suites: 40 mechanics tasks and 40 RAG tasks.
- Corpus: 2,104 rows (1,199 mechanics, 905 RAG; 296 suite-overlap rows dropped).
- Training: from `LFM3-1.7B-Base`, 3 epochs, 396 steps, separate adapter.
- Gate: mechanics and RAG each ≥0.90 HF and GGUF, HF↔GGUF gap ≤0.05.
- **Result: gate PASS.** Mechanics **40/40 HF, 39/40 GGUF** (0.975); RAG
  **40/40 HF and GGUF**. Median 0.08 s mechanics, 0.06 s RAG.
- Candidate on `:8159`, alias `reader-1.7b`; internal run dir
  `clerk-sft-v1`. Release slug `hive-reader-1.7b-v1`.
- Gate report: `work/train-runs/clerk-sft-v1/gate-report.json`.

## Curriculum recovery (queued)

The failed 3-role model (`multirole-cpp-v1`) is the “before”:

| role | before |
|---|---:|
| controller | 54/60 (9/9 original) |
| planner | 68/77, **14/17 original** |
| placement | 45/46 (0.978) |
| placement-plan | 8/12 (0.667) |

The recovery run adds a placement-planner corpus (`placement_plan` plus
placement assignments with a `check` trace) to controller/planner/placement
replay, starting from the failed 3-role model. It is gated by
`experiments/curriculum_gate.py`:

- controller ≥57/60 and original 9/9
- planner original ≥16/17
- placement ≥0.90
- placement-plan ≥0.90 and better than 8/12

If it passes, `run_curriculum_queue.sh` continues to critic-only; if it fails,
the queue stops and Scion is restored. The runner also restores Scion at the
end. Scion baseline for the comparison: controller 50/60, planner 15/17
original.

**Recovery result:** controller **57/60** (up from 54), planner
**69/77 / 15/17 original** (up from 68/14), placement **44/46**, placement-plan
**12/12** (up from 8/12). The gate failed only on planner original: 15/17 is
one short of the 16/17 bar.

A waves/planner anchor then moved controller to **59/60** and kept placement
44/46 and placement-plan 12/12, but planner original fell to **14/17**
(`planner-102` critical path, `planner-106` waves, plus `planner-201` schedule).
The planner does not hold at the two-role 16/17 level when a third role is
merged, even with planner-for-role data. The curriculum is validated as a
**capability-recovery method**, not as a way around the 1.7B’s two-role capacity
limit. The queue stopped; no candidate was promoted; Scion was restored.

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
