# Cascade run artifacts

Inventory generated 2026-11-12.

Referenced runs stay in the tree. Unreferenced runs were moved to
`/run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work/hive-artifacts-archive-2026-11-12/results` so they can be restored without a git revert.

| run | size MB | referenced by | state |
|---|---:|---|---|
| `a2-oracle-stage0-20261017` | 0.18 | `experiments/cascade/vision_oracle.py`, `harness/cascade/registry.py` | kept |
| `api-frontier-20261022` | 0.02 | `docs/AB-RESULTS.md`, `experiments/cascade/api_frontier_arm.py` | kept |
| `bonsai-escalation-20261021` | 0.04 | `docs/AB-RESULTS.md`, `experiments/cascade/bonsai_arm.py` | kept |
| `cascade-arm-escalation-10-20261020` | 0.01 | — | archived |
| `cascade-arm-escalation-20261020` | 0.04 | `experiments/cascade/bonsai_arm.py`, `experiments/cascade/controller_loop.py`, `experiments/cascade/measure_checker.py` | kept |
| `cascade-arm-escalation-norepair-20261020` | 0.04 | — | archived |
| `cascade-arm-gate-intern-20261020` | 0.05 | — | archived |
| `cascade-arm-gate-intern-s1-20261020` | 0.04 | — | archived |
| `cascade-arm-gate-intern-s2-20261020` | 0.05 | — | archived |
| `cascade-arm-judge-none-20261020` | 0.04 | — | archived |
| `cascade-arm-judge-none-s1-20261020` | 0.04 | — | archived |
| `cascade-arm-judge-none-s2-20261020` | 0.04 | — | archived |
| `cascade-arm-repair-auto-20261020` | 0.05 | — | archived |
| `cascade-arm-repair-auto-s1-20261020` | 0.04 | — | archived |
| `cascade-arm-repair-auto-s2-20261020` | 0.05 | — | archived |
| `cascade-arm-repair-off-20261020` | 0.04 | — | archived |
| `cascade-arm-repair-off-s1-20261020` | 0.04 | — | archived |
| `cascade-arm-repair-off-s2-20261020` | 0.04 | — | archived |
| `cascade-arm-router-jev-10-20261020` | 0.02 | — | archived |
| `cascade-e2e-20261018` | 0.01 | `experiments/cascade/cascade.py` | kept |
| `cascade-e2e-budget640-20261018` | 0.01 | — | archived |
| `cascade-e2e-sampled-20261018` | 0.01 | `cascade/CASCADE-PAPER.md`, `cascade/CASCADE-PILOT.md` | kept |
| `cascade-organic-176b-10-20261020` | 0.01 | — | archived |
| `cascade-organic-35b-10-20261020` | 0.01 | — | archived |
| `cascade-organic-35b-20261020` | 0.04 | — | archived |
| `checker-v0-20261022` | 0.02 | `docs/AB-RESULTS.md`, `experiments/cascade/measure_checker.py` | kept |
| `clef-flash-validator-20261003` | 0.24 | `cascade/CASCADE-PILOT.md` | kept |
| `consistency-v2-20261018` | 0.00 | `experiments/cascade/consistency_check.py` | kept |
| `controller-20261022` | 0.42 | `docs/AB-RESULTS.md`, `experiments/cascade/api_replaces_27b.py`, `experiments/cascade/controller_loop.py` | kept |
| `decision-alt-questions-20261018` | 0.01 | `experiments/cascade/alt_questions.py` | kept |
| `decision-cascade-accounting-20261018` | 0.00 | — | archived |
| `decision-harvest-20261017` | 0.85 | `experiments/cascade/decision_readout.py`, `experiments/cascade/harvest.py`, `experiments/cascade/harvest_merge.py` | kept |
| `decision-harvest-r2-20261017` | 1.31 | `experiments/cascade/alt_questions.py`, `experiments/cascade/blindspot_marginal.py`, `experiments/cascade/consistency_check.py` | kept |
| `decision-lofo-truthfulqa-20261018` | 0.03 | — | archived |
| `decision-oracle-20261017` | 0.44 | `experiments/cascade/decision_oracle.py`, `experiments/cascade/decision_readout.py`, `harness/cascade/registry.py` | kept |
| `decision-oracle-harvest-20261017` | 1.30 | — | archived |
| `decision-readout-20261017` | 0.01 | `experiments/cascade/decision_readout.py`, `harness/cascade/registry.py` | kept |
| `decision-readout-harvest-20261017` | 0.11 | `harness/cascade/registry.py` | kept |
| `decision-readout-lora-r2-20261018` | 0.03 | `harness/cascade/registry.py` | kept |
| `decision-readout-r2-20261017` | 0.04 | `experiments/cascade/LEARNED-GATE-PLAN.md`, `experiments/cascade/alt_questions.py`, `experiments/cascade/eval_partner_transfer.py` | kept |
| `dual-generation-20261018` | 0.02 | `experiments/cascade/analyze_dual_cascade.py`, `experiments/cascade/analyze_repair.py`, `experiments/cascade/cascade.py` | kept |
| `dual-scion-20261020` | 0.00 | `cascade/CASCADE-PAPER.md`, `docs/AB-RESULTS.md` | kept |
| `e4-baseline-20261002` | 0.32 | `docs/ENGINE.md` | kept |
| `fresh-silent-audit-20261018` | 0.08 | `experiments/cascade/LEARNED-GATE-PLAN.md`, `experiments/cascade/fresh_silent_audit.py` | kept |
| `fusion-probe-20261018` | 0.03 | `cascade/BONSAI-RUNTIME.md`, `experiments/cascade/eval_self_head.py` | kept |
| `fusion-v1-20261018` | 0.08 | `experiments/cascade/fusion_v1.py` | kept |
| `gate-replay-20261018` | 0.02 | `cascade/CASCADE-PILOT.md`, `experiments/cascade/gate_replay.py` | kept |
| `h-threshold-ab-20261020` | 6.20 | — | archived |
| `jev-readout-20261006` | 4.80 | `experiments/cascade/analyze_logit_community.py`, `experiments/cascade/answer_retry.py`, `experiments/cascade/attribution_audit.py` | kept |
| `judge-gate-20261021` | 0.02 | `docs/AB-RESULTS.md`, `experiments/cascade/bonsai_arm.py`, `experiments/cascade/controller_loop.py` | kept |
| `lora-verdict-20261017` | 0.05 | `experiments/cascade/lora_report.py`, `experiments/cascade/lora_verdict.py`, `harness/cascade/registry.py` | kept |
| `p1p10-35b-20261020` | 0.12 | — | archived |
| `paired-ab-176b-stock-15x16-20261020` | 0.06 | — | archived |
| `paired-ab-35b-20261020` | 0.28 | — | archived |
| `paired-ab-35b-large-20261020` | 1.67 | — | archived |
| `paired-ab-qwen-15x16-20261020` | 0.61 | — | archived |
| `paired-ab-qwen-emb-20261020` | 0.03 | — | archived |
| `paired-ab-stock-15x16-20261020` | 0.59 | — | archived |
| `partner-bonsai-organic-20261018` | 0.62 | `cascade/BONSAI-RUNTIME.md`, `cascade/HUMAN-GATE.md`, `experiments/cascade/eval_partner_transfer.py` | kept |
| `repair-pass-20261018` | 0.01 | `cascade/CASCADE-PILOT.md`, `experiments/cascade/analyze_repair.py`, `experiments/cascade/repair_pass.py` | kept |
| `runner-cascade-gate-nothink-20261018` | 0.18 | `cascade/CASCADE-PILOT.md`, `experiments/cascade/LEARNED-GATE-PLAN.md`, `experiments/cascade/fresh_silent_audit.py` | kept |
| `runner-cascade-sampled-20261018` | 0.13 | `cascade/CASCADE-PAPER.md`, `cascade/CASCADE-PILOT.md`, `experiments/cascade/LEARNED-GATE-PLAN.md` | kept |
| `runner-cascade-sampled-nothink-20261018` | 0.16 | `cascade/CASCADE-PAPER.md`, `cascade/CASCADE-PILOT.md`, `experiments/cascade/fresh_silent_audit.py` | kept |
| `scion-flashnext-community-20261009` | 0.00 | — | archived |
| `scion-strata-20261009` | 0.41 | — | archived |
| `scion176b-harvest-20261018` | 0.24 | `experiments/cascade/consistency_check.py`, `experiments/cascade/eval_harvest_head.py`, `experiments/cascade/fusion_v1.py` | kept |
| `scion176b-self-organic-20261018` | 0.04 | `cascade/BONSAI-RUNTIME.md`, `cascade/CASCADE-PILOT.md`, `experiments/cascade/LEARNED-GATE-PLAN.md` | kept |
| `silent-labeled-corpus` | 0.04 | `experiments/cascade/sample_check.py` | kept |
| `silent-labeled-corpus-broad` | 0.08 | — | archived |
| `silent-labeled-corpus-merged` | 0.08 | — | archived |
| `stability-500-20261020` | 0.00 | — | archived |
| `strata-bench-test-20261003` | 0.17 | `cascade/CASCADE-PILOT.md`, `docs/ENGINE.md`, `tests/unit/test_bench_report.py` | kept |
| `strata-bench-test-capped-20261003` | 0.16 | `cascade/CASCADE-PILOT.md` | kept |
| `strata-bench-test-labels-20261003` | 0.05 | `experiments/cascade/attribution_audit.py`, `experiments/cascade/judge_bench.py`, `experiments/cascade/source_audit.py` | kept |
| `strata-bench-train-20261003` | 0.11 | `experiments/cascade/attribution_audit.py`, `experiments/cascade/judge_bench.py`, `experiments/cascade/source_audit.py` | kept |
| `strata-cascade-20261003-easy` | 0.01 | — | archived |
| `strata-cascade-20261003-hard` | 0.02 | — | archived |
| `strata-feasibility-20261002` | 0.04 | `cascade/CASCADE-PILOT.md`, `docs/ENGINE.md`, `experiments/cascade/bench_strata.py` | kept |
| `strata-launcher-smoke-20261003` | 0.00 | — | archived |
| `strata-local-smoke-20261003` | 0.00 | `cascade/CASCADE-PILOT.md` | kept |

38 unreferenced run director(ies) archived; 42 kept.
