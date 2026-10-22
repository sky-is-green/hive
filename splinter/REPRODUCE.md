# Reproduce

Every headline claim in this repository maps to a command below. The point of
this file is that you do **not** have to trust the author: run the command and
read the exit code.

The claims themselves and their measured verdicts live in
[`SPLINTER-WHITE-PAPER.md`](SPLINTER-WHITE-PAPER.md) §5 (predictions) and §8
(measured-outcome summary). The executable suite lives in the same repository
(the root `tests/`, `experiments/` and `testing/` trees).

## Reproducibility levels

| Level | Meaning |
|---|---|
| **D — deterministic** | No LLM calls, no API keys, no GPU. CPU-only, fixture-driven, seconds to a few minutes. The command's exit code *is* the evidence. |
| **L — live** | Needs an OpenAI-compatible backend (LM Studio / llama.cpp) and, practically, a GPU. Minutes to hours. Re-measures the claim; it does not replay a stored run. |
| **R — recorded** | The number is in the white paper with a run id, but the raw run bundle is not published, so a third party can only re-measure it live, not replay it. Labelled honestly rather than dressed up as reproducible. |

## Setup

The memory system and the suite live in one checkout:

```powershell
git clone https://github.com/sky-is-green/hive.git
cd hive
python -m venv .venv
.\.venv\Scripts\python -m pip install -e splinter
.\.venv\Scripts\python -m pip install -e .
```

Linux/macOS: `python3 -m venv .venv && .venv/bin/python -m pip install -e splinter && .venv/bin/python -m pip install -e .`
from the repository root.

All commands below run **from the repository root** unless noted; the memory
system is resolved from `splinter/` (override with `$SPLINTER_HOME`).

> First run note: the deterministic protocol tests use the real
> `paraphrase-MiniLM-L3-v2` drone (~60 MB). It is fetched once from Hugging
> Face, after which the suite is fully offline. The pure unit suite needs no
> download.

## Offline suite (level D)

```bash
pytest tests/unit tests/integration -q          # full suite, no LLM, no keys, ~30 s
python tests/run_hive_tests.py --group maximum  # same, grouped + timed
python tests/run_hive_tests.py --group speed    # latency / PES thresholds
```

Expected: all collected tests pass. The per-group definitions are in
`tests/run_hive_tests.py`; the current counts are tracked in the
[`hivebench/README.md`](../hivebench/README.md) rather than hard-coded here.

## Prediction map (P1–P11)

`experiments.run_p1_p10` drives the protocol. Without `--live` it runs against a
fake encoder and a mock transport: that exercises the **driver wiring**, not the
measured verdicts.

```bash
python -m experiments.run_p1_p10 --mock          # P1-P10 driver, offline
```

| Claim | Level | Command (from the repository root) | Expected |
|---|---|---|---|
| P1 constant throughput | R/L | `python -m experiments.run_p1_p10 --live --model <model> --base-url http://localhost:1234` | Decode tps flat within ±10% across 500 turns. Recorded: 14.5→15.5 (+6.7%) over 308 turns. |
| P2 retrieval recall | R | `python -m experiments.retrieval_diagnostic <run_dir>` (needs a live run bundle) | Stated-facts recall ≥90%. Recorded: 90.3%. Precision half is **FAIL** (10.7%, encoder ceiling). |
| P3 context sufficiency | **D** | `pytest tests/integration/test_protocol.py::test_p3_long_conversations_close_sufficiency` | PASS — splinter ≥ FIFO on ≥80% of retrievable turns. |
| P4 domain decay separation | **D** | `pytest tests/integration/test_protocol.py::test_p4_horizon_corpus_separates_domains` | PASS — code vs prose m90 gap >0.2. |
| P5 targeted masking | **D** | `python -m experiments.p5_targeted_masking --steps 300 --output models/p5/report.json` | PASS — held-out precision beats random masking; far lower MLM loss. |
| P6 escalation | R | (no passing test — mechanism premise failed) | **FAIL** by design; see white paper §7.6 and Threat 6. |
| P7 auditor–human agreement | R | `python -m experiments.human_label` (GUI + rater) | PASS — 90.25% on 400 valid items. Not scriptable end-to-end. |
| P8 routing accuracy | **D**/R | `pytest tests/unit/test_classifier.py` | Offline classifier behaviour here; the live 100% figure is recorded. |
| P9 duplicate-pair merge | **D** | `pytest tests/integration/test_protocol.py::test_p9_duplicate_pairs_merge` | PASS. |
| P9 densest-beats-recency | **D** | `pytest tests/integration/test_protocol.py::test_p9_densest_beats_recency` | PASS — densest wins on informative turns. |
| P10 drift reset | R | (no passing test) | **FAIL** by design; white paper §7.10. |
| P11 comb resurrection | **D** | `pytest tests/integration/test_protocol.py::test_p11_comb_return_protocol_pass` | PASS — 100% return-turn recall under budget pressure. |

The `tests/integration/test_protocol.py` checks are regression locks: they
assert `status == "PASS"`, so a green run is the reproduction.

## Headline comparison (level L)

The splinter-vs-baseline PES comparison and the paired A/B answer-quality run:

```bash
python -m experiments.run_compare --live --model <model>
python -m experiments.paired_ab --live --model <model> --max-turns 45 \
    --fifo-budget 1500 --checkpoint-every 2 --output runs/paired_ab.json
```

Expected direction: splinter PES well above the FIFO/rolling baselines; paired
A/B at parity overall with splinter-only wins dominating once the FIFO window
starts dropping facts. The specific recorded numbers (PES 80.0 vs 12.2/11.6;
84.1% overall) are from live runs `20260822_211131` and the paired battery —
level **R**, re-measurable but not replayable from a public bundle.

## Not reproducible from the public artifacts

Stated plainly, so nobody mistakes recorded for reproducible:

- The live run bundles behind the §8 table (`20260822_211131`,
  `20260822_live3`, `20260823_014521`) are **not published**. Their numbers are
  **R**.
- The auditor/human-labelling session (P7) is a GUI-assisted process, not a
  one-command run.
- The private coordination log and task ids referenced in some docs are not
  published.

If you re-run a live battery and get a materially different direction, that is
the interesting result — please open an issue with the config and seed.

## Frozen anchors

- Wire contract: [`docs/INTEGRATE.md`](docs/INTEGRATE.md). Route paths
  (`/v1/splinter/*`), the `X-Splinter-Conversation` header, and the MCP tool
  names (`splinter_search`, `splinter_remember`) are the stable surface.
- The sister forensics project pins a byte-level contract with a canonical hash
  (`0d2c008b4aee726351f9b90e44ec003c18b579d8690db24c77a089d9e1fc652b`, checked
  by `python -m bonsai_forensics.spec_hash --check`). This repository does not
  yet expose an equivalent hash; releases should record the `docs/INTEGRATE.md`
  digest once the surface is frozen.
