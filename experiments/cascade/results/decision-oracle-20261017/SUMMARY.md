# Decision oracle — stage 0 (C2/D2 escalation judges)

Measured on CPU beside the resident 176B serve (port 8081), 2026-10-17.
Frozen J-Bench test split (n=116: 29 correct / 87 incorrect answers,
recorded + perturbed). AUC/accuracy are directly comparable with the
GPU readout runs in `results/jev-readout-20261006/`.

| model | bytes | AUC | acc@0.5 | FA | FR | no-leak thr | median ms | peak RSS |
|---|---|---|---|---|---|---|---|---|
| tiny-jev-1.7b | 3.44 GB | 0.889 | 87/116 | 25 | 4 | - | 479 ms | 3.43 GiB |
| intern-decision-0.8b | 1.71 GB | 0.771 | 74/116 | 35 | 7 | 0.89 (92/116) | 375 ms | 2.24 GiB |
| intern-decision-2b | 4.43 GB | 0.933 | 95/116 | 19 | 2 | 0.89 (98/116) | 736 ms | 4.38 GiB |
| intern-decision-4b | 9.08 GB | 0.979 | 104/116 | 11 | 1 | 0.97 (101/116) | 1939 ms | 8.85 GiB |

## Frontier (role=D2, unit=call)

Non-dominated rows for `D2`:

- **tiny-jev-1.7b** — AUC 0.889, 478.8 ms/verdict, 3.44 GB
- **intern-decision-0.8b** — AUC 0.7711, 375.3 ms/verdict, 1.71 GB
- **intern-decision-2b** — AUC 0.9326, 736.3 ms/verdict, 4.43 GB
- **intern-decision-4b** — AUC 0.9792, 1939.1 ms/verdict, 9.08 GB

## Reading

Operating point first: the D2 verdict feeds the C2 gate at threshold 0.5 (the
harness validator default). False accepts are the escalation danger (a bad
answer passes unverified); false rejects waste an escalation-or-retry. Rates
below are against the test split's 87 wrong / 29 correct items:

- **intern-decision-4b** — AUC 0.979, 104/116 (89.7%), FA 11/87, FR 1/29 —
  strongest by a clear margin, but 1.9 s and 8.9 GB on CPU; resident it leaves
  ~11 GiB of the 21 GiB for the 176B's page cache.
- **intern-decision-2b** — AUC 0.933, 95/116 (81.9%), FA 19/87, FR 2/29 —
  0.74 s, 4.4 GB. The balance point.
- **tiny-jev-1.7b** — AUC 0.889, 87/116 (75.0%), FA 25/87, FR 4/29 — 0.48 s,
  3.4 GB; the harness's production validator already wraps it (threshold 0.5,
  CPU default), so it is the proven service path.
- **intern-decision-0.8b** — AUC 0.771, 74/116 (63.8%), FA 35/87, FR 7/29 —
  weakest; not a verifier, at best a cheap filter.

Against the July GPU runs on the same bench (all split) the ranking is stable:
4B 0.990 GPU / 0.979 CPU > 2B 0.955 / 0.933 > 1.7B 0.871 / 0.889 > 0.8B 0.793
/ 0.771. CPU costs ~7–10× the GPU latency (4B: 184 ms → 1.94 s) and buys what
matters now: no ROCm context, no VRAM, full coexistence with the resident 176B
(MemAvailable min 19.9 GiB across the run).

**Recommended pair (data-backed):** `intern-decision-2b` resident as the D2
verifier (0.74 s, 4.4 GB, 82% operating accuracy); `tiny-jev-1.7b` as the cheap
second rung and the already-wired validator path; the 4B stays the on-demand
"hard verification" option (it loads in seconds from page cache when the 2B
lands in the uncertain band). The 0.8B keeps a router seat only if a C1 eval
says so — its verdict quality does not.

Escalation-budget note: at the 2B operating point roughly every 5th wrong
answer slips through (FA 19/87) and ~1 in 15 correct answers is rejected
(FR 2/29) — each false reject is what an API escalation call would cost. The
no-leak thresholds (2B: 0.89 → 98/116; 4B: 0.97 → 101/116) are the dial to
retune once the API tier's price is known.

## Coexistence with the live 176B serve

- `/health` before: 200, after: 200 (serve stayed up).
- MemAvailable across the run: min 19.94 GiB (see `memwatch.log`); total run 448.6 s.
- All four ran CPU-only (no ROCm context beside the serve). The July
  readout runs scored the intern models on GPU; those numbers remain
  the GPU reference.

## Artifacts

- `judge-<name>.json` — per-item verdicts + metrics (`score_judges.py` schema).
- `<name>.log` — per-model run logs.
- `measurements.json` — CandidateMeasurement rows (also copied to
  `experiments/cascade/decision-oracle-20261017.json` for `/v1/cascade/frontier`).
- `memwatch.log` — host MemAvailable samples.
