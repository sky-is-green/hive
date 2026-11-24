# Experiments

This directory holds the experiment code, inputs, and selected evidence. It is
not a scratch directory.

| Path | Contents |
|---|---|
| `cascade/` | cascade tasks, runners, streams, and result bundles |
| `cascade/results/` | **tracked evidence** for the papers and reports; see `MANIFEST.md` |
| `encoder-suite/` | encoder sweeps, held-out studies, and tuned encoder results |
| `model-matrix/` | model comparison matrices |
| `roles/` | role-specialist studies |
| `moe_ternary/`, `flashnext-perf-2026-10-17/` | model/runtime studies |

Rules:

- Generated checkpoints, captures, prompt dumps, and engine logs are ignored by
  `.gitignore`; keep them out of git.
- A result belongs in `cascade/results/<run>/` only when a paper, release note,
  or `docs/` document cites it. Add or update the `MANIFEST.md` entry when a run
  becomes evidence.
- Scratch runs go under `/tmp` or an external archive, not into the repo.
- Rebuildable model files and engine binaries belong under `HIVE_HOME`, not in
  the checkout.
