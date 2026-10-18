# Continuation — after Window B (2026-10-18)

Fresh-session handoff.  Paste the "Start here" section (or this whole file) as the first
message of the next session.

---

## Start here

Repo `hivebench`, branch `hive_cascade` (one step per commit; **never** `git push`).
The 176B serve is up on 8081 (user unit `scion-flashnext`) — keep it up; any serve-down
window needs the user's explicit go-ahead first, and the window script has a trap that
always restores the service.

**Window B ran and the self signal is real.**  All 937 harvest judge states are on disk
(gitignored); the 176B's own head refit on 871 harvest train items and tested on the 66 organic
items — whose tasks share **0** tasks with its train items — lands at **organic-66 AUC 0.832**
cross-task, *above* the in-domain community head's 0.803.  In the honest out-of-fold fusion
(the 176B signal is now on all 1,133 train rows) the gate reaches **organic 0.897 / test-182
0.988**, against 0.852 for the best no-self fusion and 0.844 for the single best 0.8B head.
Report: `experiments/cascade/results/scion176b-harvest-20261018/WINDOW-B.md`.

Three things to carry:

1. **The generator's own readout is the strongest organic signal we have, and it is free** —
   one engine pass, no second body, no extra VRAM.  The deployed gate is now
   `L0.5+MLP+p_raw+self-harvest`, frontier **7/9 organic wrongs caught at 5 of 57 correct
   rejections** (was 5/9 at 1).
2. **`self-jbench` stays out of the gate.**  It is a QA/structured verifier (qa-75 0.979,
   organic 0.671); every fusion containing it loses organic AUC (0.897 → 0.809).  Keep it as a
   task-form router input, not a gate feature.
3. **Reweighting the existing 0.8B heads is not a lever** (fusion v1: +0.008 over the best single
   head, inside noise), and **leave-one-out on the test split inflates by ~0.03-0.04 organic** —
   the `p_raw` row that read 0.858 honestly reads 0.827.  Both lines are closed.

## A leak worth remembering

The Window B refit head is fit on the 871 harvest train rows, so wiring the exported head
straight into the fusion as a feature reads **organic 0.924** — in-sample on the train side, the
best-looking number in the project, and wrong.  `fusion_v1.py` refits it out-of-fold on the train
side (`GroupKFold(5)` by `task_id`, same lambda) and uses the full-fit head everywhere else, which
is honest because the organic-66 tasks are disjoint from its train tasks and the bench items were
never in its training corpus.  Honest: **0.897**.  Any future head added to the fusion needs the
same treatment.

## Which lines are closed

- Dense partner (Bonsai-27B) as an organic judge: **closed** (0.624 / 0.630, chance).
- Bench is saturated for every bench-fitted head (0.988-0.999).
- Natural-error detection for the 0.8B heads: **closed** at ~0.84-0.85 per sample.
- Fusion by reweighting the existing signals: **closed**.
- LOO-on-the-test fusion as an evaluation scheme: **closed**.
- Two organic items are missed by *every* instrument measured and are the residual:
  `gsm8k-0033:hs1`, `math-0013:hs1`.  `humaneval-0001:hs1` is no longer one of them — both self
  heads reject it (0.341, 0.185).

## What remains

- **API provider** for the escalation tier: the user deferred it ("not now" as of Window B).  It
  blocks the production API tier; nothing else is blocked by it.
- Human-gate coverage of the router's pre-generation `gate -> API` path (the post-verifier path
  is built and tested).
- Bench ingest builders; 9B Qwen ternary (deferred).
- Optional, if the gate is to move further: the next clean instrument is a *different* evidence
  source, not another reader of the same candidate text.

## Next moves (in order)

1. Wire `self-harvest` into the serving gate and measure the escalate-vs-accept decision on the
   frozen organic-66 at a fixed budget (7/9 at 5 CR) — this is the first deployment-shaped use of
   the fusion.
2. Human-gate coverage of the router's pre-generation `gate -> API` path.
3. Keep pending: API provider (user deferred), bench ingest builders, 9B Qwen ternary.

## Hard constraints

- Never `git push`; one step per commit.
- Do not touch `scion/docs/HARVEST-NEMOTRON-FLASHNEXT.md` or `scion/docs/DRAFT-nemotron-field-notes.md`.
- Serve windows: ask first; the script's EXIT trap always restarts `scion-flashnext` and waits for
  health.  Budget **~95 min cold / ~50 min warm** for a 937-prompt window (3.0 s/prompt warm,
  5-7 s/prompt cold); the probe now checkpoints every 25 items and rides out engine restarts.
- No large artifacts in `/tmp`; durable outputs go to the work-drive lane dirs.
- Partner stays CPU-preferred (no persistent dGPU use); no torch-ROCm context beside the serve.
- Runtime corrections must be gold-free (`code_exec`, `constraint_check`, `schema_ast`, `consistency`);
  `numeric_gold` / `choice_made_gold` are eval-only.
- Before any API escalation, ask the user; two top-level choices only: solve locally or escalate.
- Human gate: first trigger verbose with full evidence, later triggers compact; outcomes exactly
  `solve locally` or `escalate to API`; `keep` only as a solve sub-option when the checker already passes.

## Key files

- `experiments/cascade/results/scion176b-harvest-20261018/WINDOW-B.md` — **Window B report**.
- `experiments/cascade/eval_harvest_head.py` — the clean cross-task refit + head export.
- `experiments/cascade/results/scion176b-harvest-20261018/harvest-refit-head.json` — the clean
  head in the standard head format; consumed by the fusion.
- `experiments/cascade/fusion_v1.py` — the honest fusion (OOF base features + OOF self feature).
- `experiments/cascade/results/fusion-v1-20261018/FUSION-V1.md` — the no-self fusion and the
  post-Window-B update.
- `experiments/cascade/run_self_head_window.sh` — serve-down window script (`WINDOW_OUT` /
  `WINDOW_BENCH` / `WINDOW_NAME`, defaults = Window A).
- `experiments/cascade/probe_scion.py` — the judge probe (checkpoint/resume/retry).
- `scion/decision-heads/PORT-REPORT.md` — port numbers (heads, veto, serving path).
- `strata-dh/` (worktree, branch `decision-head`) — the decision-head build (`/v1/verdict`, `/embedding`).