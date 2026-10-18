# Continuation — after Fusion v1 (2026-10-18)

Fresh-session handoff.  Paste the "Start here" section (or this whole file) as the first
message of the next session.

---

## Start here

Repo `hivebench`, branch `hive_cascade` (one step per commit; **never** `git push`).
The 176B serve is up on 8081 (user unit `scion-flashnext`) — keep it up; any serve-down
window needs the user's explicit go-ahead first, and the window script has a trap that
always restores the service.

Fusion v1 has run (`experiments/cascade/results/fusion-v1-20261018/FUSION-V1.md`): the
fusion gate fit honestly on the r2 train split with `GroupKFold(5)`-by-task out-of-fold base
probabilities, one pass on test-182.  Twelve seconds, CPU only, no serve contact.

Three results worth carrying:

1. **The honest gate is `L0.5+MLP`** — test-182 **0.978**, bench-116 0.991, organic-66 **0.852**
   (frontier 6/9 caught at 9 of 57 correct rejections).  But it beats the single 0.8B MLP head by
   +0.008 organic AUC, i.e. nothing.  Reweighting the existing heads is not the lever.
2. **The previously reported best feature set was a LOO artifact.**  `L0.5+MLP+p_raw` scored 0.858
   organic under leave-one-out on the test set itself and **0.827** honestly fitted; `p_raw` has
   organic AUC 0.603 alone.  All six base features: 0.819.  Treat the fusion tables in
   `FUSION-PROBE.md` / `SELF-HEAD.md` as upper bounds; LOO overstates by ~0.03-0.04 organic.
3. **The 176B self-community head is the only feature that adds** — +0.03 organic AUC in all three
   fit subsets (1,133 / 332 / 262 / 70 rows), and it is the only instrument in the whole cascade
   that catches `humaneval-0001:hs1`, a "universal miss" every 0.8B/Bonsai signal sits above 0.87
   on.  `self-jbench` is a trap: +0.007 bench, −0.03 organic (the fusion over-weights it at
   coefficient 3.03 because it is near-perfect on the bench rows it is fit against).

## The Window B decision (needs the user's go-ahead — do not start it)

Measured here: `176b-own-community` was fit on the port community corpus, whose train split covers
465 tasks.  **61% of organic-66 (40/66 items, 7 of 9 wrongs) is in-domain for that head.**  The
clean cross-task organic slice is 26 items with **2** wrongs — both `humaneval-0001`.  So the
cross-task value of the generator's own head is currently *unmeasured and unmeasurable offline*:
the +0.03 is 61% in-domain, and the honest fusion cannot give the self signal a trustworthy weight
because it has no organic harvest train row (it is fit on 332 bench/qa rows, 79% bench).

Window B (937 harvest items, ~50-55 min serve-down, same `run_self_head_window.sh` pointed at the
full harvest bench) is the only step that fixes both: the 176B signal on the organic train side,
and a harvest-refit head whose test tasks are disjoint from its train tasks.  Expected outcome is
not a number — it is confirming or killing the self signal.

## Which lines are closed

- Dense partner (Bonsai-27B) as an organic judge: **closed** (0.624 / 0.630 organic, chance).
- Bench is saturated for every bench-fitted head (0.988-0.994).
- Natural-error detection tops out ~0.83-0.86 per-sample across instruments.
- **Fusion by reweighting the existing signals: closed** (honest v1 = best single head + 0.008).
- LOO-on-the-test fusion as an evaluation scheme: closed; it inflates by ~0.03-0.04.

## What remains

- **Window B** (optional, ~50-55 min serve-down, ask first) — see above.
- **API provider** for the escalation tier: still unnamed by the user; blocks the production API tier.
- Human-gate coverage of the router's pre-generation `gate -> API` path; bench ingest builders;
  9B Qwen ternary (deferred).

## Next moves (in order)

1. Ask the user for the Window B go-ahead (nothing else in the offline line is open).
2. On a yes: run `run_self_head_window.sh` against the full harvest bench, then refit the community
   head on 871 harvest train items and re-run `fusion_v1.py` — the clean cross-task organic number
   is the deliverable, and it decides whether `self-community` goes in the deployed gate.
3. Keep pending: name the API provider; human-gate coverage of the router path; bench ingest
   builders; 9B Qwen ternary.

## Hard constraints

- Never `git push`; one step per commit.
- Do not touch `scion/docs/HARVEST-NEMOTRON-FLASHNEXT.md` or `scion/docs/DRAFT-nemotron-field-notes.md`.
- Serve windows: ask first; the script's EXIT trap always restarts `scion-flashnext` and waits for health.
- No large artifacts in `/tmp`; durable outputs go to the work-drive lane dirs.
- Partner stays CPU-preferred (no persistent dGPU use); no torch-ROCm context beside the serve.
- Runtime corrections must be gold-free (`code_exec`, `constraint_check`, `schema_ast`, `consistency`);
  `numeric_gold` / `choice_made_gold` are eval-only.
- Before any API escalation, ask the user; two top-level choices only: solve locally or escalate.
- Human gate: first trigger verbose with full evidence, later triggers compact; outcomes exactly
  `solve locally` or `escalate to API`; `keep` only as a solve sub-option when the checker already passes.

## Key files

- `experiments/cascade/fusion_v1.py` — this step's analysis (rerunnable, ~12 s).
- `experiments/cascade/results/fusion-v1-20261018/FUSION-V1.md` — report; `fusion-v1.json` — numbers.
- `experiments/cascade/results/scion176b-self-organic-20261018/SELF-HEAD.md` — Window A report.
- `experiments/cascade/run_self_head_window.sh` — serve-down window script (rerunnable; Window B = same script).
- `experiments/cascade/fusion_probe.py`, `eval_self_head.py` — the earlier LOO fusion probes.
- `experiments/cascade/results/decision-readout-r2-20261017/` — r2 states/heads/splits (0.8B reference).
- `scion/decision-heads/PORT-REPORT.md` — port numbers (heads, veto, serving path).
- `strata-dh/` (worktree, branch `decision-head`) — the decision-head build (`/v1/verdict`, `/embedding`).