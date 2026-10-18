# Continuation — after Window A (2026-10-18)

Fresh-session handoff.  Paste the "Start here" section (or this whole file) as the first
message of the next session.

---

## Start here

Repo `hivebench`, branch `hive_cascade` (one step per commit; **never** `git push`).
The 176B serve is up on 8081 (user unit `scion-flashnext`) — keep it up; any serve-down
window needs the user's explicit go-ahead first, and the window script has a trap that
always restores the service.

Window A has run: the 176B's own judge states for the 140-item harvest transfer subset
were pulled through the decision-head build (`strata-dh` server on 8897, the established
port pattern).  States (gitignored, on the work drive):
`experiments/cascade/results/scion176b-self-organic-20261018/probe-states-scion176b-transfer.npz`
(140×2560, L1 = unembedding input, same readout as the port).

Result line: the generator's **own in-domain head is a real organic instrument — 0.803 AUC on
organic-66** (catch@0.5 4/9), far above the two Bonsai cross-family heads (0.624 / 0.630) and
~4 points under the 0.8B MLP (0.844); it is free at serve time (one engine pass, no second body).
The own J-Bench head is the complementary QA instrument (qa-75 0.979, organic 0.671).  Adding the
self signal to the fusion reproduces the previous best organic number (0.858) and test-182 0.975,
**without the Bonsai body in the stack**.  Window: 7 min 13 s, 2.7 s/prompt, service restored
healthy; 4 shared seed states bit-identical across windows.  Full numbers + caveats in
`SELF-HEAD.md`; do not confuse the probe's own 4-seed ridge printout (0.780) with an instrument.

## Which lines are closed

- Dense partner (Bonsai-27B) as an organic judge: **closed**.  Frozen head 0.624, community
  in-domain head 0.630 organic AUC — chance on the 176B's natural errors; cross-generator
  transfer failure, not an under-trained head.  The 7 h in-domain extraction is not justified
  as a judge.  Any dense-body role is generation/repair, not verification.
- Bench is saturated for every bench-fitted head (0.988–0.994).  Natural-error detection tops
  out at ~0.83–0.86 per-sample across instruments.

## What remains

- **Fusion v1** (the purpose-trained fusion gate, the user's "weigh the probabilities"):
  train on the r2 train split (1,133 items) with out-of-fold base probabilities, validate once
  on test-182, report per family.  The 176B own-head signal exists for bench-116 (port states)
  and organic-66 + qa-75 (Window A); it does **not** exist for the train split.
- **Window B** (optional, ~50–55 min serve-down, ask first): embed all 937 harvest judge prompts
  → clean harvest refit (871 train / 66 test) + the 176B signal for every fusion candidate.
  Same script as Window A, just point `--bench` at the full harvest bench.
- **API provider** for the escalation tier: still unnamed by the user; blocks the production API tier.
- Human-gate coverage of the router's pre-generation `gate -> API` path; bench ingest builders;
  9B Qwen ternary (deferred).

## Next moves (in order)

1. Fusion v1 offline (no window): out-of-fold features on the r2 train split, single validation on
   test-182; variants with and without the 176B own-head signal (community + J-Bench) where it exists.
2. Report per family and decide from that whether Window B is worth the 50–55 min
   (it is the only way to give the fusion the 176B signal on the training side).
3. Keep pending: name the API provider; human-gate coverage of the router path; bench ingest builders;
   9B Qwen ternary.

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

- `experiments/cascade/results/scion176b-self-organic-20261018/SELF-HEAD.md` — Window A report.
- `experiments/cascade/results/scion176b-self-organic-20261018/self-head-eval.json` — eval numbers.
- `experiments/cascade/run_self_head_window.sh` — serve-down window script (rerunnable; Window B = same script).
- `experiments/cascade/eval_self_head.py` — applies the two 176B heads to window states + exploratory LOO fusion.
- `experiments/cascade/fusion_probe.py` — LOO fusion over the existing signals.
- `experiments/cascade/results/fusion-probe-20261018/FUSION-PROBE.md` — previous report + window plan.
- `experiments/cascade/results/decision-readout-r2-20261017/` — r2 states/heads/splits (0.8B reference).
- `scion/decision-heads/PORT-REPORT.md` — port numbers (heads, veto, serving path).
- `strata-dh/` (worktree, branch `decision-head`) — the decision-head build (`/v1/verdict`, `/embedding`).
