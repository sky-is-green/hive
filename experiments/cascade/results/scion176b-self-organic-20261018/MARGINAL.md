# What a learned gate adds over the detectors (marginal catch)

2026-10-18.  The blind-spot question from the runner runs, answered offline
on the stored harvest-organic-66 candidates — **no serve, no model runs**:
the deterministic detectors (`consistency` + `code_exec`) are recomputed on
the stored drafts, the learned signals come from the states and readouts
that already exist on disk (`probe-states-scion176b-transfer.npz`, the two
frozen 176B heads, the r2 0.8B readouts), and the question is only what each
signal adds **beyond** the detectors.

Reproduce: `.venv/bin/python experiments/cascade/blindspot_marginal.py --out
experiments/cascade/results/scion176b-self-organic-20261018/marginal-catch.json`

## Detectors first (the baseline)

Caught **6 of 9** wrongs (`code_exec` 4, `consistency` 2) at **0 false flags
on the 57 corrects**.  The three they miss — the blind spot — are
`gsm8k-0009:hs2`, `gsm8k-0015:hs2`, `math-0020:hs2`.

## Marginal catch over the detectors

`cap k` = the learned signal may flag at most `k` corrects the detectors did
not flag; among the detector-missed wrongs it then catches:

| signal | cap0 | cap1 | cap2 | cap3 | total (all 9 wrongs) @0.5 |
|---|---|---|---|---|---|
| **176b-own-community** | **+1** (0 flags) | **+2** (1 flag) | +2 | +2 | 4/9 caught, 2/57 CR |
| 176b-own-jbench | +1 (0 flags) | +1 | +2 (2 flags) | +2 | 0/9 caught, 0/57 CR |
| **0.8B-L0.75** | **+2** (0 flags) † | +2 (0 flags) | +2 (0 flags) | +2 (0 flags) | 3/9 caught, 0/57 CR |
| 0.8B-mlp-concat | +0 | +2 (1 flag) | +2 | +2 | 5/9 caught, 7/57 CR |
| 0.8B-L0.5 | +0 | +1 (1 flag) | +2 (2 flags) | +2 | 6/9 caught, 9/57 CR |
| 0.8B-L1 | +0 | +1 (1 flag) | +2 (2 flags) | +2 | 4/9 caught, 5/57 CR |
| 0.8B-ridge-concat | +0 | +0 | +1 (2 flags) | +1 | 3/9 caught, 4/57 CR |
| 0.8B-p_raw | +0 | +0 | +0 | +0 | 0/9 caught, 9/57 CR |

† L0.75's cap0 threshold is 0.5912 and the second catch, `math-0020:hs2`,
sits at 0.5910 — 2e-4 of margin, so that particular +2 is slice-tuned.  At
the canonical 0.5 cut (the head's own boundary) it catches one of the two,
`gsm8k-0015:hs2` (p 0.074), with 0 flags.  The freeze discussion lives in
`LEARNED-GATE-PLAN.md`.

Items, at the operating points that matter:

- `176b-own-community` cap0 bills nothing and catches `math-0020:hs2`;
  cap1 adds `gsm8k-0015:hs2` and pays one flag — `math-0020:hg`, the
  *correct* greedy sibling of the wrong sampled item.
- `176b-own-jbench` is the mirror image: cap0 catches `gsm8k-0015:hs2`;
  cap2 adds `gsm8k-0009:hs2` paying `humaneval-0015:hg` + `truthfulqa-0015:hg`.
- The two own heads are complementary on exactly this slice (as
  `SELF-HEAD.md` said): `own-community` at cap0 plus `own-jbench` at cap2
  catch **all three** missed wrongs between them, at two extra flags on 66
  items.
- The 0.8B head **L0.75** is the precision leader: +2 of the missed wrongs at
  zero flags (†) and no correct-rejections at the canonical 0.5 cut either
  (+1 there).  `mlp-concat` reaches +2 at one flag; `L0.5` is the recall
  dial — the only signal that also catches `gsm8k-0009:hs2`, at nine flags.
- The own heads ride the serve (one extra serve forward per score); the 0.8B
  heads need a second, CPU-side body.  The source/head choice for the gate is
  worked out in `LEARNED-GATE-PLAN.md`.

## Verdict against the pre-registered rule

Rule: *≥2 catches of detector-missed wrongs at ≤2 extra flags → a scoped
learned-gate plan is justified; 0–1 → accept the blind spot.*

**Pass.**  `176b-own-community` at cap1 gives +2 catches for one extra flag
per 66 items; even cap0 is strictly positive at zero cost.  Wiring a learned
gate as a third trigger (finding → the existing greedy repair → human gate /
API) is justified.  The path choice — the serve's `/embedding` needs a
green-lit restart (it 404s today; probed read-only 2026-10-18) while the
intern-0.8B CPU readout needs nothing — is made in `LEARNED-GATE-PLAN.md`,
with `0.8B-L0.75` as the precision default the extended table surfaced.

## Caveats

- n=9 wrongs on stored (harvest) drafts.  The fresh runner arms have
  different silent sets (arm B: `gsm8k-0015:hs2`, `gsm8k-0033:hs1`,
  `gsm8k-0036:hs1`, `math-0023:hg`, `truthfulqa-0015:hg`) with overlap
  (`gsm8k-0015:hs2` in both; old `math-0020:hs2` ~ fresh `math-0020:hg`).
  Read the +2 as "the direction is measured, the per-run gain is 1–3".
- A false flag on a correct answer now costs one local repair round (~6 s
  without thinking; ~67 s with), and if the repair does not clear it the
  item reaches the human gate — so a *persistent* false flag is visible
  friction, not silent waste.  The measured repair arm saw 0/8 content harm
  when re-asking correct answers, but that is one small control.
- The own head is in-domain (train task overlap, `SELF-HEAD.md` caveat);
  a clean cross-task number would need the Window B refit.
- `gsm8k-0009:hs2` is caught only by `own-jbench` at cap2 (or the L0.5
  recall dial, at nine flags) — some wrongs will remain under every signal
  measured so far.

Files: `marginal-catch.json` (machine-readable), `blindspot_marginal.py` (the
analysis), `SELF-HEAD.md` (head provenance).
