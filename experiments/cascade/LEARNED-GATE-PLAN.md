# The learned gate (C): scoped plan and the path choice

2026-10-18, off the back of `results/scion176b-self-organic-20261018/
MARGINAL.md` (the measurement: the deterministic stack catches 6/9 stored
wrongs; the learned signals add on top) and its extended head table.
Status: **stages M1–M4 done, M4's acceptance failed → the gate default
stays OFF.**  The path-B scorer is `harness/cascade/learned_gate.py` (22
offline tests reproducing the stored fitted p; live cross-check opt-in via
`HIVE_GATE_LIVE=1`), the runner takes `--gate off|intern-0.8b|scion` with
the blind-spot-only trigger, the quoted finding, the strict re-verify and
the gate report fields (`tests/unit/test_run_cascade_repair.py`, gate
section), M3's replay reproduced the study body bit-for-bit and M4's live
paired run measured the gate on fresh serve data: 1 fresh catch, 0
false-flag damage, +8.7 % wall clock, **3 false flags > the ≤2 budget →
default unchanged (off)** (`results/runner-cascade-gate-nothink-20261018/
RESULT.md`).  M5 (path A) is still plan and needs a serve restart — a
decision that is yours; the false-flag risk this run exposed (~3/66 at
L0.75@0.5 on fresh drafts) is the number M5 would have to beat against the
same rule.

Next (post-M4, no restart): the escalation lane is now wired into the runner
(`experiments/cascade/escalation_lane.py`, `--no-escalation-lane` to opt out)
— every ladder-resolved item is harvested as JSONL next to the report, and the
fresh silent-wrong corpus is audited across all five heads in
`results/fresh-silent-audit-20261018/AUDIT.md` (195 silent drafts: 28 wrong).
The pooled picture is the gate's problem in one table — L0.75 catches 9
wrongs at 8 false flags at 0.5, and with the cap-2 rule only 3 catches / 0
flags; no head separates (wrongs' p spans 0.00-1.00 and overlaps the
corrects' mass).  mlp-concat looks best per-run (armB: 7 catches / 1 flag)
but does not pool (4 / 1) — run-specific, i.e. overfit.  Conclusion
unchanged: do not restart M5; grow the corpus and audit heads before any new
acceptance.  The lane feeds three consumers: (1) gate re-audit via
sample-and-check on escalated tasks, (2) repair SFT on
(prompt, draft, finding, fixed) triples, (3) local-tier distillation of
API/human answers.

## What the gate is

A third trigger beside `consistency` and `code_exec`, with three deliberate
properties:

- **Blind-spot only.**  The gate scores a candidate only when the
  deterministic detectors come up empty — that is where the measured value
  is (+1..+2 of the three detector-missed wrongs per 66, `MARGINAL.md`), and
  it skips the wasted scoring of items already rejected.
- **One more rejection, nothing else.**  A fired gate is a finding like any
  other: the verdict rejects, the same greedy repair quotes the finding, and
  the ladder (human gate, `--api-fallback`) is unchanged.  Gold-free: the
  gate sees question + candidate, never `expect` or `tests`.
- **Strict re-verify.**  After a repair that the gate triggered, the repaired
  answer is re-scored; the finding clears only if the score clears.  A
  persistent flag stays flagged — that is the point of C.

## The score, and the two ways to get it

One interface: `score(question, candidate) -> (p_correct, ms)`.  Both paths
apply the frozen head **client-side** (ridge / MLP over a 2560-d decision
state, plus the Platt pair); neither needs engine changes for the head math.

### Path B — the intern-decision-0.8B readout on CPU (available today)

- Mechanism: `experiments/cascade/decision_readout.py`'s extraction path —
  `internlm/Intern-Decision-0.8B`, bfloat16 on **CPU** (the documented
  ROCm-beside-the-serve hazard is why), one row per (question, candidate)
  through the same row builder the heads were fitted on, then the r2 head
  (`results/decision-readout-r2-20261017/readout-head-intern-decision-
  0.8b-r2.npz`).
- Measured cost: 0.44 s/item median (the 1315-item r2 extraction: 651 s,
  MemAvailable min 18.84 GiB).  No serve touch at all.

### Path A — the 176B serve's own readout (`/embedding`)

- Mechanism (the window's exact path, `probe_scion.py`): `SYSTEM` +
  `readout_judge.order_body({prompt, candidate}, swap=False)` →
  `/apply-template` → `/embedding` (`embd_normalize: -1`) → the frozen
  own-community head.  The runner's task carries the same bare `prompt`
  string the transfer bench used (verified equal), so the fitted prompt is
  reproducible exactly; the format-lock invariant from `SIDECAR-PROTOTYPE.md`
  is satisfied by construction on this path.
- Measured cost: the 140-item transfer embed took 6 m 15 s ≈ 2.7 s/item on
  the GPU, serialized with generation — roughly +3 min on a 66-item run.
  No second body.
- **Blocker (probed read-only 2026-10-18): the product serve 404s
  `/embedding`.**  Enabling it means `--embeddings --pooling last
  --embd-normalize -1` on the `scion-flashnext` ExecStart **and a restart =
  a serve-down window that needs explicit go-ahead**, plus a parity test
  first (the window ran the `strata-dh` build; the product build's states
  must match the fitted representation — embed 5 stored items, compare state
  cosine and head p).

## Operating points at the canonical 0.5 cut

At the head's own boundary, p < 0.5 — the freeze candidate, because anything
tighter is slice-tuned (L0.75's cap0 point needs 0.5912, with the deciding
item 2e-4 from the line):

| source (head) | marginal catch of the missed 3 | extra flags /66 | notes |
|---|---|---|---|
| B: **0.8B-L0.75** | **+1** (`gsm8k-0015:hs2`, p 0.074) | **0** | precision-first default |
| A: 176b-own-community | +2 (`gsm8k-0015:hs2` + `math-0020:hs2`) | 2 | best low-noise recall |
| B: 0.8B-mlp-concat | +2 | 7 | bench-strong, organic-noisier |
| B: 0.8B-L0.5 | +3 (all three) | 9 | recall dial, noisy |

`math-0020:hs2` is the own head's unique edge; `gsm8k-0009:hs2` (p 0.426
under L0.5, ≈1.0 elsewhere) is the recall dial's unique catch.  `L1` (+2 at
5 flags), `ridge-concat` (+1 at 4) and `p_raw` (+0 at 9) are dominated or
empty.

## Recommendation

- **v1 = path B, head L0.75, threshold 0.5.**  No serve risk, precision-
  first, everything already on disk; `--gate off` is the default and the
  undo.
- **Path A as a fast follow, on your go-ahead for the restart** — it buys
  the own-community head's extra catch (+2 vs +1), no second body, and A+B
  together are what the exploratory fusion (0.858 organic LOO) ate.
- The recall dial (L0.5 @ 0.5) stays a flag, not the default: nine extra
  repairs per 66 corrects is against this line's precision-first record.

## Staged plan

1. **M1 — the scorer.**  `harness/cascade/learned_gate.py`: load the 0.8B on
   CPU + the r2 head; `score(question, candidate) -> (p, ms)`; the row
   builder is `decision_readout.py`'s, imported, not re-derived.  Unit tests
   run the head math over stored states (must reproduce stored p exactly);
   one live extraction cross-checks a stored item.
2. **M2 — runner wiring.**  `experiments/cascade/run_cascade.py`:
   `--gate off|intern-0.8b|scion` (default off), `--gate-head`,
   `--gate-threshold 0.5`; a `learned_gate` finding joins
   `deterministic_verdict` *only when detectors are silent*;
   `finding_digest` gains its line; `repair_user_text` gains a template that
   quotes p vs threshold and asks for an internal-inconsistency re-check (no
   gold, nothing else); re-verify re-scores when the gate fired; report
   fields `gate`/`gate_scored`/`gate_fired`/`gate_cleared`; the D3 telemetry
   row carries p.
3. **M3 — offline replay (go/no-go data, nothing served differently).**
   Score the served answers in both committed arm reports
   (`results/runner-cascade-sampled-20261018` and `...-nothink-20261018`)
   and reproduce the 66-item marginal table through the production scorer.
4. **M4 — live paired run.**  Arm-B config ± gate, `--api-fallback` off.
   Pre-registered acceptance: **≥1 marginal catch of a fresh silent wrong at
   ≤2 false flags; no new silent wrongs from false-flag repairs; ≤ ~10 %
   wall-clock overhead** — else keep the gate off.
5. **M5 — path A (green-lit).**  Parity on 5 stored items (state cosine ≥
   0.999, |Δp| ≤ 1e-2; looser needs a head refit on product-serve states),
   then the restart; `--gate scion` becomes the second source behind the
   same interface.

## Costs

| | B (intern-0.8B, CPU) | A (176B `/embedding`) |
|---|---|---|
| per score | 0.44 s CPU | ≈2.7 s GPU, serialized with generation |
| per 66-item run | ≈+30 s | ≈+3 min |
| deployment | none | ExecStart flags + restart (go-ahead) + parity |
| resident body | 0.8B CPU (~2 GB) | none |

## Open questions for scion (path A)

1. Does the product `strata` build support `/embedding` with last pooling and
   `embd_normalize: -1` (the `strata-dh` window build did)?
2. Does enabling embeddings cost generation memory or throughput?
3. Restart routine + rollback (the window script's systemd trap is the
   precedent).

## Risks and expectations

- Fresh-distribution drift: every number above is the stored harvest slice;
  the arms' fresh silent sets differ (`gsm8k-0015:hs2` recurs in both).  The
  threshold is the one dial; live telemetry is the read-out.
- A false flag costs one repair round and, if persistent, a flagged item —
  visible in the human gate; with `--api-fallback` off, nothing escalates
  silently.  The repair-pass control measured 0/8 content harm re-asking
  correct answers; M4 re-checks that on fresh data.
- The 0.8B head is in-domain (train task overlap — the `SELF-HEAD.md`
  caveat).  The universal miss `humaneval-0001:hs1` (min p 0.76) stays a
  miss under every signal measured so far; some wrongs will always remain.

**M5 shelved (2026-10-18):** path A requires modifying strata's serve/server.py
to add an /embedding endpoint for the strata-embd model — strata engineering,
not a hivebench serve restart. The 0.8B readout does not separate on 3
measurements, and M5's cost/benefit (strata-repo work, risk to live serve,
marginal gate value +1 catch) does not justify it. Gate stays opt-in-only.
Revisit only if a strata embedding endpoint lands.
