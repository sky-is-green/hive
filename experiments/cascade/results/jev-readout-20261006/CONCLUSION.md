# JEV-READOUT — conclusion (2026-10-08)

Internal closing record for the decision-head line.

**Question:** can a fitted readout on a generation model's own hidden states
cheaply improve its outputs — accuracy first, else compute at no accuracy
loss?

## The answer, in one paragraph

Yes, but modestly and mostly as *selection*.  The model's own correctness
signal is readable at AUC .86–.90 from the judge prompt's last hidden state
(and .86–.88 from its raw answer-slot distribution alone, zero training); a
frozen head turns that into best-of-N accuracy at ~half the tokens via gating,
and +5–7 tasks over majority via the veto, at zero extra generation.  The
engine-side packaging is exact (Δ≈1e-8).  It makes nothing smarter: samples
stay as good as the model makes them, most oracle headroom remains, and early
cancellation is ruled out by a small ceiling.

## Established

1. **Readout ≫ prompted text** (J-Bench): probe 0.999 vs prompted 0.90;
   own-answer self-check catches 7/28 errors vs the head's 23/28
   (`COMPARISON.md`, `SCION-HEAD-REPORT.md`).
2. **Readout class generalizes**: trained-low-bit bodies keep it (Scion 0.999,
   Bonsai 0.988, dense + MoE); post-hoc 2-bit breaks it (Clef QAT 0.712,
   h128 0.648).
3. **Selection/gating**: gated sequential sampling = best-of-3 accuracy at
   ~half the pool cost (Bonsai 311k vs 628k tokens); correction study +5 over
   single at honest pool cost; retry @0.87 +7 over single (sequential cost).
4. **Veto (majority + head override)**: Scion 154→161 (+7, 9 fixes/2 breaks),
   Bonsai 168→173 (5/0), raw-margin variant 174 (6/0); pooled **14 fixes / 2
   breaks over 398 held-out tasks (exact p=.004)**.  Zero extra generation.
5. **Raw answer-slot margin**: strong zero-training signal (.880 Scion / .858
   Bonsai); clean blends best on Scion; no signal dominates on both models —
   per-model validation should pick the policy.
6. **Sidecar**: engine parity Δ≈1e-8 after fixing pooled-embedding extraction
   and applying the embedded correction adapter (`llama_set_adapters_lora`);
   batch readout (hidden + token probabilities) added to the harness.
7. **Monitor**: real trajectory signal in generation context (.79–.83 early,
   .93–.99 late; Qwen .87–.99), useful as an **end-of-trace verifier**.

## Not established / falsified

- Marked accuracy improvements: no — the veto gives +2.5–3.5 points, and the
  rest of the oracle gap is pool knowledge the head cannot conjure.
- Head over raw self-assessment: no universal edge (raw wins on Scion, frozen
  head on Bonsai, blends help one of two).
- Early cancellation: no (oracle ceiling 21–34%; realistic 10–19% savings at
  3–15% false cancels; thresholds do not transfer).
- Trajectory-based selection: no better than the raw margin where headroom
  exists (Scion); matches oracle only on a saturated split (Qwen).
- Fine-tuned decision models: not competitive for this task (external:
  LiquidAI d1-3B regressed against its own base on ImajevBench, 64.0 vs 66.8).

## Where it pays

Self-hosted / edge sampling loops that already generate several candidates and
have (or can fit) correctness labels: ~2× token efficiency and a small
consistent veto gain, with calibrated per-model verification.  It does not
change single-shot chat, frontier serving, or tasks with stronger checkers.

## Status

All local — nothing published or pushed.  Fork changes committed locally
(decision-head sidecar + adapter fixes).  This directory holds every artifact
needed to regenerate the results.
