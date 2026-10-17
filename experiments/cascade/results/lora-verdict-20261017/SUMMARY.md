# LoRA verdict adaptation — round-2 corpus (pre-registered)

**Result in one line:** the adapter learned J-Bench-style verification almost
perfectly (bench AUC 0.998, 115/116, 1 false accept, 86/87 wrongs caught) but
that did **not** transfer to organic 176B errors: organic-test AUC sits at
0.746 in both epochs, with 0/9 and 3/9 wrongs rejected at 0.5 and probabilities
compressed near 1. The pre-registered verdict is **FAIL** (organic AUC < 0.80,
caught ≤ 4/9). The frozen-feature heads remain the best gate.

## Design

Verdict cross-entropy at the decision marker (`yes` 9405 / `no` 2083) over the
exact deployment prompt (`verdict_request` → `validate_request` → `backend.encode`).
1,133 train rows: J-Bench train 262 (68 correct / 194 wrong) + harvest-r2 train
871 (410 organic correct / 461 wrong: 43 organic + 418 perturbations);
478 correct / 655 wrong overall, and the organic subset is 90.5% correct.
Rank 16, alpha 32, dropout 0.05, lr 1e-4, accum 8, seed 1337; language
`q/k/v/o_proj`, `gate/up/down_proj`, linear-attn `in_proj_qkv`, `in_proj_z`,
`out_proj` (10.2M trainable, 1.18%); vision frozen. CPU bf16 beside the live
176B; 2 epochs, **5,156 s (86 min)**, 2.08–2.17 s/item. Serve `/health` 200 at
launch and after; the only concurrent work was a 1-item adapter-path smoke test
(MemAvailable never below 10.9 GiB).

Rules, the reference table, and the selection rule were fixed in
`PRE-REGISTRATION.md` while training was running, before any eval was seen.

## Results

| epoch | bench AUC | bench acc | FA/FR | organic AUC | organic acc | FA/FR | wrongs caught | ext FR |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.9960 | 113/116 | 3/0 | 0.7456 | 56/66 | 9/1 | 0/9 | 0/4 |
| 2 | 0.9976 | 115/116 | 1/0 | 0.7456 | 60/66 | 6/0 | 3/9 | 0/4 |
| final | 0.9976 | 115/116 | 1/0 | 0.7456 | 60/66 | 6/0 | 3/9 | 0/4 |

The pre-registration is silent on exact AUC ties; the code's `max` names
**epoch 1** because both epochs tie at 0.7456. At the operating point epoch 2
dominates (same rank ordering, +2 bench, 3 catches instead of 0, no correct
rejections), so **epoch 2 is the adapter to keep**. The verdict is FAIL for
either epoch.

Frozen references (commit `7bfa7fb`): raw verbalizer organic AUC 0.603 (0/9
caught); refit MLP 0.844 (5/9); L0.5 0.827 (6/9); L0.75 0.819 (3/9).

## Why it failed

- **The bench gain is real and large:** 0.771 → 0.998 AUC, 86/87 bench wrongs
  caught — better than any frozen head (0.991). The objective works on the
  perturbation/verification task it was trained on.
- **Organic rank ordering does not move between epochs** (identical AUC to four
  decimals) while the confidence level does: wrongs' median p fell 0.971 →
  0.924 and corrects' 0.996 → 0.988. The adapter's organic effect is mostly a
  level shift on a fixed rank function that plateaus at 0.746 — above the raw
  readout (0.603) but below the frozen MLP head (0.844).
- **Style prior:** the organic train slice is 410 correct / 43 wrong (90.5%
  positive), and the model learns "176B-style answer ⇒ yes": at epoch 2 the
  wrongs' median p is 0.924 and the model rejects no correct answer (FR 0).
  The 418 perturbations and J-Bench errors teach the perturbation signature,
  which transfers to the bench test but not to organic errors.
- **Misses:** `math-0013:hs1` (p 0.991) and `humaneval-0001:hs1` (p 0.965)
  remain invisible — the same two universal misses. Epoch 2 newly catches
  `humaneval-0001:hs2` but loses `humaneval-0015:hs2` vs the frozen heads: a
  different failure subset, no net operating-point gain.
- Rule 5 (adapted-backbone refit) is not triggered: it required a rule-2 pass.

## Where that leaves the gate

| operating point | organic wrongs caught | corrects rejected |
|---|---|---|
| MLP head (r2) | 5/9 | 7/57 (12%) |
| L0.5 probe | **6/9** | 9/57 (16%) |
| L0.5 + MLP union | **6/9** | 14/57 (25%) |
| L0.75 probe | 3/9 | **0/57 (0%)** |

The LoRA adapter is not a gate candidate. The pre-registered failure rule says
to stop representational work and choose an operating point from the existing
heads; the choice is an escalation-cost trade-off:

- **L0.75 first tier** is free: 3/9 wrongs escalated, no correct answer ever
  blocked. A second tier (MLP or L0.5) buys the remaining catches at 12–16%
  false rejections.
- **L0.5 alone** is the best recall point: 6/9 caught, 16% of correct answers
  escalated; it misses `gsm8k-0033:hs1` (p 0.525), `humaneval-0001:hs1`, and
  `math-0013:hs1`.

If the representational line is kept open, the specific untested hypothesis is
**class-balanced LoRA** — oversample the organic wrongs so the adapter cannot
use the 90.5%-positive organic prior. That directly targets the observed
failure mode, but it is outside the pre-registered pass chain and needs its own
pre-registration and ~90 min CPU.

## Artifacts

- `lora-intern-decision-0.8b-lora-r2.json` — config, per-epoch eval blocks,
  per-item p for bench/organic/extension.
- `lora-comparison.json` — pre-registered selection/verdict, frozen references,
  epoch table.
- `PRE-REGISTRATION.md` — rules fixed before the eval.
- Adapters (3 × 40 MB) on disk, gitignored: final + epoch1 + epoch2 snapshots;
  `adapter_config.json` / README are committed as provenance.
- `lora-run.log` — ignored.
