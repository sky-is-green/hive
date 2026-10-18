# WINDOW smoke-4 — both tiers on strata, two cards, no contradiction

**Date** 2026-10-18 15:07–15:09 · `results/dual-generation-20261018/` (4-item slice; full 66 in `full-66/`)

## The runtime question, settled

Bonsai is a **dense `qwen35`** GGUF; the 35B pack is **`qwen35moe`**. They are two families to
strata, and the family enum on the serving branch (`scion-0.1.44`) names only
`qwen4exp` (Flash-Next) and `qwen35moe` (Qwen3.6-35B-A3B), so `strata-q36` refuses Bonsai by name:

```
strata-q36: architecture is 'qwen35', this engine requires 'qwen4exp'
```

The dense work exists on branch **`bonsai2-dense`**, checked out at
`/work/strata-bonsai2`, and is a *separate binary*: **`strata-dense`** (built Oct 14), which
"reads the GGUF blocks as they are" — no pack step, `--native <model.gguf>`, behind the same
`serve/server.py` (the two worktrees' copies are byte-identical) and the same HTTP protocol.

| tier | engine | worktree | port |
|---|---|---|---|
| 35B-A3B `qwen35moe` | `strata-q36` | `work/strata` | 8082 |
| Bonsai-2 27B dense `qwen35` | `strata-dense` | `work/strata-bonsai2` | 8083 |

No llama.cpp anywhere; both tiers are strata, as required.

## Co-residency: it holds, with no contention

| engine | lone | both churning |
|---|---|---|
| 35B-A3B (card A) | 116.8 tok/s | **113.9 / 116.3 / 117.5 / 117.6 tok/s** |
| Bonsai-2 27B (card B) | — | **63.4 / 64.4 / 66.2 / 66.2 tok/s** |

Neither engine lost throughput to the other. Load: dense 27B healthy in **8 s**, 35B in **19 s**
(the 176B serve needs ~32 s). VRAM after both: `card0 12794 MiB free`, `card1 4227 MiB free` —
one model per card, no overlap.

**Ordinals are not what the config says.** The dense 27B (`"gpu": [1]`) settled on the card sysfs
calls *card0*, and the 35B (`"gpu": [0]`) on *card1*: HIP's ordinal order is the reverse of
`/sys/class/drm/cardN`. Both engines *did* respect `HIP_VISIBLE_DEVICES` — each had a private
card — but anyone reading placement off sysfs will misread which model sits where. If exact
placement ever matters, verify with the engine's own device list, not sysfs.

## The quality smoke, and the trap in it

- **35B: 4/4 correct** (0 wrong, nothing flagged).
- **Bonsai: 2/4 "wrong" — both were the 320-token cap, not reasoning.** Its two cut answers end
  mid-derivation (`… 4. Subtract 8 from both sides:` and nothing), so no `Final answer:` was ever
  emitted, `final=None`, and the checkers scored them False. The two it finished, it got right and
  agreed with the 35B.

The checkers were doing their job — an unfinished derivation *is* inconsistent — but scoring a
budget as a wrong answer measures the harness. Hence `finish_reason` and `completion_tokens` are
now recorded per model per item, and the summary carries `budget.hit_cap` / `budget.no_final`.

## What this implies for the cascade (before the 66)

**The smaller model is the slower one.** Bonsai (27B dense, 27B active per token) runs at 65 tok/s
and writes ≥239–320-token chains of thought; the 35B (A3B, 3B active) runs at **116 tok/s** and
finishes in 87–212 tokens. "Smallest first" as a *latency* strategy is a trap: per item, Bonsai
costs ~2× the wall time of the much larger 35B. Ordering should follow measured latency × solve
rate, not parameter count — which is what the 66-item run is for.

Latency at temperature 0 also *is* a signal we paid for: the two `hs1`/`hg` variants of one task
produced identical text on both engines, so item-to-item disagreement here reflects task hardness,
not sampling noise.
