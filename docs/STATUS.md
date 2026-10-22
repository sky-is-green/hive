# Status — 2026-10-22

**This is a research harness, not a usable product yet.** The reference
serves are down or reserved for engine work, and the paid tier's key is
expired. The free OpenCode API tier is measured separately. What is
published is the measurement record, the code that produced it and the
contracts the next phase is built against.

## Measured (the same 28 organic tasks unless stated)

| the arm | the result | the cost |
|---|---|---|
| the 35B alone | 21/28 (0.750) | the cheapest local |
| **Bonsai 2 27B** | **26/28 (0.929); +5 net** (6 fixed, 1 broken) | 8.92 s/item |
| the 176B generating | net 0 (2 fixed, 2 broken) | 20.68 s/item |
| the 176B head judging the 35B | AUC 0.864; 3 of 7 errors, **0 of 21** correct sent up | 2.52 s/verdict |
| **the checker (execution + arithmetic)** | **6/7 errors; 4/4 at p >= 0.97; 0/21 false rejects; 17/17 mutations** | **0.23 ms median; 30 ms p95** |
| the judge + Bonsai | 23/28 (0.821); 3 escalations, all errors | - |
| the head + live 27B/API | 24/28 (0.857); 3 escalations; **0/21** correct sent up | 10.21 s mean; 17.81 s conditional |
| the oracle gate + Bonsai | 27/28 (0.964) | - |
| **the API (DeepSeek V4.1 Flash Free)** | **28/28 (1.000)** | 2.608 s median; free tier, not a paid cost |
| **the scripted controller (checker + head + 27B + API)** | **28/28 (1.000); 7 escalations (5 27B, 2 API); 0/21** correct sent up | **9.24 s mean; 17.22 s conditional** |
| **the API in the 27B slot (DeepSeek V4.1 Flash Free)** | **28/28 (1.000); 7 API escalations; 0/21** correct sent up | **8.57 s mean; 14.54 s conditional** |
| the API in the 27B slot (GLM 5.3 Flash Free) | 27/28 (0.964); 7 API escalations; **0/21** correct sent up | 7.80 s mean; 11.45 s conditional |

The latency decomposition and the resident tier are in
`docs/AB-RESULTS.md` §1B and §1B-bis: cold 33.8 s vs 17.3 s, warm
84.7-94.3 → 93.3-101.9 tok/s, storage traffic 54-261 MB → 0.6-2.4 MB.
The curation replay (§2C-quater): the stock encoder keeps 60.51% of the
facts in 374 tokens; the tuned dedup encoder keeps 32.66% in 139. The
dedup decision cannot be moved; the drift threshold is the free dial.

## Invalidated (do not cite these)

- **The drift numbers in `AB-RESULTS.md` §2C through §2C-sexies.** The
  corpus behind them was 79% an ingested knowledge-base attachment and
  one conversation was loaded twice. The drift AUC 0.957/0.945, the
  student's 0.457, the fine-tune's 0.277 and the head's 0.994 were
  measuring "short query against a document", not topic drift. The
  correction is §2C-septies; the clean corpus cannot support the split
  (2 drift cases in train, 176 in the hold-out).
- **The encoder card's deployment recommendation.** It used to advise a
  dedup threshold of 0.67; the replay prices that as a fact deletion.
  The checkpoint is public as a research record, marked not-a-release.

## Unmeasured (the next phase)

- the **grounding verifier**: the execution checker is measured (6/7,
  4/4 at `p >= 0.97`, 0/21, 0.23 ms); the citation path is implemented but
  not measured on a corpus with claims.
- the **API-driven controller** (the API tier and the scripted controller
  are measured; the frontier-driver direction is not) and the **drift
  benchmark** (being rebuilt: size-matched windows, conversation-disjoint
  split).

## Blocked

The paid tier: the `opencode-go` key is expired, so the paid API cost is
unmeasured. The free OpenCode tier is measured at 28/28; it is not a paid
cost measurement.

## Where to start

| the document | what it is |
|---|---|
| [`AB-RESULTS.md`](AB-RESULTS.md) | the campaign's log, with the corrections and the negative results |
| [`cascade/CASCADE-PAPER.md`](../cascade/CASCADE-PAPER.md) | the method and the measured tier results |
| [`cascade/HARNESS-CONTROLLER.md`](../cascade/HARNESS-CONTROLLER.md) | the drop-in controller contract: four tools, four invariants, the bounded loop |
| [`cascade/CONTROLLER-PRIOR-ART.md`](../cascade/CONTROLLER-PRIOR-ART.md) | what other harnesses do for verification and routing |
| [the memory card](https://huggingface.co/SkyIsNotGreen/splinter-memory) | the memory system, its limits and the correction |
| [`splinter/docs/ENCODER-PLAN.md`](../splinter/docs/ENCODER-PLAN.md) / [`splinter/docs/ENCODER-CARD.md`](../splinter/docs/ENCODER-CARD.md) | the encoder programme and the research checkpoint |

## The published artifacts

| the artifact | where | the state |
|---|---|---|
| **the cascade paper** | `github.com/sky-is-green/hive/blob/main/cascade/CASCADE-PAPER.md` | public; the canonical text, versioned with the code and the run bundles |
| **the memory paper** | `github.com/sky-is-green/hive/blob/main/splinter/SPLINTER-WHITE-PAPER.md` | public; the long-form paper, with the HF edition in `splinter/SPLINTER-MEMORY.md` |
| **the page** | `huggingface.co/spaces/SkyIsNotGreen/hive-cascade-paper` | public, a static Space: seven tabs (premise, memory, routing, controller, measured, checker, paper) with a per-tab example stage - the drone lab (the query, the relevance and the selected context, vectors, cosines, a live dedup threshold), the routing map, which pauses on hover and explains each correspondence in a tooltip, and the controller hub - every tool call leaves and returns to it, with small packets in both directions and the measured time per stage, and the hero strip that echoes it, including the memory/prompt packet exchange, the score bars, the mutation test - the four tools and the measured table. The paper tab points at the repository, and the Space `README.md` is a short card. Built by `scripts/hf_space.py`; the Blog Article format needs PRO |
| the memory system card | `huggingface.co/SkyIsNotGreen/splinter-memory` | public |
| the encoder checkpoint | `huggingface.co/SkyIsNotGreen/splinter-encoder` | public, **research checkpoint** |
| the task sets | `huggingface.co/datasets/SkyIsNotGreen/hive-cascade` | public; a **dataset card** (the two sets, fields, sources), not the paper |

The page is generated by `scripts/hf_space.py`; the papers live in
`cascade/` and `splinter/`, and the dataset card is `docs/HF-DATASET-CARD.md`. If the
account gets PRO, the paper can be posted as a Blog Article and will then
appear in the sidebars of the three repos it mentions.

The corpus's local paths were replaced with `<work>`, `<drive>` and
`<drive-id>` in the published tree; the git history predating that commit
still contains them (rewriting it would need a force-push).
