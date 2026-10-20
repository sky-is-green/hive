---
license: mit
tags:
  - llm
  - self-verification
  - cascade
  - inference
  - research-paper
  - local-first
---

# The Hive Cascade: detect → repair → escalate

**A gold-free verification-and-repair loop for local language models.**

A model that is wrong is rarely *uncertain* - it is confidently wrong, and the
confidence is indistinguishable from the confidence of a right answer. Every
LLM application inherits this problem: a wrong answer costs the same as a
right one, unless something checks it.

The Hive Cascade is a serving loop that checks every answer before it is
served, repairs what can be repaired, and escalates what cannot - using
**verifiers that never see the answer key**.

This card is the paper: the method, the protocol, the measured results, and
the negative results we kept.

---

## 1. The problem

Self-verification in language models is usually a model asking itself whether
its answer is right, and a judge asking the same. Both are unreliable in the
same direction: the failure modes of a generator and a judge trained on the
same distribution are correlated, so the cases that are hardest for the
generator are usually the cases the judge waves through.

Two facts make the problem sharper:

1. **The detector is not the generator.** A wrong answer that is internally
   inconsistent (the arithmetic does not add up) or whose code does not pass
   the task's own tests is detectable by *running something*, not by asking a
   model. A 176B model's arithmetic slip is caught by a calculator.
2. **The check must not be the answer key.** If the verifier compares the
   answer to the expected answer, the loop is a benchmark, not a system. It
   cannot run in production, where there is no expected answer, and it
   cannot measure anything, because the detector is then trivially perfect.

The cascade is built around those two facts.

## 2. The method

Per item, the loop is four stages:

```
solve  →  detect  →  repair  →  escalate
```

**solve.** One request to the local model at the run's temperature. A draft
that hits the token cap is **not an answer**: it retries once at double the
budget, and a still-truncated draft escalates rather than being scored as a
wrong answer.

**detect.** Gold-free findings, reported as *findings* (the false equation,
the traceback) rather than as a probability:

| detector | what it checks | needs |
|---|---|---|
| `consistency` | the answer contradicts itself (the arithmetic in the derivation does not add up) | nothing |
| `code_exec` | the code does not pass **the task's own tests** | a test block in the task |

That is the whole detector set. `numeric_gold` and `choice_made_gold` exist
for scoring, and the loop never calls them.

**repair.** One re-attempt with the finding quoted, at temperature 0 (the
strongest local re-attempt; a greedy re-ask is not a second sample, it is a
correction). Three rules keep the loop honest:

- a truncated attempt retries once at double the budget and is otherwise
  **never accepted** as the answer;
- a byte-identical attempt **stops** the loop (the model is not going to
  change its mind);
- an attempt that clears the detectors is re-framed to the task's interface
  (`\boxed{…}`, `Final answer: n`, a letter, a fenced block) and becomes the
  served answer - framing only, never content.

**escalate.** Still flagged, or nothing consumable extracted → the human gate
(verbose on the first trigger, compact afterwards) → solve locally again
(`keep` is a sub-option of solve) or escalate to the API tier. There is no
API tier until a provider is named: the escalation lane is a policy, not a
default.

Verdicts: `accepted`, `accepted_repaired`, `escalated`, `error`.

### The gold-free runtime contract

This is the part that makes the rest measurable.

Every task spec carries a `checker`. For a gold-dependent task that checker
has an `expect` field - **the answer key**. The loop is never allowed to see
it. Specs pass through one function:

```python
def runtime_spec(spec: dict) -> dict:
    """Everything the loop is allowed to know: the answer frame and the task's own tests."""
    ck = spec.get("checker") or {}
    rs = {"kind": ck.get("type")}
    for key in ("tests", "prelude", "footer"):
        if key in ck:
            rs[key] = ck[key]
    if any(word in key for key in rs for word in GOLD_WORDS):
        raise AssertionError(f"gold-shaped key reached the loop: {sorted(rs)}")
    return rs
```

`expect` is dropped here, and the function raises if a gold-shaped key ever
reaches the loop. So no stage - detect, repair, canonicalise, gate - can
read the answer key even by accident. Gold-based scoring is a **separate
post-loop step** (`--score`), never consulted by the loop.

The consequence: the loop is the same code on a task whose answer key you
have and on one you do not, and the measured improvement is a property of
the serving loop, not of the benchmark.

## 3. The taxonomy: what a hive is made of

A serving system for local models is a set of *roles* and the *paths* between
them. The cascade ships the taxonomy explicitly (22 roles, A1-H1):

| group | roles | what they do |
|---|---|---|
| **A** - perception | A1 embedding, A2 vision | encode the input |
| **B** - knowledge | B1 retrieval, B2 reranking, B3 late interaction, B4 curation | find and prune what matters |
| **C** - control | C1 routing, C2 gating, C3 next-step, C4 scoring | decide what happens next |
| **D** - verification | D1 consistency, D2 accept/reject | check the answer |
| **E** - generation | E1-E4, by size | write the answer |
| **F/G/H** - escalation, latency, audit | the ladder, the budgets, the trail | keep the record |

and seven **paths** (P0-P6) that compose them:

| path | trigger | shape |
|---|---|---|
| P0 | a near-duplicate prompt | cache lookup only, no generation |
| P1 | easy prompt, high route confidence | small generator, escalate to P3/P5 |
| P2 | normal question | route → gate → retrieve → rerank → generate → verify |
| P3 | low confidence, or the verifier rejects | P2's stages, then a larger generator |
| P4 | code or tool request | the tool loop, verification pipelined |
| P5 | hard reasoning | the largest local model, capped |
| P6 | image input | vision encode → generate → verify |

The point of the taxonomy is not the diagram. It is that **each role has a
measured latency band and a device class**, so a path can be costed before
it is run, and a slow role can be moved off the hot path (the router and the
consistency check are async in P2).

## 4. The measured results

The organic set: 66 items, sampled at T=0.7, live against the local 176B
(A6B ternary, MoE). The bench: GSM8K-style arithmetic, HumanEval-style code,
and a multiple-choice set.

| | items | served-correct | false rejects | note |
|---|---|---|---|---|
| no cascade (sampled) | 66 | 58 | - | the baseline |
| **+ detect + repair** | 66 | **62** | **0** | 6 of the 9 wrong answers detected; 4 of the 5 seen live repaired |
| no cascade (greedy) | 66 | 65 | - | the greedy regime is nearly clean |
| **+ detect + repair (greedy)** | 66 | 65 | 0 | 0 detections - the loop is a no-op |

**Reading the greedy row honestly.** At temperature 0 the bench is nearly
clean, so the loop has nothing to do. Sampling is the cascade's dial: the
cascade earns its keep exactly where the model's own distribution is
uncertain. A deployment that runs greedy gets nothing from this machinery -
that is a result, not a failure, and it is why the sampled regime is the
one reported first.

**The false-reject budget is zero.** 57 correct answers, 0 false rejects.
This is the number that matters most: a verifier that rejects correct
answers is worse than no verifier, because it sends good work to the
expensive lane.

**Repairs do not harm correct answers.** The repair leg is only entered on a
rejection, and a cleared attempt is re-framed, never rewritten.

**Truncation is caught by the ladder.** The runner's own re-runs
(`results/runner-cascade-sampled-20261018/`,
`results/runner-cascade-sampled-nothink-20261018/`) reported 3 truncations
caught and flagged, and 0 false rejects in both.

### 4.1 The tiered refresh: the two Scions, measured side by side

The 2026-10-18 numbers above come from the 176B alone. The question the
tiered design actually poses is *which tier does the work*, so both
Scion models were served at once - on one card, sharing it - and run over
the same 28 organic tasks.

**The models, head to head** (one draw each, T=0.7, 640 tokens, 62 minutes
wall, 0 errors; `experiments/cascade/results/dual-scion-20261020/full28/`):

| | the 35B-A3B | the 176B-A6B |
|---|---|---|
| correct (the task's own checker) | **21/28** | 19/28 |
| the weight | 11.3 GB | ~132 GB |
| latency, median (under co-hosting) | 84.9 s | 93.9 s |
| latency, p90 | 189.5 s | 213.5 s |

They are **equivalent on this class within noise** - the split on the
disagreements is 8 to 6 - and the 176B is twelve times the weight. They
also **disagree on 16 of 28**, so neither dominates: the disagreement is
the raw material for an arbiter, and a router that knows *which* model to
trust per item is worth more than either model alone.

**The cascade over the 35B** (the same 28 tasks, the deterministic judge,
the repair leg on, T=0.7):

| | |
|---|---|
| the model's own accuracy | 0.821 (23/28) |
| **the cascade's served-correct** | **0.857 (24/28)** |
| the repair leg | 1 attempted, **1 cleared** |
| **false rejects** | **0** |
| false accepts | 4 (0.143) |
| the mean per-item latency | **6.9 s** |

The loop does its job on the small tier: one wrong answer caught and
repaired, nothing false-rejected, and the served answer is better than the
model's own. The 176B's arm over the same set is running.

**What this does and does not say.** One draw per task, 28 tasks, one
class (arithmetic, code, multiple choice), and the latencies are measured
under co-hosting - so they are the numbers a tiered deployment sees, not
the isolated ones. The tiering claim that survives this measurement is
narrow and worth stating exactly: **on this class the small tier can take
the work outright**, and the value of the large tier is not accuracy but
the 8 items where it was right and the small one was not - which is
exactly what an arbiter would be for.

## 5. The learned gate: a negative result we kept

The deterministic detectors have a measured blind spot: answers that are
*internally consistent and wrong*. A third trigger - a small learned head
over the draft - was built, fitted, wired and measured
(`harness/cascade/learned_gate.py`).

It works: the intern-decision-0.8B readout on the CPU reproduces the stored
fitted probability exactly from the committed states/heads, and the live
extraction cross-check reproduced the flagship catch at p=0.073751.

And it **stays off by default**, by its own pre-registered rule:

| measurement | result |
|---|---|
| fresh silent-wrong catches | +1 |
| false flags | 3 (budget: ≤2) |
| wall clock | +8.7% |
| verdict | **3 false flags > the pre-registered ≤2 budget → the gate is off** |

Three independent measurements agreed that no 0.8B head separates on the
silent distribution. A gate that fires more often than it should is a tax on
every answer to catch one mistake; at this precision it is not worth it. The
implementation is tested and available as `--gate intern-0.8b`, and the
default is off. **A negative result, pre-registered, is a result.**

## 6. The escalation lane: the harvest

Every run writes `escalation.jsonl` next to its report: one line per item
the ladder resolved - the task, the full draft, the structured detector
findings, the repair attempt, and the ladder's resolution (human answer, API
answer, or a persistent flag).

The serving loop appends and forgets; **nothing in the runtime reads it**.
That is deliberate: the gold-free runtime contract stays intact while the
lane feeds offline training. It is the seed for three consumers:

1. the **gate discriminator** - escalate a task, sample fresh local drafts on
   it, checker-label each against the escalated answer, and re-audit any head
   on the silent distribution before pre-registering a restart;
2. **repair SFT** on `(prompt, draft, finding, fixed answer)` triples;
3. **local-tier distillation** of API and human answers.

The lane is what turns a serving loop into a training-data pipeline without
touching the runtime's contract.

## 7. The reporting

A verdict nobody can trace is a rumour. Every run therefore leaves three
artifacts, and they are published next to this card.

**`report.json`** - the per-item record. For every item: the verdict
(`accepted`, `accepted_repaired`, `escalated`, `error`), the detector
findings (the false equation, the traceback), the repair attempt and its
outcome, the generations it took, the timing, and the ladder's
resolution. The summary block carries the served-correct count, the
repaired set, the blind spot and the detector tallies.

**`manifest.json`** - the provenance. The model's sha256, the stack, the
git commit, the task-set sha and the decode settings, plus the per-task
and per-benchmark aggregates with both the sample-accuracy and the
task-macro. A run can be reproduced because the manifest says what to
reproduce.

**`MODEL-CARD.md`** - generated from the manifest, for the model page.

**`escalation.jsonl`** - the harvest: one line per item the ladder
resolved, carrying the task, the full draft, the structured findings, the
repair attempt and the resolution. The serving loop appends and forgets;
nothing in the runtime reads it, so the gold-free contract is untouched
while the lane feeds offline training.

The point of the separation is that the *runtime* evidence (what the
verifiers caught, what the repair fixed, what the ladder did) and the
*evaluation* evidence (whether the answers were right) never mix. The
first is produced by the serving path with no access to the answer key;
the second is produced afterwards, with the key, and is never consulted
by the loop.

## 8. What is not claimed

- **Not** that the cascade fixes most wrong answers. It caught 6 of 9 wrong
  answers in one regime, and 0 in the greedy regime.
- **Not** that a learned gate is useless. It is off because of *this* head on
  *this* distribution, at a *pre-registered* false-flag budget.
- **Not** that the measurements generalise. 66 organic items, one model, one
  bench. The priority plan is an evaluation floor first - a held-out split
  plus a real benchmark slice (GSM8K / MATH / HumanEval / TruthfulQA, local)
  at 3-5 samples per task, thresholds fitted on train and reported on test.
- **Not** that the taxonomy is complete. 22 roles is a map of what the
  measurements needed so far.

## 9. Reproduce

```bash
git clone https://github.com/sky-is-green/hivebench
cd hivebench && python -m pip install -r requirements-dev.txt

# the deterministic parts, no model needed
pytest tests/unit tests/integration -q

# the live loop against a local model
.venv/bin/python experiments/cascade/run_cascade.py \
    --tasks experiments/cascade/tasks-bench.json \
    --judge deterministic --repair auto \
    --api-fallback http://127.0.0.1:4000/v1     # optional; escalate, not needed to repair

# the console: every knob, one form
open http://127.0.0.1:8799/cascade
```

Runs land under `experiments/cascade/results/<run>/`: `manifest.json`
(provenance - model sha256, stack, git commit, task-set sha, decode
settings - plus per-task and per-benchmark aggregates with sample-accuracy
and task-macro), `streams.json` (the full traces, regenerable), and a
generated `MODEL-CARD.md`. Manifests and cards are tracked in git; the large
stream traces are ignored.

## 10. License

MIT.

```
@misc{hive-cascade-2026,
  title  = {The Hive Cascade: detect, repair, escalate - a gold-free
            verification-and-repair loop for local language models},
  author = {{HiveBench Studio contributors}},
  year   = {2026},
  url    = {https://github.com/sky-is-green/hivebench}
}
```

**Community invitation.** The protocol is written to be challenged: the
verdicts, the budgets and the pre-registration rules are all in the
repository, including the ones that failed. If your hardware produces a
different number, that is a result - open an issue with the run bundle.
