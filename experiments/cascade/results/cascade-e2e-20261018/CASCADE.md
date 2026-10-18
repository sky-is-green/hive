# The local cascade, end to end (2026-10-18)

`solve -> detect -> repair -> human gate` over 66 items against the live 176B on `http://127.0.0.1:8081`; drafts generated, 1 repair round(s), human gate in `auto` mode.

```sh
.venv/bin/python experiments/cascade/cascade.py \
    --bench /run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work/hivebench/experiments/cascade/results/dual-generation-20261018/organic-66.jsonl \
    --out experiments/cascade/results/cascade-e2e-20261018 --score
```

What may run inside the loop is fixed by `bench_registry.json`'s runtime column: for a gold-dependent bench the runtime checks are the declared gold-free fallbacks, `consistency` always and `code_exec` when the task supplies tests.  `runtime_spec()` strips `expect` before a spec enters the loop, so no stage can read the answer key.

## Scoring note

Greedy (`--temperature 0`): the regime the product serves when it wants the model's best single
answer.  The detectors fired on nothing, so this run's cascade was a no-op and the number to read
is the raw accuracy: 65/66, with `gsm8k-0015:hs2` wrong under both regimes (41.43 greedy, 24.3 as
a T=0.7 sample) and invisible to arithmetic checking -- a systematic error, which is the escalation
tier's business, not the repair leg's.  `SCORING.md` is the authoritative re-score; the section
below was written at run time by the first scorer.

## Verdicts

| verdict | n |
| --- | --- |
| accepted | 66 |

Detections fired: none
Requests: 66 (343s wall); truncated at solve, then escalated not answered: 0.

## Scoring (post-loop, gold-based, eval-only)

| verdict | n | content-correct artifact |
| --- | --- | --- |
| accepted | 66 | 65 |

- Accepted artifacts that are wrong (the detectors' blind spot, not the gate's): 1 (gsm8k-0015:hs2).
- Repaired and accepted: 0, correct: 0 (-).
- Items the cascade would serve correctly, had everything been as benign as this bench: 65/66.

