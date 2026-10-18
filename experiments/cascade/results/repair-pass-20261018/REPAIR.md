# Repair pass on the 176B's caught wrongs (2026-10-18)

Same serve, same prompt shape as the harvest; the live 176B on `http://127.0.0.1:8081` answered a second time, shown its own draft and (arm A) the finding one of the gold-free detectors produced.  Gold is used only to score the outcome.

```sh
.venv/bin/python experiments/cascade/repair_pass.py \
    --bench /run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work/hivebench/experiments/cascade/results/dual-generation-20261018/organic-66.jsonl \
    --out experiments/cascade/results/repair-pass-20261018
```

Scoring note: the tables below use the task's *registered* checker (strict).  `SCORING.md` adds the
content-tolerant scoring, and footnote 1 reflects it; a re-run of `repair_pass.py` would regenerate
this file, so the footnotes live here by hand.

## The detected set (gold-free, recomputed here)

| detector | items |
| --- | --- |
| `consistency` | 2: gsm8k-0033:hs1, math-0013:hs1 |
| `code_exec` | 4: humaneval-0001:hs1, humaneval-0001:hs2, humaneval-0015:hs1, humaneval-0015:hs2 |
| **union** | **6** of the 9 organic wrongs |

False rejects on the 57 correct items: 0.

## Arms

| arm | items | corrected by the repair | still wrong | broken (was right) | hit token cap |
| --- | --- | --- | --- | --- | --- |
| A evidence repair | 6 | 5/6 | 1 | 0 | 0 |
| B generic recheck | 6 | 2/6 | 4 | 0 | 0 |
| C harm control | 8 | 0 | — | 0 (2 under the strict checker, both format¹) | 0 |

¹ The two strict-only "harms" are `math-0002:hg` and `math-0002:hs1`: the repaired answer states
the same, correct value, but as "Final answer: 3600" instead of `\boxed{3600}`, and the registered
checker for `boxed` tasks reads only the box.  Content is stable, framing is not.  `SCORING.md`
carries both full scorings; the numbers in this file are the strict, registered-checker ones.

## Per item (A evidence / B generic / C control, where the item appears)

| item | finding | A | B | C |
| --- | --- | --- | --- | --- |
| `gsm8k-0005:hg` | - | · | · | yes |
| `gsm8k-0005:hs1` | - | · | · | yes |
| `gsm8k-0033:hs1` | consistency | yes | no | · |
| `humaneval-0001:hg` | - | · | · | yes |
| `humaneval-0001:hs1` | code_exec | yes | yes | · |
| `humaneval-0001:hs2` | code_exec | yes | yes | · |
| `humaneval-0002:hg` | - | · | · | yes |
| `humaneval-0015:hs1` | code_exec | yes | no | · |
| `humaneval-0015:hs2` | code_exec | no | no | · |
| `math-0002:hg` | - | · | · | yes¹ |
| `math-0002:hs1` | - | · | · | yes¹ |
| `math-0013:hs1` | consistency | yes | no | · |
| `truthfulqa-0008:hg` | - | · | · | yes |
| `truthfulqa-0011:hg` | - | · | · | yes |

## Reading

- **The finding is not decoration, but it is not doing all the work either.**  Arm A repairs 5 of 6,
  arm B 2 of 6, so the specific finding buys **3 net repairs**.  On the other 3 items A and B
  produced *byte-identical* text at temperature 0 -- the model discarded the evidence and simply
  re-derived the problem; when it engages with the finding it wins 3 of 3, when it ignores it,
  nothing changes.  Re-asking alone is worth 2.
- **The residual is one item**: `humaneval-0015:hs2`, where the model re-emitted its wrong code
  verbatim, twice (A and B identical, 218 tokens each).  That is exactly the case the escalation
  tier exists for, and it is still blocked on a provider.
- **Re-asking does not break content.**  0 of 8 control answers became wrong.  It does change
  *framing*: 2 of 8 correct answers came back as "Final answer: 3600" instead of `\boxed{3600}`,
  which the strict registered checker scores as failed.  A consumer that insists on the box would
  call that damage; the answer itself is right.  Repair therefore needs a canonical re-framing step
  (or a tolerant consumer) before it can be stacked on the cascade.
- Cost of the whole pass: 20 requests, 191 s of serve wall, ~4.0k completion tokens, no window and
  no swap -- the repair leg is ~88 s of extra serve time for the 6 detected items (arm A alone).
- 0 cap hits; every answer finished with `stop`.
