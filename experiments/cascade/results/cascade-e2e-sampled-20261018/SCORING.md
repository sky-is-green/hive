# Cascade scoring (gold-based, eval-only) -- cascade-e2e-sampled-20261018

Re-derived by `cascade.py --rescore`; the loop itself never reads gold.  For repaired code rows the raw fenced repair text is scored.

| verdict | n | content-correct artifact |
| --- | --- | --- |
| accepted | 59 | 58 |
| accepted_repaired | 5 | 4 |
| escalated | 2 | 0 |

- Unrepaired accepts that are wrong (the detectors' blind spot, not the gate's): 1 (math-0023:hg).
- Repaired and accepted: 5, correct: 4 (gsm8k-0033:hs1, humaneval-0003:hg, humaneval-0003:hs1, humaneval-0007:hs2).
- Served-correct artifacts: 62/66.

| item | verdict | content |
| --- | --- | --- |
| `gsm8k-0005:hg` | accepted | pass |
| `gsm8k-0005:hs1` | accepted | pass |
| `gsm8k-0006:hg` | accepted | pass |
| `gsm8k-0006:hs1` | accepted | pass |
| `gsm8k-0006:hs2` | accepted | pass |
| `gsm8k-0008:hg` | accepted | pass |
| `gsm8k-0008:hs1` | accepted | pass |
| `gsm8k-0008:hs2` | accepted | pass |
| `gsm8k-0009:hg` | accepted | pass |
| `gsm8k-0009:hs1` | accepted | pass |
| `gsm8k-0009:hs2` | accepted | pass |
| `gsm8k-0014:hg` | accepted | pass |
| `gsm8k-0014:hs1` | accepted | pass |
| `gsm8k-0014:hs2` | accepted | pass |
| `gsm8k-0015:hs2` | accepted_repaired | fail |
| `gsm8k-0020:hg` | accepted | pass |
| `gsm8k-0020:hs1` | accepted | pass |
| `gsm8k-0020:hs2` | accepted | pass |
| `gsm8k-0033:hg` | accepted | pass |
| `gsm8k-0033:hs1` | accepted_repaired | pass |
| `gsm8k-0035:hg` | accepted | pass |
| `gsm8k-0035:hs1` | accepted | pass |
| `gsm8k-0035:hs2` | accepted | pass |
| `gsm8k-0036:hg` | accepted | pass |
| `gsm8k-0036:hs1` | accepted | pass |
| `gsm8k-0036:hs2` | accepted | pass |
| `gsm8k-0037:hg` | accepted | pass |
| `gsm8k-0037:hs1` | accepted | pass |
| `gsm8k-0037:hs2` | accepted | pass |
| `gsm8k-0038:hg` | accepted | pass |
| `gsm8k-0038:hs1` | accepted | pass |
| `gsm8k-0038:hs2` | accepted | pass |
| `humaneval-0001:hg` | accepted | pass |
| `humaneval-0001:hs1` | accepted | pass |
| `humaneval-0001:hs2` | accepted | pass |
| `humaneval-0002:hg` | accepted | pass |
| `humaneval-0002:hs1` | accepted | pass |
| `humaneval-0003:hg` | accepted_repaired | pass |
| `humaneval-0003:hs1` | accepted_repaired | pass |
| `humaneval-0004:hg` | accepted | pass |
| `humaneval-0004:hs1` | accepted | pass |
| `humaneval-0004:hs2` | accepted | pass |
| `humaneval-0007:hg` | accepted | pass |
| `humaneval-0007:hs1` | accepted | pass |
| `humaneval-0007:hs2` | accepted_repaired | pass |
| `humaneval-0015:hg` | accepted | pass |
| `humaneval-0015:hs1` | escalated | fail |
| `humaneval-0015:hs2` | escalated | fail |
| `math-0002:hg` | accepted | pass |
| `math-0002:hs1` | accepted | pass |
| `math-0002:hs2` | accepted | pass |
| `math-0006:hg` | accepted | pass |
| `math-0006:hs1` | accepted | pass |
| `math-0006:hs2` | accepted | pass |
| `math-0013:hs1` | accepted | pass |
| `math-0013:hs2` | accepted | pass |
| `math-0016:hg` | accepted | pass |
| `math-0016:hs1` | accepted | pass |
| `math-0020:hg` | accepted | pass |
| `math-0020:hs2` | accepted | pass |
| `math-0023:hg` | accepted | fail |
| `truthfulqa-0008:hg` | accepted | pass |
| `truthfulqa-0011:hg` | accepted | pass |
| `truthfulqa-0013:hg` | accepted | pass |
| `truthfulqa-0015:hg` | accepted | pass |
| `truthfulqa-0015:hs1` | accepted | pass |
