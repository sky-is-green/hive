# Cascade scoring (gold-based, eval-only) -- cascade-e2e-budget640-20261018

Re-derived by `cascade.py --rescore`; the loop itself never reads gold.  For repaired code rows the raw fenced repair text is scored.

**Superseded run:** six of the seven "wrong" unrepaired accepts below are drafts the 640-token
budget cut off mid-derivation -- the code now retries at 2x and escalates instead of accepting them.
The one real error in this file is `gsm8k-0015:hs2`.

| verdict | n | content-correct artifact |
| --- | --- | --- |
| accepted | 65 | 58 |
| escalated | 1 | 0 |

- Unrepaired accepts that are wrong (the detectors' blind spot, not the gate's): 7 (gsm8k-0015:hs2, math-0002:hg, math-0002:hs1, math-0002:hs2, math-0013:hs1, math-0013:hs2, math-0023:hg).
- Repaired and accepted: 0, correct: 0 (-).
- Served-correct artifacts: 58/66.

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
| `gsm8k-0015:hs2` | accepted | fail |
| `gsm8k-0020:hg` | accepted | pass |
| `gsm8k-0020:hs1` | accepted | pass |
| `gsm8k-0020:hs2` | accepted | pass |
| `gsm8k-0033:hg` | accepted | pass |
| `gsm8k-0033:hs1` | accepted | pass |
| `gsm8k-0035:hg` | accepted | pass |
| `gsm8k-0035:hs1` | accepted | pass |
| `gsm8k-0035:hs2` | accepted | pass |
| `gsm8k-0036:hg` | accepted | pass |
| `gsm8k-0036:hs1` | accepted | pass |
| `gsm8k-0036:hs2` | accepted | pass |
| `gsm8k-0037:hg` | accepted | pass |
| `gsm8k-0037:hs1` | accepted | pass |
| `gsm8k-0037:hs2` | accepted | pass |
| `gsm8k-0038:hg` | escalated | n/a |
| `gsm8k-0038:hs1` | accepted | pass |
| `gsm8k-0038:hs2` | accepted | pass |
| `humaneval-0001:hg` | accepted | pass |
| `humaneval-0001:hs1` | accepted | pass |
| `humaneval-0001:hs2` | accepted | pass |
| `humaneval-0002:hg` | accepted | pass |
| `humaneval-0002:hs1` | accepted | pass |
| `humaneval-0003:hg` | accepted | pass |
| `humaneval-0003:hs1` | accepted | pass |
| `humaneval-0004:hg` | accepted | pass |
| `humaneval-0004:hs1` | accepted | pass |
| `humaneval-0004:hs2` | accepted | pass |
| `humaneval-0007:hg` | accepted | pass |
| `humaneval-0007:hs1` | accepted | pass |
| `humaneval-0007:hs2` | accepted | pass |
| `humaneval-0015:hg` | accepted | pass |
| `humaneval-0015:hs1` | accepted | pass |
| `humaneval-0015:hs2` | accepted | pass |
| `math-0002:hg` | accepted | fail |
| `math-0002:hs1` | accepted | fail |
| `math-0002:hs2` | accepted | fail |
| `math-0006:hg` | accepted | pass |
| `math-0006:hs1` | accepted | pass |
| `math-0006:hs2` | accepted | pass |
| `math-0013:hs1` | accepted | fail |
| `math-0013:hs2` | accepted | fail |
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
