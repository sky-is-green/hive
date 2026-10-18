# Repair pass: two scorings (2026-10-18)

`strict` is the task's registered checker (what `label` and `gold_ok_after` use): for `boxed` tasks it requires `\boxed{...}`.  `content` extracts the final answer (box, else 'Final answer: n', else letter) and compares it with the expected value -- the same extraction the harvest's `final` column used.  Both are gold-based and eval-only.

| arm | n | strict repaired | content repaired | strict harm | content harm |
| --- | --- | --- | --- | --- | --- |
| A evidence repair | 6 | 5/6 | 5/6 | 0/6 | 0/6 |
| B generic recheck | 6 | 2/6 | 2/6 | 0/6 | 0/6 |
| C harm control | 8 | 6/8 | 8/8 | 2/8 | 0/8 |

## Where the two scorings disagree

| item | arm | strict | content | expected | new final |
| --- | --- | --- | --- | --- | --- |
| `math-0002:hg` | C | fail | pass | 3600 | 3600 |
| `math-0002:hs1` | C | fail | pass | 3600 | 3600 |

## Per item

| item | arm | kind | finding | strict | content | flags after | tok | s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `gsm8k-0005:hg` | C | number | - | pass | pass | no | 219 | 8.9 |
| `gsm8k-0005:hs1` | C | number | - | pass | pass | no | 84 | 5.9 |
| `gsm8k-0033:hs1` | A | number | consistency | pass | pass | no | 404 | 38.1 |
| `gsm8k-0033:hs1` | B | number | consistency | fail | fail | no | 185 | 9.6 |
| `humaneval-0001:hg` | C | code | - | pass | pass | no | 142 | 6.0 |
| `humaneval-0001:hs1` | A | code | code_exec | pass | pass | no | 157 | 16.5 |
| `humaneval-0001:hs1` | B | code | code_exec | pass | pass | no | 157 | 6.4 |
| `humaneval-0001:hs2` | A | code | code_exec | pass | pass | no | 145 | 6.1 |
| `humaneval-0001:hs2` | B | code | code_exec | pass | pass | no | 145 | 5.1 |
| `humaneval-0002:hg` | C | code | - | pass | pass | no | 194 | 6.7 |
| `humaneval-0015:hs1` | A | code | code_exec | pass | pass | no | 203 | 9.0 |
| `humaneval-0015:hs1` | B | code | code_exec | fail | fail | no | 171 | 5.9 |
| `humaneval-0015:hs2` | A | code | code_exec | fail | fail | no | 218 | 7.0 |
| `humaneval-0015:hs2` | B | code | code_exec | fail | fail | no | 218 | 7.0 |
| `math-0002:hg` | C | boxed | - | fail | pass | no | 275 | 11.4 |
| `math-0002:hs1` | C | boxed | - | fail | pass | no | 217 | 8.2 |
| `math-0013:hs1` | A | boxed | consistency | pass | pass | no | 411 | 11.4 |
| `math-0013:hs1` | B | boxed | consistency | fail | fail | yes | 193 | 6.8 |
| `truthfulqa-0008:hg` | C | choice | - | pass | pass | no | 170 | 10.1 |
| `truthfulqa-0011:hg` | C | choice | - | pass | pass | no | 82 | 5.4 |
