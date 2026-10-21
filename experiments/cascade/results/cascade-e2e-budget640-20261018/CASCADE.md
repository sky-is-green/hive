# The local cascade, end to end (2026-10-18)

`solve -> detect -> repair -> human gate` over 66 items against the live 176B on `http://127.0.0.1:8081`; drafts generated, 1 repair round(s), human gate in `auto` mode.

```sh
.venv/bin/python experiments/cascade/cascade.py \
    --bench <work>/hivebench/experiments/cascade/results/dual-generation-20261018/organic-66.jsonl \
    --out experiments/cascade/results/cascade-e2e-20261018 --score
```

What may run inside the loop is fixed by `bench_registry.json`'s runtime column: for a gold-dependent bench the runtime checks are the declared gold-free fallbacks, `consistency` always and `code_exec` when the task supplies tests.  `runtime_spec()` strips `expect` before a spec enters the loop, so no stage can read the answer key.

## Scoring note

Superseded run, kept as evidence: this pass used a 640-token budget, six math answers were cut off
mid-derivation, and the loop of that day accepted them (the budget ladder -- retry once at 2x, then
escalate -- was added after this run).  Of the seven unrepaired accepts `SCORING.md` calls wrong,
six are those truncated drafts and one is the real error (`gsm8k-0015:hs2`).  Read the final runs'
directories for current numbers.

## Verdicts

| verdict | n |
| --- | --- |
| accepted | 65 |
| escalated | 1 |

Detections fired: `consistency` x1
Requests: 67 (369s wall); truncated at solve (cap, not an answer): 6.

## The human gate

First trigger as the human would have seen it, then the compact form for the rest (auto mode records them instead of asking).

```
=== CASCADE ESCALATION (first trigger, verbose) ===
  item       : gsm8k-0038:hg   (task gsm8k-0038, kind number)
  detections : consistency

  prompt:
    Anthony is sending out coupons for his pizza parlor through the mail. He wants to send out 700 small coupons and twice as many big coupons. If each small coupon costs 5 cents to mail and each big coupon costs 15 cents, how much does he spend on postage total?
    
    Solve the problem, showing your work. End your answer with 'Final answer: <number>'.

  the model's draft:
    To find the total amount Anthony spends on postage, we need to calculate the cost for both the small coupons and the big coupons separately, then add them together.
    
    **Step 1: Calculate the number of big coupons.**
    Anthony wants to send twice as many big coupons as small coupons.
    Number of small coupons = 700
    Number of big coupons = $700 \times 2 = 1400$
    
    **Step 2: Calculate the postage cost for the small coupons.**
    Each small coupon costs 5 cents to mail.
    Total cost for small coupons = $700 \times 5$ cents = $3500$ cents
    
    **Step 3: Calculate the postage cost for the big coupons.**
    Each big coupon costs 15 cents to mail.
    Total cost for big coupons = $1400 \times 15$ cents = $21000$ cents
    
    **Step 4: Calculate the total postage cost.**
    Total postage = Cost for small coupons + Cost for big coupons
    Total postage = $3500$ cents + $21000$ cents = $24500$ cents
    
    To convert cents to dollars, we divide by 100:
    $24500$ cents / $100 = 245$ dollars.
    
    Final answer: 245

  the consistency finding:
    [{'equation': '100 = 245', 'lhs': 100.0, 'rhs': 245.0}]

  the repair attempt:
    To find the total amount Anthony spends on postage, we need to calculate the cost for both the small coupons and the big coupons separately, then add them together.
    
    **Step 1: Calculate the number of big coupons.**
    Anthony wants to send twice as many big coupons as small coupons.
    Number of small coupons = 700
    Number of big coupons = $700 \times 2 = 1400$
    
    **Step 2: Calculate the postage cost for the small coupons.**
    Each small coupon costs 5 cents to mail.
    Total cost for small coupons = $700 \times 5$ cents = $3500$ cents
    
    **Step 3: Calculate the postage cost for the big coupons.**
    Each big coupon costs 15 cents to mail.
    Total cost for big coupons = $1400 \times 15$ cents = $21000$ cents
    
    **Step 4: Calculate the total postage cost.**
    Total postage = Cost for small coupons + Cost for big coupons
    Total postage = $3500$ cents + $21000$ cents = $24500$ cents
    
    To convert cents to dollars, we divide by 100:
    $24500$ cents / $100 = 245$ dollars.
    
    Final answer: 245


```

## Scoring (post-loop, gold-based, eval-only)

| verdict | n | content-correct artifact |
| --- | --- | --- |
| accepted | 65 | 53 |
| escalated | 1 | 0 |

- Accepted artifacts that are wrong (the detectors' blind spot, not the gate's): 12 (gsm8k-0015:hs2, math-0002:hg, math-0002:hs1, math-0002:hs2, math-0006:hg, math-0006:hs1, math-0006:hs2, math-0013:hs1, math-0013:hs2, math-0016:hg, math-0016:hs1, math-0023:hg).
- Repaired and accepted: 0, correct: 0 (-).
- Items the cascade would serve correctly, had everything been as benign as this bench: 53/66.

