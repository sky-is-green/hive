# Labeled silent-corpus audit

Corpus: 20 drafts (12 correct, 8 wrong) from sample_check.py (mined-hard tasks, k fresh drafts each, checker-labeled).

## Per-head separation

| head | wrongs p min-max | corrects p 10th pct (n) | caught@0.5 | flags@0.5 | cap-2 best catch/flags |
|---|---|---|---:|---:|---:|
| L0.5 | 0.0000-0.9945 | 0.0231 (n=12) | 5 | 5 | 3/1 |
| L0.75 | 0.0342-0.9956 | 0.6532 (n=12) | 5 | 1 | 5/1 |
| L1 | 0.0000-0.9998 | 0.2250 (n=12) | 3 | 4 | 3/2 |
| ridge-concat | 0.0003-0.9980 | 0.2077 (n=12) | 5 | 4 | 4/1 |
| mlp-concat | 0.0000-0.9999 | 0.0623 (n=12) | 5 | 3 | 5/2 |

## Verdict

**No head separates.** Wrongs' p ranges overlap the corrects' mass on the silent distribution — the gate's problem is features/labels, not threshold. Do not restart M5 or retrain yet; grow the corpus.

## Per-item p (sorted by L0.75)

| task | truth | L0.5 | L0.75 | L1 | ridge-concat | mlp-concat |
|---|---|---:|---:|---:|---:|---:|
| math-0020:hg | W | 0.0465 | 0.0342 | 0.7551 | 0.0061 | 0.6385 |
| humaneval-0015:hg | W | 0.0000 | 0.0765 | 0.0000 | 0.0083 | 0.0000 |
| math-0020:hg | C | 0.0023 | 0.2207 | 0.0532 | 0.0034 | 0.0000 |
| humaneval-0007:hs2 | W | 0.9358 | 0.2575 | 0.6181 | 0.4067 | 0.0605 |
| humaneval-0015:hs1 | W | 0.0019 | 0.3398 | 0.0560 | 0.0878 | 0.0036 |
| math-0020:hg | W | 0.0034 | 0.4345 | 0.3589 | 0.0003 | 0.0011 |
| gsm8k-0035:hs1 | C | 0.0231 | 0.6532 | 0.2250 | 0.2077 | 0.9048 |
| humaneval-0015:hs1 | C | 0.0282 | 0.7537 | 0.3659 | 0.5992 | 0.2920 |
| humaneval-0015:hg | C | 0.5013 | 0.7805 | 0.3705 | 0.9705 | 0.0623 |
| humaneval-0007:hs2 | C | 0.9995 | 0.9014 | 0.9864 | 0.9558 | 0.8003 |
| humaneval-0007:hs2 | C | 0.9998 | 0.9258 | 0.9920 | 0.9772 | 0.8621 |
| humaneval-0007:hs2 | C | 0.9991 | 0.9441 | 0.9798 | 0.9614 | 0.8109 |
| humaneval-0007:hs2 | C | 0.9991 | 0.9441 | 0.9798 | 0.9614 | 0.8109 |
| humaneval-0015:hs1 | W | 0.4157 | 0.9658 | 0.9248 | 0.9949 | 0.2078 |
| gsm8k-0035:hs1 | C | 0.9891 | 0.9711 | 0.9058 | 0.9371 | 0.9964 |
| gsm8k-0035:hs1 | C | 0.9989 | 0.9797 | 0.9837 | 0.9997 | 0.9999 |
| gsm8k-0038:hs1 | C | 0.0672 | 0.9809 | 0.9533 | 0.4798 | 0.9948 |
| gsm8k-0038:hs1 | C | 0.2076 | 0.9877 | 0.9870 | 0.2221 | 0.9963 |
| gsm8k-0033:hs1 | W | 0.9662 | 0.9955 | 0.9998 | 0.9925 | 0.9972 |
| gsm8k-0033:hs1 | W | 0.9945 | 0.9956 | 0.9998 | 0.9980 | 0.9999 |
