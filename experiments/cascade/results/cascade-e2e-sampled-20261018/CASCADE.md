# The local cascade, end to end (2026-10-18)

`solve -> detect -> repair -> human gate` over 66 items against the live 176B on `http://127.0.0.1:8081`; drafts generated, 1 repair round(s), human gate in `auto` mode.

```sh
.venv/bin/python experiments/cascade/cascade.py \
    --bench /run/media/penis/cbfa1404-241a-484e-9136-de07ee2fa8ad/work/hivebench/experiments/cascade/results/dual-generation-20261018/organic-66.jsonl \
    --out experiments/cascade/results/cascade-e2e-sampled-20261018 --temperature 0.7 --score
```

What may run inside the loop is fixed by `bench_registry.json`'s runtime column: for a gold-dependent bench the runtime checks are the declared gold-free fallbacks, `consistency` always and `code_exec` when the task supplies tests.  `runtime_spec()` strips `expect` before a spec enters the loop, so no stage can read the answer key.

## Scoring note

This run sampled at `--temperature 0.7`, the harvest's own regime -- the stored `hs` candidates were
T=0.7 samples, which is why the nine organic wrongs do not reproduce under greedy decoding.  The
scoring section below was written at run time by the first version of the scorer, which mis-read
fenced code and nested braces; **`SCORING.md` in this directory is the authoritative re-score**
(`cascade.py --rescore`), and it reads: 58/59 unrepaired accepts correct, 4/5 repairs correct, so
62/66 served-correct against 58/66 for the raw first attempts.

## Verdicts

| verdict | n |
| --- | --- |
| accepted | 59 |
| accepted_repaired | 5 |
| escalated | 2 |

Detections fired: `code_exec` x5, `consistency` x2
Requests: 76 (496s wall); truncated at solve, then escalated not answered: 0.

## The human gate

First trigger as the human would have seen it, then the compact form for the rest (auto mode records them instead of asking).

```
=== CASCADE ESCALATION (first trigger, verbose) ===
  item       : humaneval-0015:hs1   (task humaneval-0015, kind code)
  detections : code_exec

  prompt:
    Complete the following Python function. Return only the full implementation in a single ```python code block.
    
    ```python
    
    
    def is_palindrome(string: str) -> bool:
        """ Test if given string is a palindrome """
        return string == string[::-1]
    
    
    def make_palindrome(string: str) -> str:
        """ Find the shortest palindrome that begins with a supplied string.
        Algorithm idea is simple:
        - Find the longest postfix of supplied string that is a palindrome.
        - Append to the end of the string reverse of a string prefix that comes before the palindromic suffix.
        >>> make_palindrome('')
        ''
        >>> make_palindrome('cat')
        'catac'
        >>> make_palindrome('cata')
        'catac'
        """
    ```

  the model's draft:
    ```python
    def is_palindrome(string: str) -> bool:
        """ Test if given string is a palindrome """
        return string == string[::-1]
    
    def make_palindrome(string: str) -> str:
        """ Find the shortest palindrome that begins with a supplied string.
        Algorithm idea is simple:
        - Find the longest postfix of supplied string that is a palindrome.
        - Append to the end of the string reverse of a string prefix that comes before the palindromic suffix.
        >>> make_palindrome('')
        ''
        >>> make_palindrome('cat')
        'catac'
        >>> make_palindrome('cata')
        'catac'
        """
        if not string:
            return ''
    
        # Find the longest palindromic suffix
        suffix_len = 0
        for i in range(len(string), 0, -1):
            if is_palindrome(string[i:]):
                suffix_len = len(string[i:])
                break
    
        # The prefix before the palindromic suffix needs to be mirrored
        prefix = string[:len(string) - suffix_len]
        return string + prefix[::-1]
    ```
    
    Final answer: 3

  the code_exec finding:
    Traceback (most recent call last):
      File "/tmp/tmpksukipmh.py", line 55, in <module>
        check(make_palindrome)
      File "/tmp/tmpksukipmh.py", line 50, in check
        assert candidate('x') == 'x'
               ^^^^^^^^^^^^^^^^^^^^^
    AssertionError

  the repair attempt:
    ```python
    def is_palindrome(string: str) -> bool:
        """ Test if given string is a palindrome """
        return string == string[::-1]
    
    def make_palindrome(string: str) -> str:
        """ Find the shortest palindrome that begins with a supplied string.
        Algorithm idea is simple:
        - Find the longest postfix of supplied string that is a palindrome.
        - Append to the end of the string reverse of a string prefix that comes before the palindromic suffix.
        >>> make_palindrome('')
        ''
        >>> make_palindrome('cat')
        'catac'
        >>> make_palindrome('cata')
        'catac'
        """
        if not string:
            return ''
    
        # Find the longest palindromic suffix
        suffix_len = 0
        for i in range(len(string), 0, -1):
            if is_palindrome(string[i:]):
                suffix_len = len(string[i:])
                break
    
        # The prefix before the palindromic suffix needs to be mirrored
        prefix = string[:len(string) - suffix_len]
        return string + prefix[::-1]
    ```
    
    Final answer: 3


=== CASCADE ESCALATION #2 (compact) ===
  humaneval-0015:hs2  kind=code  still flagged after repair
  finding: code_exec: Traceback (most recent call last):
  File "/tmp/tmp34zkjhh7.py", line 52, in <module>
    check(make_palindrome)
  File "/tmp/tmp34zkjhh7.py", line 47, in check
  draft tail:   t palindromic suffix
    for i in range(len(string), 0, -1):
        suffix = string[i:]
        if is_palindrome(suffix):
            prefix = string[:i]
            return string + prefix[::-1]

```
  attempt tail: t palindromic suffix
    for i in range(len(string), 0, -1):
        suffix = string[i:]
        if is_palindrome(suffix):
            prefix = string[:i]
            return string + prefix[::-1]

```

```

## Scoring (post-loop, gold-based, eval-only)

| verdict | n | content-correct artifact |
| --- | --- | --- |
| accepted | 59 | 58 |
| accepted_repaired | 5 | 1 |
| escalated | 2 | 0 |

- Accepted artifacts that are wrong (the detectors' blind spot, not the gate's): 5 (gsm8k-0015:hs2, humaneval-0003:hg, humaneval-0003:hs1, humaneval-0007:hs2, math-0023:hg).
- Repaired and accepted: 5, correct: 1 (gsm8k-0033:hs1).
- Items the cascade would serve correctly, had everything been as benign as this bench: 59/66.

