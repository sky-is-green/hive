#!/usr/bin/env python3
"""Prototype: a gold-free arithmetic-consistency verifier over the model's own answer.

The two organic items that every detector misses are the same failure.  `gsm8k-0033:hs1` writes
`5 + 10 + 10 + 2.5 + 2.5 = 25` (it is 30, which is the gold); `math-0013:hs1` writes
`2007 = 6 x 334 + 1` (6x334 is 2004, so the remainder is 3).  In both the reasoning chain is right
and one arithmetic step is wrong, which is why every learned readout accepts them with high
confidence -- L0.5 0.525/0.999, MLP 1.000/0.724, self-harvest 0.787/0.774.

That failure is mechanically checkable without the gold answer: find the equations the model asserts
about itself, evaluate both sides, flag the ones that disagree.  This is what the project's own
`consistency` verifier class is defined as ("disagreement / arithmetic mismatch", gold_free_runtime
true) and it is not implemented yet.

The question this answers is precision, not recall: if we flag an item every time its own arithmetic
disagrees with itself, how many of the flagged items are actually wrong?  Measured on the organic-66
(natural errors only, 9 wrong / 57 correct) and, as a wider check, on the 871 harvest train items.

    .venv/bin/python experiments/cascade/consistency_check.py \
        --out experiments/cascade/results/scion176b-harvest-20261018/consistency-check.json
"""
from __future__ import annotations

import argparse
import json
import re
import warnings
from fractions import Fraction
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HARVEST = REPO / "experiments/cascade/results/decision-harvest-r2-20261017/harvest-bench-r2.jsonl"

# Only equations inside math spans are considered: prose contains plenty of "3 x 4" shapes that are
# not assertions the answer is making, and a verifier that flags prose is worse than no verifier.
MATH_SPANS = re.compile(
    r"\$\$(.+?)\$\$"        # display math
    r"|\\\[(.+?)\\\]"      # \[ ... \]
    r"|\\\((.+?)\\\)"      # \( ... \)
    r"|\$([^$\n]+?)\$",    # inline math
    re.S)
# inside a math span: a maximal run of digits/operators, then '=', then another run
ARITH = r"[\d\s\+\-\*/\(\)\.]"
EQ = re.compile(rf"({ARITH}+)=({ARITH}+)")


def _normalise(span: str) -> str:
    e = span.replace("\\times", "*").replace("\\cdot", "*").replace("\\div", "/")
    e = e.replace("\\cdotp", "*").replace("\\!", "").replace("\\,", "").replace("\\ ", " ")
    for sym, op in (("×", "*"), ("÷", "/"), ("−", "-"), ("·", "*")):
        e = e.replace(sym, op)
    e = re.sub(r"(?<=\s)[xX](?=\s)", "*", e)          # a spaced x is multiplication, not a variable
    e = e.replace("\\boxed{", "").rstrip("}").replace("$", "")
    return e.replace(",", "").strip()


def _is_arithmetic(e: str) -> bool:
    return bool(re.fullmatch(r"[\d\s\+\-\*/\(\)\.]+", e)) and any(c.isdigit() for c in e)


def _evaluate(e: str) -> Fraction | None:
    """Evaluate a pure-arithmetic expression exactly, as rationals.  None if it is not safe to run."""
    e = e.strip()
    if not e or len(e) > 120 or not _is_arithmetic(e):
        return None
    if "//" in e or "**" in e:
        return None
    try:
        with warnings.catch_warnings():                 # "5 (2)" compiles with a warning; let eval reject it
            warnings.simplefilter("ignore")
            node = compile(e, "<arith>", "eval")       # noqa: S307 - digits/operators only, guarded
            val = eval(node, {"__builtins__": {}}, {})  # noqa: S307
    except Exception:  # noqa: BLE001 - any failure means "do not trust this line"
        return None
    if isinstance(val, bool) or not isinstance(val, (int, float, Fraction)):
        return None
    try:
        return Fraction(val).limit_denominator(10 ** 9)
    except Exception:  # noqa: BLE001
        return None


def check_answer(answer: str) -> dict:
    """Return the self-inconsistent equations the answer asserts (gold-free)."""
    bad = []
    checked = 0
    for m in MATH_SPANS.finditer(answer):
        raw = m.group(0)
        # a span mixing prose into the math ("...\text{ with a remainder of }...") states a
        # convention rather than an equality we can evaluate
        if "\\text{" in raw or "remainder" in raw.lower():
            continue
        span = _normalise(raw)
        if "=" not in span:
            continue
        for eq in EQ.finditer(span):
            s, e = eq.span()
            lhs, rhs = eq.group(1).strip(), eq.group(2).strip()
            if not rhs or "=" in rhs:
                continue
            # "2.5x = 15" is variable notation, not an arithmetic claim: if either side runs into a
            # letter or digit we did not capture, this is not an equation we know how to check
            if s > 0 and (span[s - 1].isalnum()):
                continue
            if e < len(span) and (span[e].isalnum()):
                continue
            lv, rv = _evaluate(lhs), _evaluate(rhs)
            if lv is None or rv is None:
                continue
            checked += 1
            if lv != rv:
                bad.append({"equation": f"{lhs} = {rhs}", "lhs": float(lv), "rhs": float(rv)})
    return {"equations_checked": checked, "inconsistent": bad, "flagged": bool(bad)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in HARVEST.read_text().splitlines() if l.strip()]
    slices = {"organic-66": [r for r in rows if r["split"] == "test"],
              "harvest-train-871": [r for r in rows if r["split"] == "train"],
              "harvest-937": rows}
    out: dict = {"verifier": "self-asserted arithmetic equations: evaluate both sides, flag disagreement",
                 "gold_free": True, "slices": {}}
    for sname, items in slices.items():
        flagged = [r for r in items if check_answer(r["candidate"])["flagged"]]
        wrong = [r for r in items if r["label"] == 0]
        corr = [r for r in items if r["label"] == 1]
        caught = [r for r in flagged if r["label"] == 0]
        fr = [r for r in flagged if r["label"] == 1]
        out["slices"][sname] = {
            "n": len(items), "wrong": len(wrong), "correct": len(corr),
            "flagged": len(flagged), "flagged_and_wrong": len(caught),
            "precision_on_flagged": (len(caught) / len(flagged)) if flagged else None,
            "recall_of_wrongs": len(caught) / len(wrong) if wrong else None,
            "false_flag_rate_on_correct": len(fr) / len(corr) if corr else None,
            "equations_checked_total": sum(check_answer(r["candidate"])["equations_checked"]
                                           for r in items),
            "flagged_items": [{"item_id": r["item_id"], "label": r["label"], "origin": r["origin"],
                               "inconsistent": check_answer(r["candidate"])["inconsistent"]}
                              for r in flagged],
        }
        s = out["slices"][sname]
        print(f"{sname:18s} n={s['n']:4d} wrong={s['wrong']:3d} | equations checked "
              f"{s['equations_checked_total']:5d} | flagged {s['flagged']:3d} of which wrong "
              f"{s['flagged_and_wrong']:3d} | precision {s['precision_on_flagged'] if flagged else 0:.3f} "
              f"| recall of wrongs {s['recall_of_wrongs']:.3f} | false-flag rate on corrects "
              f"{s['false_flag_rate_on_correct']:.4f}")

    org = out["slices"]["organic-66"]
    print("\norganic-66 flagged items:")
    for f in org["flagged_items"]:
        print(f"  {f['item_id']:26s} label={f['label']} {f['origin']}")
        for eq in f["inconsistent"]:
            print(f"      {eq['equation']}   (lhs={eq['lhs']} rhs={eq['rhs']})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())