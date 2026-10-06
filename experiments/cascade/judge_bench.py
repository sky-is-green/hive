#!/usr/bin/env python
"""Build the frozen balanced judge benchmark (J-Bench) for the cascade verifier.

J-Bench is built once from cascade-bench-v1 and then frozen (sha256 in the
manifest).  It answers one question: given a question and a candidate answer,
is the candidate correct?

Positives: the recorded generator answers from the benchmark runs (checker
says correct).  Negatives: checker-verified perturbations of those answers
(number flips, boxed edits, entity swaps, choice flips, code edits) plus any
naturally wrong recorded answers.  The task split (train/test) is preserved;
the test split is the held-out measurement set and is never used for fitting.

Usage:
    python experiments/cascade/judge_bench.py \
        --out experiments/cascade/results/jev-readout-20261006
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.cascade.checkers import check  # noqa: E402

CASCADE = REPO / "experiments" / "cascade"
REPORTS = [
    "results/strata-bench-train-20261003/report.json",
    "results/strata-bench-test-labels-20261003/report.json",
]

DECOYS = ["unknown", "not applicable", "the other option", "unspecified"]


# ---------------------------------------------------------------- perturbations
def _fmt_num(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return f"{value:g}"


def _replace_last_number(text: str, value: float) -> Optional[str]:
    matches = list(re.finditer(r"-?\d+(?:\.\d+)?", text.replace(",", "")))
    if not matches:
        return None
    # prefer the number attached to the checker's "Final answer" marker
    marker = list(re.finditer(
        r"final answer[^0-9\-+]*([-+]?\d[\d,]*(?:\.\d+)?)", text, re.I | re.S))
    token = matches[-1]
    if marker:
        m = marker[-1]
        return text[:m.start(1)] + _fmt_num(value) + text[m.end(1):]
    return text[:token.start()] + _fmt_num(value) + text[token.end():]


def _replace_boxed(answer: str, value: str) -> Optional[str]:
    idx = answer.rfind("\\boxed{")
    if idx < 0:
        return None
    depth, start = 0, idx + len("\\boxed{")
    for i in range(start, len(answer)):
        if answer[i] == "{":
            depth += 1
        elif answer[i] == "}":
            if depth == 0:
                return answer[:start] + value + answer[i:]
            depth -= 1
    return None


def _code_candidates(answer: str) -> list[str]:
    from experiments.cascade.checkers import _code_block

    code = _code_block(answer)
    if code is None:
        return []
    out = []
    # flip the last numeric literal
    nums = list(re.finditer(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w.])", code))
    if nums:
        last = nums[-1]
        v = float(last.group(1))
        for nv in (v * 2 + 1, v + 7, v - 1):
            if abs(nv - v) > 1e-9:
                out.append(code[:last.start()] + _fmt_num(nv) + code[last.end():])
    # force a wrong return at the end of the first function
    out.append(re.sub(r"(\n\s*return\b[^\n]*)", "\n    return None", code, count=1))
    out.append(code + "\n    raise AssertionError()")
    # rebuild the answer with the mutated block
    from experiments.cascade.checkers import _code_block as cb
    m = list(re.finditer(r"```(?:python|py)?\s*\n(.*?)```", answer, re.S))
    if not m:
        return []
    blk = m[-1]
    return [answer[:blk.start(1)] + c + answer[blk.end(1):] for c in out]


def _runs_clean_wrong(task: dict, answer: str) -> bool:
    """A code negative is accepted when the checker rejects it for a semantic
    reason: either the program runs to completion but the tests do not pass, or
    it raises at runtime (assertion/exception).  Syntax errors are rejected as
    too artificial."""
    from experiments.cascade.checkers import _code_block

    code = _code_block(answer)
    if code is None:
        return False
    spec = task["checker"]
    tests = spec["tests"]
    if isinstance(tests, list):
        tests = "\n".join(tests)
    program = (str(spec.get("prelude", "")) + code + "\n\n" + tests + "\n"
               + str(spec.get("footer", "print('PASS')")) + "\n")
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     dir="/tmp/opencode") as fh:
        fh.write(program)
        path = fh.name
    try:
        proc = subprocess.run([sys.executable, path], capture_output=True,
                              text=True, timeout=15)
    except subprocess.TimeoutExpired:
        return False
    finally:
        try:
            Path(path).unlink()
        except OSError:
            pass
    stderr = proc.stderr or ""
    if "SyntaxError" in stderr or "IndentationError" in stderr:
        return False
    if proc.returncode == 0 and "PASS" in proc.stdout:
        return False
    return True


def perturbations(task: dict, answer: str, decoys: list[str]) -> list[tuple[str, str]]:
    spec = task["checker"]
    kind = spec["type"]
    out: list[tuple[str, str]] = []
    if kind == "number":
        expect = float(spec["expect"])
        for nv in (expect * 2, expect + 7, expect - 1, -expect):
            if abs(nv - expect) <= float(spec.get("tol", 1e-6)):
                continue
            c = _replace_last_number(answer, nv)
            if c and c != answer:
                out.append(("num-flip", c))
    elif kind == "boxed":
        expect = str(spec["expect"])
        variants: list[str] = []
        try:
            variants.append(_fmt_num(float(expect) + 1))
            variants.append(_fmt_num(float(expect) * 2 + 1))
        except ValueError:
            pass
        variants += [expect + "+1", expect + "1", "0" if expect != "0" else "1"]
        for v in variants:
            c = _replace_boxed(answer, v)
            if c and c != answer:
                out.append(("boxed-edit", c))
    elif kind == "contains":
        for e in spec["expect"]:
            repl = next((d for d in decoys
                         if _norm(d) and _norm(e) not in _norm(d)), None)
            if repl is None:
                continue
            c = re.sub(re.escape(e), repl, answer, flags=re.I)
            if c != answer:
                out.append((f"entity-swap:{e}", c))
    elif kind == "choice":
        expect = str(spec["expect"]).upper()
        matches = list(re.finditer(r"\b([A-H])\b", answer.upper()))
        if matches:
            m = matches[-1]
            for L in "ABCD":
                if L != expect:
                    out.append((f"choice-flip:{L}",
                                answer[:m.start(1)] + L + answer[m.end(1):]))
    elif kind == "code":
        for c in _code_candidates(answer):
            if _runs_clean_wrong(task, c):
                out.append(("code-edit", c))
    return out


def _norm(text: str) -> str:
    from experiments.cascade.checkers import _normalize
    return _normalize(text)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path,
                    default=CASCADE / "results" / "jev-readout-20261006")
    ap.add_argument("--max-neg", type=int, default=3)
    args = ap.parse_args()

    tasks = json.loads((CASCADE / "tasks-bench.json").read_text())["tasks"]
    by_id = {t["id"]: t for t in tasks}

    # recorded answers, newest report wins per task
    recorded: dict[str, dict] = {}
    for rel in REPORTS:
        report = json.loads((CASCADE / rel).read_text())
        for rec in report.get("records", []):
            if rec.get("scion_answer"):
                recorded[rec["id"]] = {**rec, "_source": Path(rel).parent.name}

    # decoy pool: other expected strings of the same checker family
    decoys = list(DECOYS) + [
        e for t in tasks for e in (
            t["checker"].get("expect")
            if isinstance(t["checker"].get("expect"), list)
            else [t["checker"].get("expect")]
        ) if isinstance(e, str) and e.strip()
    ]

    items: list[dict] = []
    para_stats = {"pos": 0, "neg_natural": 0, "neg_perturbed": 0, "skipped": 0}
    per_kind: dict[str, dict] = {}
    for tid, rec in sorted(recorded.items()):
        task = by_id.get(tid)
        if task is None:
            para_stats["skipped"] += 1
            continue
        kind = task["checker"]["type"]
        stats = per_kind.setdefault(kind, {"pos": 0, "neg": 0, "no_neg": 0})
        answer = rec["scion_answer"]
        ok = rec.get("scion_ok")
        if ok is None:
            ok = check(task, answer)
        base = {"task_id": tid, "bucket": task["bucket"], "split": task["split"],
                "prompt": task["prompt"]}
        if ok:
            items.append({**base, "item_id": f"{tid}:pos", "label": 1,
                          "origin": "recorded", "candidate": answer})
            stats["pos"] += 1
            para_stats["pos"] += 1
        else:
            items.append({**base, "item_id": f"{tid}:nat", "label": 0,
                          "origin": "recorded-wrong", "candidate": answer})
            para_stats["neg_natural"] += 1
            stats["neg"] += 1
            continue
        # perturbations of the correct answer
        seen: set[str] = {answer}
        n = 0
        for label, cand in perturbations(task, answer, decoys):
            if n >= args.max_neg:
                break
            if cand in seen:
                continue
            seen.add(cand)
            if check(task, cand):
                continue  # verified-wrong only
            n += 1
            items.append({**base, "item_id": f"{tid}:neg{n}", "label": 0,
                          "origin": f"perturbed:{label}", "candidate": cand})
        para_stats["neg_perturbed"] += n
        stats["neg"] += n
        if n == 0:
            stats["no_neg"] += 1

    # verification pass: every positive passes, every negative fails
    bad = []
    for item in items:
        task = by_id[item["task_id"]]
        got = check(task, item["candidate"])
        if got != bool(item["label"]):
            bad.append(item["item_id"])
    if bad:
        raise SystemExit(f"checker verification failed for {len(bad)} items: {bad[:5]}")

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "judge-bench.jsonl"
    payload = "\n".join(json.dumps(i, sort_keys=True) for i in items) + "\n"
    jsonl.write_text(payload)
    digest = hashlib.sha256(payload.encode()).hexdigest()

    by_split = {}
    for item in items:
        s = by_split.setdefault(item["split"], {"pos": 0, "neg": 0})
        s["pos" if item["label"] else "neg"] += 1
    manifest = {
        "created": "2026-10-06",
        "source": "cascade-bench-v1 (experiments/cascade/tasks-bench.json)",
        "reports": REPORTS,
        "rule": ("positives = recorded answers the checker accepts; negatives = "
                 "checker-verified perturbations (num-flip, boxed-edit, "
                 "entity-swap, choice-flip, code-edit) + recorded wrong answers; "
                 "test split frozen for measurement"),
        "items": len(items),
        "split_counts": by_split,
        "stats": para_stats,
        "per_checker": per_kind,
        "sha256": digest,
    }
    (out_dir / "judge-bench.manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))
    print(f"wrote {jsonl} ({len(items)} items, sha {digest[:12]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
