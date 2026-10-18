#!/usr/bin/env python3
"""Repair pass: does the 176B fix its own caught wrongs when shown the evidence?

The cascade is detect -> repair -> escalate.  Detect is measured (the gold-free stack catches 6 of
the 176B's 9 organic-66 wrongs at 0 false rejects).  Escalate is blocked (no provider named).
Repair is the untested leg, and it is the cheapest one to test: no window, no second model, no
gold -- just re-ask the same serve with the detector's finding in the prompt.

Three arms, all against the live 176B on :8081:

  A  evidence repair   draft + the specific finding, on the 6 detected-wrong items
  B  generic recheck   draft only, on the same items        ("think again" alone, no finding)
  C  harm control      generic recheck on correct, unflagged items (does re-asking break them?)

Arm B separates the finding from the mere act of re-asking; arm C prices the risk that a false
flag (should one ever fire) costs a correct answer.  Correctness of each repaired answer is scored
with the task's own checker -- gold, therefore eval-only, never inside the prompt.

    .venv/bin/python experiments/cascade/repair_pass.py --detect-only
    .venv/bin/python experiments/cascade/repair_pass.py --out <dir>
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "cascade"))

from checkers import check  # noqa: E402
from consistency_check import check_answer  # noqa: E402

SYSTEM = "Solve the problem, showing your work. End your answer with 'Final answer: <number>'."
CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)
DEFAULT_SPECS = ["tasks-bench.json", "tasks-community.json", "tasks-community-code.json"]


# ---------------------------------------------------------------- engine side

def post(url: str, path: str, body: dict, timeout: float = 300.0) -> dict:
    req = urllib.request.Request(url.rstrip("/") + path,
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def generate(url: str, user_text: str, max_tokens: int, temperature: float
             ) -> tuple[str, str | None, int | None]:
    """Same request shape as dual_generation.py, so answers are comparable across runs."""
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_text}]
    body = {"messages": msgs, "max_tokens": max_tokens, "temperature": temperature}
    try:
        out = post(url, "/v1/chat/completions",
                   {**body, "chat_template_kwargs": {"enable_thinking": False}})
    except Exception:  # noqa: BLE001 - engines without thinking control
        out = post(url, "/v1/chat/completions", body)
    ch = out["choices"][0]
    return (ch["message"]["content"], ch.get("finish_reason"),
            (out.get("usage") or {}).get("completion_tokens"))


# ---------------------------------------------------------------- detect side

def run_tests(spec: dict, text: str) -> tuple[bool, str]:
    """Run the draft's last code block against the task's own tests; return (ok, output tail).

    Mirrors checkers.py's `code` branch (prelude + code + tests + footer) but keeps the process
    output, because the traceback is the repair evidence the registry asks for.
    """
    ck = spec["checker"]
    blocks = CODE_BLOCK.findall(text or "")
    if not blocks:
        return False, "the draft contains no python code block"
    tests = ck["tests"]
    if isinstance(tests, list):
        tests = "\n".join(tests)
    program = (str(ck.get("prelude", "")) + blocks[-1].strip() + "\n\n"
               + tests + "\n" + str(ck.get("footer", "print('PASS')")) + "\n")
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(program)
        path = fh.name
    try:
        proc = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=15)
        ok = proc.returncode == 0 and "PASS" in proc.stdout
        tail = (proc.stderr or proc.stdout or "").strip()
        return ok, tail[-700:]
    except subprocess.TimeoutExpired:
        return False, "the draft's code did not finish within 15 seconds"
    finally:
        try:
            Path(path).unlink()
        except OSError:
            pass


def detect(row: dict, spec: dict | None) -> dict:
    """Gold-free detectors only: the arithmetic verifier and the task's own tests."""
    found: dict = {}
    cons = check_answer(row["candidate"])
    if cons["flagged"]:
        found["consistency"] = cons["inconsistent"]
    if spec and (spec.get("checker") or {}).get("type") == "code":
        ok, tail = run_tests(spec, row["candidate"])
        if not ok:
            found["code_exec"] = tail
    return found


# ---------------------------------------------------------------- prompt side

DRAFT_HEAD = "Here is a draft answer to the task above:\n\n---\n{}\n---\n\n"


def evidence_prompt(row: dict, found: dict) -> tuple[str, str]:
    """Return (arm detail, prompt).  The finding is quoted, never the expected answer."""
    head = DRAFT_HEAD.format(row["candidate"].strip())
    if "consistency" in found:
        e = found["consistency"][0]
        lhs, rhs = e["lhs"], e["rhs"]
        fmt = lambda v: (f"{v:g}")
        detail = (f"A check of the draft's own arithmetic found that this step does not hold: "
                  f"`{e['equation']}` (the left side is {fmt(lhs)}, the right side is {fmt(rhs)}). "
                  "Find the mistake, redo the work correctly, and end with 'Final answer: <number>'.")
        return "consistency", row["prompt"].rstrip() + "\n\n" + head + detail
    detail = ("Running the draft against the task's own tests fails:\n\n```\n"
              + found["code_exec"] + "\n```\n\n"
              "Fix the implementation. Return only the corrected implementation, in a single "
              "```python code block.")
    return "code_exec", row["prompt"].rstrip() + "\n\n" + head + detail


def generic_prompt(row: dict) -> str:
    detail = ("Recheck the draft carefully for any arithmetic or logical mistakes. If something is "
              "wrong, correct it. Give the final answer -- for a coding task, the corrected "
              "implementation in a single ```python code block; otherwise end with "
              "'Final answer: <number>'.")
    return row["prompt"].rstrip() + "\n\n" + DRAFT_HEAD.format(row["candidate"].strip()) + detail


# ---------------------------------------------------------------- scoring

def final_answer(text: str) -> str | None:
    m = re.findall(r"\\boxed\{(.+?)\}", text or "")
    if m:
        return m[-1].strip()
    m = re.findall(r"final answer[^0-9\-+]*([-+]?\d[\d,]*(?:\.\d+)?)", text or "", re.I | re.S)
    if m:
        return m[-1].replace(",", "")
    m = re.findall(r"\b([A-D])\b", (text or "").upper())
    return m[-1] if m else None


def gold_ok(spec: dict | None, text: str) -> bool | None:
    if not spec:
        return None
    try:
        return bool(check(spec, text))
    except Exception:  # noqa: BLE001 - malformed spec/answer is not a pass
        return None


# ---------------------------------------------------------------- main

def load_bench(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def load_specs(files: list[Path]) -> dict[str, dict]:
    specs: dict[str, dict] = {}
    for f in files:
        d = json.loads(f.read_text())
        for it in (d.get("items") or d.get("tasks") or []):
            if "checker" in it:
                specs.setdefault(str(it.get("id") or it.get("task_id")), it)
    return specs


def pick_control(rows: list[dict], found: dict, n: int) -> list[dict]:
    """Correct, unflagged items, round-robin over families so every checker kind is covered."""
    clean = [r for r in rows if r["label"] == 1 and r["item_id"] not in found]
    families: dict[str, list[dict]] = {}
    for r in clean:
        families.setdefault(str(r["task_id"]).split("-")[0], []).append(r)
    for v in families.values():
        v.sort(key=lambda r: r["item_id"])
    out: list[dict] = []
    names = sorted(families)
    while len(out) < n and any(families[name] for name in names):
        for name in names:
            if families[name] and len(out) < n:
                out.append(families[name].pop(0))
    return sorted(out, key=lambda r: r["item_id"])


def one_request(args, row, spec, arm, detail, user_text) -> dict:
    t0 = time.time()
    row_out = {"item_id": row["item_id"], "task_id": row["task_id"], "arm": arm,
               "detail": detail, "orig_label": row["label"],
               "kind": (spec or {}).get("checker", {}).get("type"),
               "gold_ok_before": row["label"] == 1}
    try:
        text, finish, ntok = generate(args.endpoint, user_text, args.max_tokens, args.temperature)
        row_out.update(text=text, final=final_answer(text), finish_reason=finish,
                       completion_tokens=ntok)
        row_out["gold_ok_after"] = gold_ok(spec, text)
        row_out["flags_after"] = check_answer(text)["flagged"]
        if spec and (spec.get("checker") or {}).get("type") == "code":
            ok, _ = run_tests(spec, text)
            row_out["code_ok_after"] = ok
    except Exception as exc:  # noqa: BLE001
        row_out.update(text="", error=f"{type(exc).__name__}: {exc}", gold_ok_after=None)
    row_out["ms"] = (time.time() - t0) * 1000
    return row_out


def write_report(out: Path, rows: list[dict], det: dict, control: list[dict],
                 starts: dict, endpoint: str, bench: Path, specs: list[Path]) -> None:
    def rate(sub: list[dict], key="gold_ok_after") -> str:
        n = len(sub)
        if not n:
            return "n/a"
        ok = sum(1 for r in sub if r.get(key) is True)
        return f"{ok}/{n}"

    a = [r for r in rows if r["arm"] == "A"]
    b = [r for r in rows if r["arm"] == "B"]
    c = [r for r in rows if r["arm"] == "C"]
    repaired_a = [r for r in a if r.get("gold_ok_after") is True]
    repaired_b = [r for r in b if r.get("gold_ok_after") is True]
    harmed_c = [r for r in c if r.get("gold_ok_after") is False]
    cap = sum(1 for r in rows if r.get("finish_reason") == "length")
    lines = [
        "# Repair pass on the 176B's caught wrongs (2026-10-18)",
        "",
        f"Same serve, same prompt shape as the harvest; the live 176B on `{endpoint}` answered a "
        "second time, shown its own draft and (arm A) the finding one of the gold-free detectors "
        "produced.  Gold is used only to score the outcome.",
        "",
        "```sh",
        ".venv/bin/python experiments/cascade/repair_pass.py \\",
        f"    --bench {bench} \\",
        f"    --out {out}",
        "```",
        "",
        "## The detected set (gold-free, recomputed here)",
        "",
        "| detector | items |",
        "| --- | --- |",
    ]
    for name in ("consistency", "code_exec"):
        ids = [i for i, f in sorted(det.items()) if name in f]
        lines.append(f"| `{name}` | {len(ids)}: {', '.join(ids) or '-'} |")
    lines += [
        f"| **union** | **{len(det)}** of the 9 organic wrongs |",
        "",
        f"False rejects on the {sum(1 for r in load_bench(bench) if r['label'] == 1)} correct items: "
        f"{len([i for i in det if starts[i][0] == 1])}.",
        "",
        "## Arms",
        "",
        "| arm | items | corrected by the repair | still wrong | broken (was right) | hit token cap |",
        "| --- | --- | --- | --- | --- | --- |",
        f"| A evidence repair | {len(a)} | {rate(a)} | {len(a) - len(repaired_a)} | 0 | "
        f"{sum(1 for r in a if r.get('finish_reason') == 'length')} |",
        f"| B generic recheck | {len(b)} | {rate(b)} | {len(b) - len(repaired_b)} | 0 | "
        f"{sum(1 for r in b if r.get('finish_reason') == 'length')} |",
        f"| C harm control | {len(c)} | 0 | — | {len(harmed_c)} (of {len(c)}) | "
        f"{sum(1 for r in c if r.get('finish_reason') == 'length')} |",
        "",
    ]
    per_item = [f"| `{r['item_id']}` | {r['detail']} | "
                f"{'yes' if r.get('gold_ok_after') else 'no'} | "
                f"{'yes' if r.get('gold_ok_after') else 'no'} | "
                f"{'yes' if r.get('gold_ok_after') else 'no'} |" for r in a]
    lines += ["## Per item (A evidence / B generic / C control, where the item appears)",
              "",
              "| item | finding | A | B | C |",
              "| --- | --- | --- | --- | --- |"]
    for item in sorted({r["item_id"] for r in rows}):
        det_name = ",".join(sorted(det.get(item, {}).keys())) or "-"
        cells = []
        for arm in ("A", "B", "C"):
            hit = next((r for r in rows if r["item_id"] == item and r["arm"] == arm), None)
            cells.append("yes" if (hit or {}).get("gold_ok_after") is True
                         else ("no" if hit else "·"))
        lines.append(f"| `{item}` | {det_name} | {cells[0]} | {cells[1]} | {cells[2]} |")
    lines += [
        "",
        "## Reading",
        "",
        f"- The repair leg costs one extra pass on the detected set "
        f"({sum(1 for r in rows if r['arm'] in 'AB')} requests here) and no swap.",
        f"- Cap hits in this pass: {cap}.  A capped answer is the budget's fault, not the model's.",
        "- If arm A ≈ arm B, the detector's *finding* is decoration: the value is just 'ask twice'.",
        "- If arm C breaks answers, a false flag is not free, and the detector's 0/57 precision is "
        "the whole reason the arm is safe.",
        "",
    ]
    (out / "REPAIR.md").write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    root = REPO / "experiments" / "cascade"
    ap.add_argument("--endpoint", default="http://127.0.0.1:8081")
    ap.add_argument("--bench", type=Path,
                    default=root / "results" / "dual-generation-20261018" / "organic-66.jsonl")
    ap.add_argument("--tasks", type=Path, nargs="+",
                    default=[root / f for f in DEFAULT_SPECS])
    ap.add_argument("--out", type=Path,
                    default=root / "results" / "repair-pass-20261018")
    ap.add_argument("--max-tokens", type=int, default=640)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--control", type=int, default=8, help="arm C size")
    ap.add_argument("--detect-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print prompts, send nothing")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = load_bench(args.bench)
    specs = load_specs([f for f in args.tasks if f.exists()])
    print(f"bench {len(rows)} items, {len(specs)} specs, endpoint {args.endpoint}")

    # Gold agreement first: the checker must reproduce the stored labels, or the scoring is void.
    agree = {}
    for spec_file in args.tasks:
        if not spec_file.exists():
            continue
        s2 = load_specs([spec_file])
        hit = tot = 0
        for r in rows:
            spec = s2.get(r["task_id"])
            if spec and "checker" in spec:
                ok = gold_ok(spec, r["candidate"])
                if ok is not None:
                    tot += 1
                    hit += int(ok == (r["label"] == 1))
        agree[spec_file.name] = f"{hit}/{tot}"
    print("checker-vs-label agreement per spec file:",
          ", ".join(f"{k} {v}" for k, v in agree.items()))

    det: dict[str, dict] = {}
    starts: dict[str, tuple[int, str]] = {}
    for r in rows:
        spec = specs.get(r["task_id"])
        found = detect(r, spec)
        starts[r["item_id"]] = (r["label"], str(r["task_id"]))
        if found:
            det[r["item_id"]] = found
    print(f"detected wrongs: {len(det)} of {sum(1 for r in rows if r['label'] == 0)}")
    for item, f in sorted(det.items()):
        print(f"  {item:<18} label={starts[item][0]} detectors={','.join(sorted(f))} "
              f"({'; '.join(str(v)[:60] for v in f.values())})")
    false_rejects = [i for i, f in det.items() if starts[i][0] == 1]
    print(f"false rejects on correct items: {len(false_rejects)} {false_rejects}")

    control = pick_control(rows, det, args.control)
    print("control:", [r['item_id'] for r in control])

    if args.detect_only:
        return 0

    by_id = {r["item_id"]: r for r in rows}
    plan: list[tuple[str, dict, str, str]] = []
    for item in sorted(det):
        row, spec = by_id[item], specs.get(by_id[item]["task_id"])
        detail, prompt = evidence_prompt(row, det[item])
        plan.append(("A", row, detail, prompt))
        plan.append(("B", row, detail, generic_prompt(row)))
    for row in control:
        plan.append(("C", row, "-", generic_prompt(row)))

    if args.dry_run:
        for arm, row, detail, prompt in plan:
            print(f"\n{'=' * 70}\narm {arm}  {row['item_id']}  ({detail})\n{'=' * 70}\n"
                  f"{prompt[:1400]}")
        return 0

    done: list[dict] = []
    t0 = time.time()
    for n, (arm, row, detail, prompt) in enumerate(plan, 1):
        res = one_request(args, row, specs.get(row["task_id"]), arm, detail, prompt)
        done.append(res)
        print(f"[{n}/{len(plan)}] {arm} {row['item_id']:<18} "
              f"after={'ok' if res.get('gold_ok_after') else 'no'} "
              f"{res['ms'] / 1000:.1f}s tok={res.get('completion_tokens')} "
              f"finish={res.get('finish_reason')} {res.get('error', '')}", flush=True)
        with (args.out / "repair.jsonl").open("a") as fh:
            fh.write(json.dumps(res) + "\n")

    summary = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "endpoint": args.endpoint, "bench": str(args.bench), "tasks": [str(t) for t in args.tasks],
        "detected": {k: sorted(v) for k, v in
                     ((n, [i for i, f in det.items() if n in f]) for n in ("consistency", "code_exec"))},
        "false_rejects": false_rejects,
        "arms": {arm: {
            "n": sum(1 for r in done if r["arm"] == arm),
            "repaired": sum(1 for r in done if r["arm"] == arm and r.get("gold_ok_after") is True),
            "still_wrong": sum(1 for r in done if r["arm"] == arm and r.get("gold_ok_after") is False),
            "cap_hit": sum(1 for r in done if r["arm"] == arm and r.get("finish_reason") == "length"),
        } for arm in "ABC"},
        "wall_s": time.time() - t0,
        "checker_vs_label": agree,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    write_report(args.out, done, det, control, starts, args.endpoint, args.bench,
                 [f for f in args.tasks if f.exists()])
    print(f"\nwall {summary['wall_s']:.0f}s\narms {json.dumps(summary['arms'])}\n"
          f"wrote {args.out}/summary.json + REPAIR.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
