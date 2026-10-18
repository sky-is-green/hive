#!/usr/bin/env python3
"""The local cascade, end to end: solve -> detect -> repair -> human gate.

Policy comes from `bench_registry.json`'s `runtime` column.  A gold-dependent bench (gsm8k, math,
truthfulqa) has no answer key at runtime, so the only checks that may run inside the loop are its
declared gold-free fallbacks: `consistency` always, `code_exec` when the task supplies tests.
That is the whole detector set.  The loop never sees `checker.expect` -- specs pass through
`runtime_spec()`, which strips everything but the frame and the task's own tests, and the gold-based
scoring is a separate post-loop step behind `--score` (eval-only, never consulted by the loop).

Flow per item:

  solve     one request to the local 176B at `--temperature` (0.7 reproduces the harvest's
            regime); a cap hit retries once at double the budget, and a draft that is still cut
            off escalates -- truncation is never scored as an answer
  detect    gold-free findings (arithmetic mismatch, or the task's tests failing)
  repair    re-ask with the finding, greedy (`--repair-temperature 0`, the strongest local
            re-attempt); the repaired answer is re-framed to the task's interface
            `--rounds` (default 1); a round that returns byte-identical text stops the loop
  gate      still flagged, or nothing consumable extracted -> human gate: verbose on the first
            trigger, compact afterwards; solve locally (`keep` is a sub-option of solve) or
            escalate to the API tier -- and there is no API tier until a provider is named.

Verdicts: `accepted`, `accepted_repaired`, `escalated`, `error`.

    .venv/bin/python experiments/cascade/cascade.py --reuse-drafts \
        --items gsm8k-0033:hs1,humaneval-0015:hs2,gsm8k-0005:hg --score
    .venv/bin/python experiments/cascade/cascade.py --out <dir> --score
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments" / "cascade"))

from checkers import check  # noqa: E402 - eval-only, used by the post-loop scoring step
from repair_pass import (  # noqa: E402 - one home for the request shape and the detectors
    evidence_prompt, final_answer, generate, run_tests,
)

CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)
GOLD_WORDS = ("expect", "gold", "answer_key", "solution")


# ---------------------------------------------------------------- policy guard

def runtime_spec(spec: dict) -> dict:
    """Everything the loop is allowed to know: the answer frame and the task's own tests.

    `checker.expect` is the answer key.  It is dropped here, so no code path downstream -- detect,
    repair, canonicalise, gate -- can read it even by accident.
    """
    ck = spec.get("checker") or {}
    rs = {"kind": ck.get("type")}
    for key in ("tests", "prelude", "footer"):
        if key in ck:
            rs[key] = ck[key]
    if any(word in key for key in rs for word in GOLD_WORDS):
        raise AssertionError(f"gold-shaped key reached the loop: {sorted(rs)}")
    return rs


def detectors_available(rs: dict) -> list[str]:
    available = ["consistency"]
    if rs.get("kind") == "code" and "tests" in rs:
        available.append("code_exec")
    return available


# ---------------------------------------------------------------- loop stages

def detect(text: str, rs: dict) -> list[dict]:
    """Gold-free findings about `text`.  Empty list means nothing found -- not that it is right."""
    from consistency_check import check_answer

    findings = []
    verdict = check_answer(text or "")
    if verdict["flagged"]:
        findings.append({"detector": "consistency", "finding": verdict["inconsistent"]})
    if rs.get("kind") == "code" and "tests" in rs:
        ok, tail = run_tests({"checker": rs}, text or "")
        if not ok:
            findings.append({"detector": "code_exec", "finding": tail})
    return findings


def ask(endpoint: str, prompt: str, max_tokens: int, temperature: float
        ) -> tuple[str, str | None, int | None, int]:
    """One generation with one budget escalation.

    A cap hit is the harness's ceiling, not an answer: the truncation rule says it must never be
    scored as a wrong answer, and a draft that stops mid-derivation has no final answer to accept.
    So this retries once at twice the budget and reports how many generations it took.
    """
    text, finish, ntok = generate(endpoint, prompt, max_tokens, temperature)
    if finish == "length":
        text2, finish2, ntok2 = generate(endpoint, prompt, max_tokens * 2, temperature)
        return text2, finish2, ntok2, 2
    return text, finish, ntok, 1


def canonical(kind: str | None, text: str) -> str | None:
    """Re-frame a repaired answer to the task's interface.  Values are never changed, only framing."""
    if not text:
        return None
    if kind == "code":
        blocks = CODE_BLOCK.findall(text)
        # the task asks for a fenced block and the code checker reads one; the canonical form
        # keeps the fence (a bare snippet is a different, unreadable artifact)
        return f"```python\n{blocks[-1].strip()}\n```" if blocks else None
    value = final_answer(text)
    if value is None:
        return None
    if kind == "boxed":
        return f"\\boxed{{{value}}}"
    if kind == "number":
        return f"Final answer: {value}"
    if kind == "choice":
        return f"Final answer: {str(value).upper()}"
    return value


def gate_text(meta: dict, prompt: str, draft: str, findings: list[dict], attempt: str | None,
              verbose: bool) -> str:
    """The human gate's rendering: full context on the first trigger, a digest afterwards."""
    if not verbose:
        f0 = findings[0]
        return (f"  {meta['item_id']}  kind={meta['kind']}  still flagged after repair\n"
                f"  finding: {f0['detector']}: {str(f0['finding'])[:160]}\n"
                f"  draft tail:   {(draft or '')[-200:].strip()}\n"
                f"  attempt tail: {(attempt or '-')[-200:].strip()}")
    lines = [
        f"  item       : {meta['item_id']}   (task {meta['task_id']}, kind {meta['kind']})",
        f"  detections : {', '.join(f['detector'] for f in findings)}",
        "",
        "  prompt:",
        "\n".join("    " + l for l in prompt.strip().splitlines()[:40]),
        "",
        "  the model's draft:",
        "\n".join("    " + l for l in (draft or '').strip().splitlines()[:60]),
        "",
    ]
    for f in findings:
        lines += [f"  the {f['detector']} finding:",
                  "    " + str(f["finding"]).replace("\n", "\n    ")[:1200], ""]
    if attempt:
        lines += ["  the repair attempt:",
                  "\n".join("    " + l for l in attempt.strip().splitlines()[:60]), ""]
    return "\n".join(lines)


def ask_human() -> tuple[str, str | None]:
    """Returns (action, payload).  `keep` exists only as a sub-option of solve."""
    while True:
        try:
            choice = input("action [solve] fix it yourself / [api] escalate / [skip] leave it: ")
        except EOFError:
            return "skip", None
        choice = choice.strip().lower()
        if choice.startswith("s"):
            try:
                answer = input("your answer (or 'keep' to keep the model's draft): ").strip()
            except EOFError:
                return "skip", None
            return "solve", ("keep" if answer.lower() in ("", "keep") else answer)
        if choice.startswith("a"):
            print("  no API tier is configured yet (blocked on a provider name); try solve or skip")
            continue
        return "skip", None


# ---------------------------------------------------------------- eval-only scoring

def rescore(run: Path, bench: Path, specs: dict[str, dict], out: Path | None) -> int:
    """Re-score an existing cascade.jsonl without touching the engine.

    Written twice now: the first reading of a run was wrong because the *scorer* was wrong, and
    re-deriving the score must not cost another pass over the serve.  For repaired code rows the
    raw fenced text is scored (the stored canonical predates the fenced form); everything else is
    scored from its artifact.
    """
    rows = [json.loads(l) for l in run.read_text().splitlines() if l.strip()]
    per: dict[str, dict] = {}
    for r in rows:
        spec = specs.get(r["task_id"])
        text = r.get("artifact")
        if (spec or {}).get("checker", {}).get("type") == "code" and r.get("rounds"):
            text = r["rounds"][-1].get("text")
        per[r["item_id"]] = score_item(spec, text)
    verdicts: dict[str, int] = {}
    for r in rows:
        verdicts[r["verdict"]] = verdicts.get(r["verdict"], 0) + 1

    lines = [f"# Cascade scoring (gold-based, eval-only) -- {run.parent.name}", "",
             "Re-derived by `cascade.py --rescore`; the loop itself never reads gold.  For "
             "repaired code rows the raw fenced repair text is scored.", "",
             "| verdict | n | content-correct artifact |", "| --- | --- | --- |"]
    for v in sorted(verdicts, key=lambda v: -verdicts[v]):
        ids = [r["item_id"] for r in rows if r["verdict"] == v]
        ok = sum(1 for i in ids if per[i]["content"] is True)
        lines.append(f"| {v} | {len(ids)} | {ok} |")
    accepted_ids = [r["item_id"] for r in rows if r["verdict"] == "accepted"]
    silent = [i for i in accepted_ids if per[i]["content"] is False]
    repaired = [r["item_id"] for r in rows if r["verdict"] == "accepted_repaired"]
    repaired_ok = [i for i in repaired if per[i]["content"] is True]
    final_ok = sum(1 for s in per.values() if s["content"] is True)
    lines += [
        "",
        f"- Unrepaired accepts that are wrong (the detectors' blind spot, not the gate's): "
        f"{len(silent)} ({', '.join(sorted(silent)) or '-'}).",
        f"- Repaired and accepted: {len(repaired)}, correct: {len(repaired_ok)} "
        f"({', '.join(sorted(repaired_ok)) or '-'}).",
        f"- Served-correct artifacts: {final_ok}/{len(rows)}.",
        "",
        "| item | verdict | content |",
        "| --- | --- | --- |",
    ]
    for r in sorted(rows, key=lambda r: r["item_id"]):
        v = per[r["item_id"]]["content"]
        lines.append(f"| `{r['item_id']}` | {r['verdict']} | "
                     f"{'pass' if v is True else ('fail' if v is False else 'n/a')} |")
    text = "\n".join(lines) + "\n"
    if out:
        (out / "SCORING.md").write_text(text)
        (out / "scoring.json").write_text(json.dumps(
            {"per_item": per, "verdicts": verdicts, "blind_spot": sorted(silent),
             "repaired_correct": sorted(repaired_ok), "served_correct": final_ok,
             "items": len(rows)}, indent=1))
        print(f"wrote {out}/SCORING.md + scoring.json")
    print("\n".join(lines[:12]))
    return 0


def score_item(spec: dict | None, text: str | None) -> dict:
    """Strict (registered checker) and content (final-answer) correctness.  Gold, eval-only."""
    from analyze_repair import tolerant_ok

    if spec is None or text is None:
        return {"strict": None, "content": None}
    try:
        strict = bool(check(spec, text))
    except Exception:  # noqa: BLE001
        strict = None
    content = strict if (spec.get("checker") or {}).get("type") == "code" else tolerant_ok(spec, text)
    return {"strict": strict, "content": content}


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    root = REPO / "experiments" / "cascade"
    ap.add_argument("--bench", type=Path,
                    default=root / "results" / "dual-generation-20261018" / "organic-66.jsonl")
    ap.add_argument("--tasks", type=Path, nargs="+", default=[root / "tasks-bench.json"])
    ap.add_argument("--endpoint", default="http://127.0.0.1:8081")
    ap.add_argument("--out", type=Path, default=root / "results" / "cascade-e2e-20261018")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--items", default="", help="comma-separated item ids (regression runs)")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="sampling for the first attempt; 0.7 reproduces the harvest's regime")
    ap.add_argument("--repair-temperature", type=float, default=0.0,
                    help="repairs stay greedy: the strongest local re-attempt")
    ap.add_argument("--human", choices=("auto", "interactive"), default="auto")
    ap.add_argument("--rescore", type=Path, default=None,
                    help="no engine: re-score an existing cascade.jsonl into SCORING.md")
    ap.add_argument("--reuse-drafts", action="store_true",
                    help="take the draft from the bench instead of calling the model")
    ap.add_argument("--score", action="store_true", help="post-loop, gold-based, eval-only")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    specs: dict[str, dict] = {}
    for f in args.tasks:
        d = json.loads(f.read_text())
        for it in (d.get("items") or d.get("tasks") or []):
            if "checker" in it:
                specs.setdefault(str(it.get("id") or it.get("task_id")), it)

    items = [json.loads(l) for l in args.bench.read_text().splitlines() if l.strip()]
    if args.rescore:
        args.out.mkdir(parents=True, exist_ok=True)
        return rescore(args.rescore, args.bench, specs, args.out)
    if args.items:
        keep = {s.strip() for s in args.items.split(",") if s.strip()}
        items = [r for r in items if r["item_id"] in keep]
    if args.limit:
        items = items[:args.limit]
    print(f"{len(items)} items; endpoint {args.endpoint}; human gate {args.human}; "
          f"drafts {'reused from the bench' if args.reuse_drafts else 'generated'}")

    gate_log: list[str] = []
    rows: list[dict] = []
    requests = 0
    t0 = time.time()

    for n, item in enumerate(items, 1):
        spec = specs.get(item["task_id"])
        rs = runtime_spec(spec) if spec else {"kind": None}
        row: dict = {"item_id": item["item_id"], "task_id": item.get("task_id"), "kind": rs.get("kind"),
                     "detectors": detectors_available(rs), "rounds": [],
                     "requests": 0, "started": time.strftime("%H:%M:%S")}

        draft = item.get("candidate") or ""
        if not args.reuse_drafts:
            ts = time.time()
            try:
                draft, finish, ntok, used = ask(args.endpoint, item["prompt"], args.max_tokens,
                                                args.temperature)
            except Exception as exc:  # noqa: BLE001
                row.update(verdict="error", error=f"solve: {type(exc).__name__}: {exc}")
                rows.append(row)
                print(f"[{n}/{len(items)}] {item['item_id']} ERROR {row['error']}", flush=True)
                continue
            requests += used
            row["requests"] += used
            row["solve"] = {"ms": (time.time() - ts) * 1000, "finish_reason": finish,
                            "completion_tokens": ntok, "chars": len(draft), "generations": used}
            if finish == "length":
                row["truncated_at_solve"] = True
        row["draft"] = draft

        findings = detect(draft, rs)
        if row.get("truncated_at_solve"):
            findings = [{"detector": "truncated",
                         "finding": f"still cut off at {args.max_tokens * 2} tokens"}] + findings
        row["detect"] = [f["detector"] for f in findings]
        attempt = None
        attempts = 0
        while findings and attempts < args.rounds and not row.get("truncated_at_solve"):
            attempts += 1
            fake_row = {"prompt": item["prompt"], "candidate": draft}
            detail, prompt = evidence_prompt(fake_row, {f["detector"]: f["finding"] for f in findings})
            ta = time.time()
            try:
                text, finish, ntok, used = ask(args.endpoint, prompt, args.max_tokens,
                                               args.repair_temperature)
            except Exception as exc:  # noqa: BLE001
                row["rounds"].append({"detail": detail, "ms": (time.time() - ta) * 1000,
                                      "error": f"{type(exc).__name__}: {exc}"})
                break
            requests += used
            row["requests"] += used
            after = detect(text, rs)
            row["rounds"].append({
                "detail": detail, "ms": (time.time() - ta) * 1000, "finish_reason": finish,
                "completion_tokens": ntok, "chars": len(text), "generations": used,
                "same_as_draft": text.strip() == (draft or "").strip(),
                "still_flagged": [f["detector"] for f in after],
                "canonical": canonical(rs.get("kind"), text), "text": text,
            })
            attempt = text
            if finish == "length":
                row["rounds"][-1]["truncated"] = True
                break
            if text.strip() == (draft or "").strip():
                row["rounds"][-1]["stopped"] = "identical to the draft; another round cannot differ"
                break
            findings, draft = after, text

        # the consumer gets: the draft if it was clean, else the re-framed repair, else nothing
        artifact = None
        if not findings:
            if attempts == 0:
                row["verdict"] = "accepted"
                artifact = row["draft"]
            else:
                artifact = row["rounds"][-1]["canonical"]
                if artifact:
                    row["verdict"] = "accepted_repaired"
                else:
                    row["verdict"] = "escalated"
                    findings = [{"detector": "extraction",
                                 "finding": "the repair produced no consumable final answer"}]
        else:
            row["verdict"] = "escalated"
        row["artifact"] = artifact

        if row["verdict"] == "escalated":
            verbose = not gate_log
            meta = {"item_id": item["item_id"], "task_id": item.get("task_id"), "kind": row["kind"]}
            rendered = gate_text(meta, item["prompt"], row["draft"], findings, attempt, verbose)
            header = ("=== CASCADE ESCALATION (first trigger, verbose) ===" if verbose
                      else f"=== CASCADE ESCALATION #{len(gate_log) + 1} (compact) ===")
            if args.human == "interactive":
                print(header)
                print(rendered)
                action, payload = ask_human()
                row["gate"] = {"action": action, "payload": payload}
                if action == "solve":
                    row["verdict"] = "solved_by_human"
                    row["artifact"] = row["draft"] if payload == "keep" else payload
                    row["gate_log_index"] = len(gate_log)
            else:
                row["gate"] = {"action": "auto_recorded"}
            gate_log.append(header + "\n" + rendered + "\n")
            print(f"[{n}/{len(items)}] {item['item_id']:<18} ESCALATED "
                  f"({', '.join(f['detector'] for f in findings)})", flush=True)
        else:
            print(f"[{n}/{len(items)}] {item['item_id']:<18} {row['verdict']:<17} "
                  f"detect={','.join(row['detect']) or '-'} rounds={len(row['rounds'])} "
                  f"req={row['requests']}", flush=True)

        rows.append(row)
        with (args.out / "cascade.jsonl").open("a") as fh:
            fh.write(json.dumps(row) + "\n")

    wall = time.time() - t0
    verdicts: dict[str, int] = {}
    for r in rows:
        verdicts[r["verdict"]] = verdicts.get(r["verdict"], 0) + 1
    by_detector: dict[str, int] = {}
    for r in rows:
        for d in r["detect"]:
            by_detector[d] = by_detector.get(d, 0) + 1
    capped = sum(1 for r in rows if r.get("truncated_at_solve"))

    lines = [
        "# The local cascade, end to end (2026-10-18)",
        "",
        f"`solve -> detect -> repair -> human gate` over {len(rows)} items against the live 176B on "
        f"`{args.endpoint}`; drafts {'reused from the bench' if args.reuse_drafts else 'generated'}, "
        f"{args.rounds} repair round(s), human gate in `{args.human}` mode.",
        "",
        "```sh",
        ".venv/bin/python experiments/cascade/cascade.py \\",
        f"    --bench {args.bench} \\",
        f"    --out {args.out}" + (" --reuse-drafts" if args.reuse_drafts else "")
        + (f" --items {args.items}" if args.items else "")
        + (f" --temperature {args.temperature}" if args.temperature else "") + " --score",
        "```",
        "",
        "What may run inside the loop is fixed by `bench_registry.json`'s runtime column: for a "
        "gold-dependent bench the runtime checks are the declared gold-free fallbacks, `consistency` "
        "always and `code_exec` when the task supplies tests.  `runtime_spec()` strips `expect` "
        "before a spec enters the loop, so no stage can read the answer key.",
        "",
        "## Verdicts",
        "",
        "| verdict | n |",
        "| --- | --- |",
    ]
    for v, c in sorted(verdicts.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {v} | {c} |")
    lines += [
        "",
        "Detections fired: " + (", ".join(f"`{d}` x{c}" for d, c in sorted(by_detector.items()))
                                 or "none"),
        f"Requests: {requests} ({wall:.0f}s wall); truncated at solve, then escalated not answered: "
        f"{capped}.",
        "",
    ]
    if gate_log:
        lines += ["## The human gate", "",
                  "First trigger as the human would have seen it, then the compact form for the "
                  "rest (auto mode records them instead of asking).", "", "```"]
        lines += gate_log
        lines += ["```", ""]

    if args.score:
        per_item = {r["item_id"]: score_item(specs.get(r["task_id"]), r.get("artifact")) for r in rows}
        lines += ["## Scoring (post-loop, gold-based, eval-only)", "",
                  "| verdict | n | content-correct artifact |", "| --- | --- | --- |"]
        for v in sorted(verdicts):
            ids = [r["item_id"] for r in rows if r["verdict"] == v]
            ok = sum(1 for i in ids if per_item[i]["content"] is True)
            lines.append(f"| {v} | {len(ids)} | {ok} |")
        silent = [i for i, s in per_item.items() if s["content"] is False
                  and next(r["verdict"] for r in rows if r["item_id"] == i) == "accepted"]
        repaired = [r["item_id"] for r in rows if r["verdict"] == "accepted_repaired"]
        repaired_ok = [i for i in repaired if per_item[i]["content"] is True]
        final_ok = sum(1 for s in per_item.values() if s["content"] is True)
        lines += [
            "",
            f"- Accepted artifacts that are wrong (the detectors' blind spot, not the gate's): "
            f"{len(silent)} ({', '.join(silent) or '-'}).",
            f"- Repaired and accepted: {len(repaired)}, correct: {len(repaired_ok)} "
            f"({', '.join(repaired_ok) or '-'}).",
            f"- Items the cascade would serve correctly, had everything been as benign as this "
            f"bench: {final_ok}/{len(per_item)}.",
            "",
        ]
        (args.out / "scoring.json").write_text(json.dumps(
            {"per_item": per_item, "verdicts": verdicts, "silent_accepts": silent,
             "repaired_correct": repaired_ok}, indent=1))

    (args.out / "CASCADE.md").write_text("\n".join(lines) + "\n")
    print(f"\nwall {wall:.0f}s, requests {requests}, verdicts {verdicts}\nwrote {args.out}/CASCADE.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
