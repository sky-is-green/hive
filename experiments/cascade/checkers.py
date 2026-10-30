"""Task checkers and JSON extraction — shared by the cascade experiments.

Kept free of ``harness`` imports on purpose: the eval scripts run under the
ROCm venv (for the local decision models), which does not carry the sidecar's
web dependencies.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import unicodedata
from typing import Optional


def _normalize(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text.lower())
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]+", " ", folded).strip()


def _last_number(text: str) -> Optional[float]:
    matches = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def _code_block(text: str) -> Optional[str]:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    if blocks:
        return blocks[-1].strip()
    return None


def _boxed(text: str) -> Optional[str]:
    """Content of the last ``\\boxed{...}`` (balanced braces), or ``None``."""
    idx = text.rfind("\\boxed{")
    if idx < 0:
        return None
    depth = 0
    out: list[str] = []
    for char in text[idx + len("\\boxed{") :]:
        if char == "{":
            depth += 1
        elif char == "}":
            if depth == 0:
                return "".join(out).strip()
            depth -= 1
        out.append(char)
    return None


def _norm_math(text: str) -> str:
    """Latex-ish normalization for exact-string math comparison."""
    value = text.strip().strip("$").strip()
    for source, target in (
        ("\\left", ""), ("\\right", ""), ("\\dfrac", "\\frac"),
        ("\\tfrac", "\\frac"), ("\\,", ""), ("\\!", ""), ("\\;", ""),
        ("\\ ", ""), ("~", ""), (" ", ""), ("%", ""), ("\\%", ""),
        ("{,}", ""), (",", ""),
    ):
        value = value.replace(source, target)
    value = re.sub(r"\\text\{([^{}]*)\}", r"\1", value)
    return value


def _math_equal(candidate: str, expect: str) -> bool:
    """Normalized-string, numeric, then sympy equality (best effort)."""
    left, right = _norm_math(candidate), _norm_math(expect)
    if left == right:
        return True
    try:
        return abs(float(left) - float(right)) <= 1e-6 * max(1.0, abs(float(right)))
    except ValueError:
        pass
    try:  # optional sympy fallback: 1/2, (x+1), a^2 ...
        import sympy

        lx = _norm_math(candidate).replace("^", "**")
        rx = _norm_math(expect).replace("^", "**")
        lx = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", lx)
        rx = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", rx)
        return bool(sympy.simplify(sympy.sympify(lx) - sympy.sympify(rx)) == 0)
    except Exception:  # noqa: BLE001 - unparseable stays unequal
        return False


def check(task: dict, answer: str) -> bool:
    """Apply the task's checker to an answer string."""
    if not answer or not answer.strip():
        return False
    spec = task["checker"]
    kind = spec["type"]
    if kind == "contains":
        normalized = _normalize(answer)
        return any(_normalize(e) in normalized for e in spec["expect"])
    if kind == "number":
        markers = re.findall(
            r"final answer[^0-9\-+]*([-+]?\d[\d,]*(?:\.\d+)?)", answer, re.I | re.S
        )
        got = (
            float(markers[-1].replace(",", ""))
            if markers
            else _last_number(answer)
        )
        if got is None:
            return False
        return abs(got - float(spec["expect"])) <= float(spec.get("tol", 1e-6))
    if kind == "choice":
        want = str(spec["expect"]).upper()
        text = answer.upper()
        # Prefer an explicit "answer: B" / "answer is B" / "choose B" phrase;
        # verbose models often append unrelated prose after the answer.
        for pattern in (
            r"\b(?:ANSWER|CHOOSE|OPTION|CHOICE|TIER)\s*(?:IS|:|\=)?\s*([A-Z])\b",
            r"^\s*([A-Z])\b",
        ):
            found = re.findall(pattern, text)
            if found:
                return found[-1] == want
        letters = re.findall(r"\b([A-Z])\b", text)
        return bool(letters) and letters[-1] == want
    if kind == "boxed":
        got = _boxed(answer)
        if got is None:
            return False
        # An exact value followed by an approximation is still that value:
        # \boxed{\frac{40}{243}\approx 0.1646} must compare as 40/243.
        got = re.split(r"\\approx|≈", got)[0].strip()
        return _math_equal(got, str(spec["expect"]))
    if kind == "schedule":
        obj = extract_json(answer)
        if not isinstance(obj, dict):
            return False
        schedule = obj.get("schedule")
        if not isinstance(schedule, list):
            return False
        specs = spec.get("tasks") or {}
        workers = {str(w) for w in (spec.get("workers") or [])}
        worker_skills = {
            str(w): {str(s) for s in skills}
            for w, skills in (spec.get("worker_skills") or {}).items()
        }
        task_skills = {str(t): str(s) for t, s in (spec.get("task_skills") or {}).items()}
        releases = {str(t): float(v) for t, v in (spec.get("release") or {}).items()}
        task_deadlines = {str(t): float(v) for t, v in (spec.get("task_deadline") or {}).items()}
        rows: dict[str, tuple[str, float, float]] = {}
        for item in schedule:
            if not isinstance(item, dict):
                return False
            task = str(item.get("task", ""))
            worker = str(item.get("worker", ""))
            if task not in specs or task in rows or worker not in workers:
                return False
            if task_skills.get(task) and task_skills[task] not in worker_skills.get(worker, set()):
                return False
            try:
                start = float(item.get("start"))
            except (TypeError, ValueError):
                return False
            duration = float(specs[task].get("duration", 0))
            if start < 0 or start < releases.get(task, 0.0):
                return False
            if task in task_deadlines and start + duration > task_deadlines[task] + 1e-9:
                return False
            rows[task] = (worker, start, start + duration)
        if set(rows) != set(specs):
            return False
        for before, after in spec.get("dependencies", []):
            if before not in rows or after not in rows:
                return False
            if rows[before][2] > rows[after][1] + 1e-9:
                return False
        for worker in workers:
            intervals = sorted((r[1], r[2]) for r in rows.values() if r[0] == worker)
            for (s1, e1), (s2, e2) in zip(intervals, intervals[1:]):
                if e1 > s2 + 1e-9:
                    return False
        makespan = max(r[2] for r in rows.values()) if rows else 0.0
        return makespan <= float(spec.get("deadline", makespan)) + 1e-9
    if kind == "ordered":
        obj = extract_json(answer)
        if not isinstance(obj, dict):
            return False
        order = obj.get(spec.get("field", "order"))
        if not isinstance(order, list):
            return False
        position = {str(item): index for index, item in enumerate(order)}
        required = spec.get("required_items") or []
        if any(str(item) not in position for item in required):
            return False
        for before, after in spec.get("before", []):
            if before not in position or after not in position:
                return False
            if position[before] >= position[after]:
                return False
        return True
    if kind == "assign":
        # Feasibility checker: any assignment that fits is correct. The role
        # suite's exact-set checker is kept for the benchmark, but a wider
        # suite must not reject a valid alternative assignment.
        obj = extract_json(answer)
        if not isinstance(obj, dict):
            return False
        rows = obj.get("assignments")
        if not isinstance(rows, list):
            return False
        task_specs = spec.get("tasks") or {}
        worker_caps = spec.get("workers") or {}
        if not isinstance(task_specs, dict) or not isinstance(worker_caps, dict):
            return False
        durations = {
            str(name): float(value.get("duration", value) if isinstance(value, dict) else value)
            for name, value in task_specs.items()
        }
        capacities = {str(name): float(value) for name, value in worker_caps.items()}
        task_skills = {str(k): str(v) for k, v in (spec.get("task_skills") or {}).items()}
        worker_skills = {
            str(k): {str(s) for s in v}
            for k, v in (spec.get("worker_skills") or {}).items()
        }
        seen: set[str] = set()
        loads = {name: 0.0 for name in capacities}
        for item in rows:
            if not isinstance(item, dict):
                return False
            task = str(item.get("task", ""))
            worker = str(item.get("worker", ""))
            if task not in durations or task in seen or worker not in capacities:
                return False
            if task_skills.get(task) and task_skills[task] not in worker_skills.get(worker, set()):
                return False
            seen.add(task)
            loads[worker] += durations[task]
        if seen != set(durations):
            return False
        return all(loads[w] <= capacities[w] + 1e-9 for w in capacities)
    if kind == "json":
        obj = extract_json(answer)
        if not isinstance(obj, dict):
            return False
        for key in spec.get("required", []):
            if key not in obj:
                return False
        for key, minimum in (spec.get("min_items") or {}).items():
            value = obj.get(key)
            if not isinstance(value, list) or len(value) < int(minimum):
                return False
        for key, needle in (spec.get("contains") or {}).items():
            if _normalize(str(needle)) not in _normalize(str(obj.get(key, ""))):
                return False
        for key, expected in (spec.get("equals") or {}).items():
            if obj.get(key) != expected:
                return False
        for key, expected in (spec.get("equals_sets") or {}).items():
            got = obj.get(key)
            if not isinstance(got, list) or set(map(str, got)) != set(map(str, expected)):
                return False
        expected = spec.get("expected_assignments")
        if expected:
            got = {
                (str(item.get("task", "")), str(item.get("worker", "")))
                for item in (obj.get("assignments") or [])
                if isinstance(item, dict)
            }
            want = {(str(item["task"]), str(item["worker"])) for item in expected}
            if got != want:
                return False
        return True
    if kind == "code":
        code = _code_block(answer)
        if code is None:
            return False
        tests = spec["tests"]
        if isinstance(tests, list):
            tests = "\n".join(tests)
        program = (
            str(spec.get("prelude", ""))
            + code
            + "\n\n"
            + tests
            + "\n"
            + str(spec.get("footer", "print('PASS')"))
            + "\n"
        )
        # /tmp/opencode was a local-box path; the pod does not have it.  COMMUNITY_TMPDIR can override, the
        # system temp directory is the default (created if missing).
        tmpdir = os.environ.get("COMMUNITY_TMPDIR") or tempfile.gettempdir()
        os.makedirs(tmpdir, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", suffix=".py", delete=False, dir=tmpdir
        ) as fh:
            fh.write(program)
            path = fh.name
        try:
            proc = subprocess.run(
                [sys.executable, path], capture_output=True, text=True, timeout=15
            )
            return proc.returncode == 0 and "PASS" in proc.stdout
        except subprocess.TimeoutExpired:
            return False
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
    raise ValueError(f"unknown checker {kind!r}")


def extract_json(text: str) -> Optional[dict]:
    """First top-level JSON object in a text (fences and prose tolerated)."""
    if not text:
        return None
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except ValueError:
        pass
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None
