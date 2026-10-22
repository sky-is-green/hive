"""The checker, execution first: run the answer, do not read it.

The verifier is gold-free.  It accepts the runtime spec only -- the answer
frame and the task's own tests -- and never sees ``expect``.  Two executable
methods ship here:

* ``code_exec``   the candidate's last fenced block against the task's tests,
                  in a bwrap sandbox: no network, a read-only root, a temp
                  cwd, a 5 s timeout and resource limits.  No in-process
                  ``eval`` anywhere.
* ``arithmetic``  the equations the answer asserts about itself, including
                  prose equations and unit conversions (cents/dollars,
                  inches/feet), evaluated with an AST walker.  A contradiction
                  is a concrete finding; a sentence with no checkable equation
                  is ``unknown``, not ``pass``.

Every ``pass`` must survive a mutation of its own claim.  If corrupting the
equation, or the code's returns, still passes, the verdict is ``unknown``.
A checker whose mutation survives is not a checker.

Grounding is a separate, weaker method: every factual claim needs a citation.
An uncited claim is ``unknown``, never ``fail``.  It is not used by the
execution measurement and does not affect the 28-task verdict.
"""

from __future__ import annotations

import ast
import os
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from fractions import Fraction
from typing import Any

CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)
FINAL_NUMBER = re.compile(
    r"final answer[^0-9\-+]*([-+]?\d[\d,]*(?:\.\d+)?)", re.I | re.S
)
CITATION = re.compile(r"\[\d+\]|https?://|\(source\s*:", re.I)

MONEY = {
    "cent": Fraction(1, 100), "cents": Fraction(1, 100), "¢": Fraction(1, 100),
    "dollar": Fraction(1), "dollars": Fraction(1), "usd": Fraction(1), "$": Fraction(1),
}
LENGTH = {
    "inch": 1, "inches": 1, "in": 1, "foot": 12, "feet": 12, "ft": 12,
    "yard": 36, "yards": 36, "yd": 36, "mile": 63360, "miles": 63360,
}
TIME = {
    "second": 1, "seconds": 1, "sec": 1, "secs": 1,
    "minute": 60, "minutes": 60, "min": 60, "mins": 60,
    "hour": 3600, "hours": 3600, "hr": 3600, "hrs": 3600,
}
UNIT_WORDS = set(MONEY) | set(LENGTH) | set(TIME) | {
    "bird", "birds", "box", "boxes", "item", "items", "log", "logs",
    "plank", "planks", "session", "sessions", "week", "weeks", "day",
    "days", "year", "years", "old", "token", "tokens", "point", "points",
    "person", "people", "card", "cards", "coupon", "coupons", "unit",
    "units",
}

SANDBOX_TIMEOUT = 5.0
SANDBOX_MEMORY = 512 << 20


# ---------------------------------------------------------------- frame

def code_block(text: str) -> str | None:
    blocks = CODE_BLOCK.findall(text or "")
    return blocks[-1].strip() if blocks else None


def boxed(text: str) -> str | None:
    idx = (text or "").rfind("\\boxed{")
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


def frame(kind: str | None, answer: str) -> dict[str, Any]:
    """The declared answer interface, without the answer key."""
    if kind == "code":
        code = code_block(answer)
        return {"ok": code is not None, "kind": kind, "code": code,
                "reason": "" if code else "no python code block"}
    if kind == "number":
        match = FINAL_NUMBER.search(answer or "")
        return {"ok": match is not None, "kind": kind,
                "value": match.group(1).replace(",", "") if match else None,
                "reason": "" if match else "no 'Final answer:' value"}
    if kind == "boxed":
        value = boxed(answer)
        return {"ok": value is not None, "kind": kind, "value": value,
                "reason": "" if value is not None else "no \\boxed{} value"}
    if kind == "choice":
        letters = re.findall(r"\b([A-E])\b", (answer or "").upper())
        return {"ok": bool(letters), "kind": kind,
                "value": letters[-1] if letters else None,
                "reason": "" if letters else "no choice letter"}
    return {"ok": False, "kind": kind, "reason": f"unknown frame {kind!r}"}


# ---------------------------------------------------------------- arithmetic

def _clean_math(text: str) -> str:
    text = (text.replace("\u00a0", " ").replace("−", "-").replace("×", "*")
            .replace("÷", "/").replace("·", "*"))
    text = text.replace("**", " ").replace("__", " ").replace("`", " ")
    text = re.sub(r"\\left|\\right", "", text)
    text = re.sub(r"\\[!,;]", "", text)
    text = text.replace("\\,", "").replace("\\ ", " ")
    text = re.sub(r"\\dfrac|\\tfrac", r"\\frac", text)
    text = re.sub(r"\\cdot|\\times", "*", text)
    text = re.sub(r"\\div", "/", text)
    text = re.sub(r"\\text\{([^{}]*)\}", r" \1 ", text)
    return text.replace("✓", " ").replace("✔", " ").replace("…", "...")


def _side(raw: str) -> tuple[str, str | None, bool] | None:
    """(safe expression, unit, unit-embedded-in-rate) for one side."""
    value = raw.strip()
    value = re.sub(r"^[\s:=\-–—•*]+", "", value)
    value = re.sub(r"^\d+[.)]\s+", "", value)
    value = value.rstrip(" .;:,")
    value = re.sub(r"\\left|\\right", "", value)
    value = re.sub(r"\\dfrac|\\tfrac", r"\\frac", value)
    value = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", value)

    unit: str | None = None
    value = value.replace("$$", " ")
    if "$" in value:
        unit = "$"
        value = value.replace("$", " ")
    for word in ("cents", "cent", "dollars", "dollar", "USD", "usd", "¢"):
        if re.search(r"(?<![A-Za-z])" + re.escape(word) + r"(?![A-Za-z])", value, re.I):
            unit = word.lower().replace("¢", "cent").rstrip("s")
            value = re.sub(
                r"(?<![A-Za-z])" + re.escape(word) + r"(?![A-Za-z])",
                " ", value, flags=re.I,
            )
            break

    words = re.findall(r"[A-Za-z]{2,}", value)
    if any(word.lower() not in UNIT_WORDS for word in words):
        return None
    for word in words:
        lowered = word.lower()
        if unit is None and (lowered in MONEY or lowered in LENGTH or lowered in TIME):
            unit = lowered
    value = re.sub(r"(?<![A-Za-z])[A-Za-z]{2,}(?![A-Za-z])", " ", value)
    value = value.replace("^", "**").replace(",", "")
    value = re.sub(r"\s+", " ", value).strip()
    if not value or not re.search(r"\d", value):
        return None
    if re.search(r"[A-Za-z\\]", value) or "..." in value or "≈" in value:
        return None
    embedded = bool(unit) and bool(re.search(r"[+\-*/]", raw))
    return value, unit, embedded


def _evaluate(expression: str) -> Fraction | None:
    """Exact AST arithmetic only; no eval, no names, no calls."""
    if len(expression) > 200:
        return None
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return None
    nodes = 0

    def walk(node: ast.AST) -> Fraction:
        nonlocal nodes
        nodes += 1
        if nodes > 100:
            raise ValueError("expression too large")
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if (isinstance(node, ast.Constant) and isinstance(node.value, (int, float))
                and not isinstance(node.value, bool)):
            return Fraction(str(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = walk(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left, right = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                if right.denominator != 1 or right < 0 or right > 8:
                    raise ValueError("exponent out of range")
                return left ** int(right)
            raise ValueError("unsupported operator")
        raise ValueError("unsupported expression")

    try:
        return walk(tree)
    except Exception:  # noqa: BLE001 - unparseable stays uncheckable
        return None


def _basis(value: Fraction, unit: str | None) -> tuple[Fraction, str] | None:
    if unit is None:
        return None
    unit = unit.lower()
    if unit in MONEY:
        return value * MONEY[unit], "money"
    if unit in LENGTH:
        return value * LENGTH[unit], "length"
    if unit in TIME:
        return value * TIME[unit], "time"
    return None


def find_equations(text: str) -> list[dict[str, Any]]:
    """Every self-asserted arithmetic equation we can evaluate safely."""
    out: list[dict[str, Any]] = []
    for line in _clean_math(text or "").splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
        if not line:
            continue
        positions = [match.start() for match in re.finditer("=", line)]
        for index, position in enumerate(positions):
            left_start = 0 if index == 0 else positions[index - 1] + 1
            segment = line[left_start:position]
            cut = max(segment.rfind(":"), segment.rfind(";"))
            if cut >= 0:
                left_start += cut + 1
            right_end = positions[index + 1] if index + 1 < len(positions) else len(line)
            left_raw = line[left_start:position].strip()
            right_raw = line[position + 1:right_end].strip()
            left, right = _side(left_raw), _side(right_raw)
            if not left or not right:
                continue
            left_expr, left_unit, left_embedded = left
            right_expr, right_unit, right_embedded = right
            left_value, right_value = _evaluate(left_expr), _evaluate(right_expr)
            if left_value is None or right_value is None:
                continue
            left_basis, right_basis = _basis(left_value, left_unit), _basis(right_value, right_unit)
            if left_basis is not None and right_basis is not None:
                if left_basis[1] != right_basis[1]:
                    continue
                ok = left_basis[0] == right_basis[0]
            elif (left_unit is None) != (right_unit is None):
                if (left_unit is None and right_embedded) or (right_unit is None and left_embedded):
                    continue
                if left_unit in LENGTH or right_unit in LENGTH:
                    known = (left_value * LENGTH[left_unit] if left_unit in LENGTH
                             else right_value * LENGTH[right_unit])
                    other = right_value if left_unit in LENGTH else left_value
                    ok = known == other
                else:
                    ok = left_value == right_value
            elif left_unit == right_unit or (left_unit is None and right_unit is None):
                ok = left_value == right_value
            else:
                continue
            out.append({
                "equation": f"{left_raw} = {right_raw}",
                "lhs_text": left_raw, "rhs_text": right_raw,
                "lhs": left_value, "rhs": right_value,
                "left_unit": left_unit, "right_unit": right_unit,
                "ok": ok,
            })
    return out


def _mutated_equation(equation: dict[str, Any]) -> str | None:
    """Corrupt the checked claim itself, preserving its units and frame.

    Appending ``+ 1`` after a unit phrase (``18 years old``) can make the
    mutation unparseable, which would silently pass the self-test.  Replace a
    numeric literal in the right side instead, so the same equation is re-read
    with a different value.
    """
    numbers = list(re.finditer(r"\d+(?:\.\d+)?", equation["rhs_text"]))
    if not numbers:
        return None
    match = numbers[-1]
    replacement = str(int(equation["rhs"]) + 1) if equation["rhs"].denominator == 1 else f"{float(equation['rhs']) + 1:g}"
    mutated_rhs = equation["rhs_text"][: match.start()] + replacement + equation["rhs_text"][match.end() :]
    return f"{equation['lhs_text']} = {mutated_rhs}"


# ---------------------------------------------------------------- code sandbox

def _limits() -> None:
    resource.setrlimit(resource.RLIMIT_AS, (SANDBOX_MEMORY, SANDBOX_MEMORY))
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))


def run_sandboxed(program: str, *, timeout: float = SANDBOX_TIMEOUT) -> dict[str, Any]:
    """Run a Python program under bwrap.  Returns rc/stdout/stderr/seconds.

    ``available`` is false when bwrap is absent; the caller must then answer
    ``unknown`` rather than run untrusted code in this process.
    """
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        return {"available": False, "rc": None, "stdout": "", "stderr": "",
                "seconds": 0.0, "timeout": False}
    path = ""
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as handle:
            handle.write(program)
            path = handle.name
        command = [
            bwrap, "--unshare-net", "--die-with-parent",
            "--ro-bind", "/", "/", "--dev", "/dev",
            "--tmpfs", "/tmp", "--ro-bind", path, "/tmp/work.py",
            "--chdir", "/tmp",
            "--setenv", "PYTHONNOUSERSITE", "1", "--setenv", "HOME", "/tmp",
            sys.executable, "-I", "-S", "/tmp/work.py",
        ]
        started = time.perf_counter()
        try:
            proc = subprocess.run(
                command, capture_output=True, text=True, timeout=timeout,
                preexec_fn=_limits,
            )
            return {
                "available": True, "rc": proc.returncode,
                "stdout": proc.stdout, "stderr": proc.stderr,
                "seconds": time.perf_counter() - started, "timeout": False,
            }
        except subprocess.TimeoutExpired:
            return {"available": True, "rc": None, "stdout": "", "stderr": "",
                    "seconds": time.perf_counter() - started, "timeout": True}
    finally:
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass


def _program(spec: dict[str, Any], code: str) -> str:
    tests = spec.get("tests", "")
    if isinstance(tests, list):
        tests = "\n".join(str(item) for item in tests)
    return (
        str(spec.get("prelude", "")) + code + "\n\n" + str(tests) + "\n"
        + str(spec.get("footer", "print('PASS')")) + "\n"
    )


def mutate_code(code: str) -> str | None:
    """Corrupt the last function's returns, or a numeric literal."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if not functions:
        return None
    target = functions[-1]
    returns = [node for node in ast.walk(target) if isinstance(node, ast.Return)]
    if returns:
        for node in returns:
            node.value = ast.Constant(value=None)
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)
    numbers = [node for node in ast.walk(target)
               if isinstance(node, ast.Constant) and isinstance(node.value, (int, float))
               and not isinstance(node.value, bool)]
    if numbers:
        numbers[-1].value = numbers[-1].value + 1
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)
    return None


# ---------------------------------------------------------------- grounding

def _claims(answer: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+|\n+", answer or "")
    return [sentence.strip() for sentence in sentences
            if len(sentence.split()) >= 5
            and (re.search(r"\d", sentence) or re.search(r"\b[A-Z][a-z]{2,}", sentence))]


def check_grounding(answer: str) -> dict[str, Any]:
    """Citation coverage.  An uncited factual claim is unknown, never fail."""
    claims = _claims(answer)
    if not claims:
        return {"verdict": "unknown", "claims": 0, "uncited": [],
                "reason": "no decomposable factual claims"}
    uncited = [claim for claim in claims if not CITATION.search(claim)]
    if uncited:
        return {"verdict": "unknown", "claims": len(claims),
                "uncited": uncited[:5], "reason": "uncited claims"}
    return {"verdict": "pass", "claims": len(claims), "uncited": [],
            "reason": "every claim carries a citation"}


# ---------------------------------------------------------------- public verify

def runtime_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Copy only the frame and the task's tests; the answer key never enters."""
    checker = spec.get("checker", spec)
    out: dict[str, Any] = {"kind": checker.get("type", checker.get("kind"))}
    for key in ("tests", "prelude", "footer"):
        if key in checker:
            out[key] = checker[key]
    return out


def verify(spec: dict[str, Any], answer: str, *, grounding: bool = False) -> dict[str, Any]:
    """A gold-free verdict with evidence, mutation result and latency."""
    started = time.perf_counter()
    public = runtime_spec(spec)
    kind = public.get("kind")
    result: dict[str, Any] = {
        "verdict": "unknown",
        "method": "execution",
        "check": "frame",
        "evidence": "",
        "mutated_ok": None,
        "latency_ms": 0.0,
        "kind": kind,
        "detail": {},
        "grounding": {"verdict": "not_applicable", "claims": 0, "uncited": []},
    }

    if kind == "code":
        declared = frame(kind, answer)
        result["detail"]["frame"] = declared
        code = declared.get("code")
        if not code:
            result.update(check="frame", evidence=declared["reason"])
        else:
            run = run_sandboxed(_program(public, code))
            result["detail"]["sandbox"] = {
                "available": run["available"], "rc": run["rc"],
                "seconds": round(run["seconds"], 3), "timeout": run["timeout"],
            }
            if not run["available"]:
                result.update(check="code_exec", evidence="bwrap unavailable")
            elif run["timeout"]:
                result.update(check="code_exec", verdict="fail",
                              evidence="the candidate's code did not finish in 5 s")
            elif run["rc"] == 0 and "PASS" in run["stdout"]:
                mutated = mutate_code(code)
                if mutated is None:
                    result.update(check="code_exec",
                                  evidence="tests pass, but no code mutation is available")
                else:
                    second = run_sandboxed(_program(public, mutated))
                    caught = (not second["available"] or second["timeout"]
                              or second["rc"] != 0 or "PASS" not in second["stdout"])
                    result["detail"]["mutation"] = {
                        "available": second["available"], "rc": second["rc"],
                        "timeout": second["timeout"],
                    }
                    if second["available"] and caught:
                        result.update(check="code_exec", verdict="pass",
                                      mutated_ok=True,
                                      evidence="tests pass; corrupting the returns fails them")
                    else:
                        result.update(check="code_exec", mutated_ok=False,
                                      evidence="tests pass, but the mutation still passes")
            else:
                tail = (run["stderr"] or run["stdout"] or "tests failed").strip()
                result.update(check="code_exec", verdict="fail", evidence=tail[-900:])
    elif kind in ("number", "boxed"):
        declared = frame(kind, answer)
        result["detail"]["frame"] = declared
        equations = find_equations(answer)
        bad = [equation for equation in equations if not equation["ok"]]
        result["detail"]["equations"] = {
            "checked": len(equations),
            "bad": [equation["equation"] for equation in bad[:3]],
        }
        if bad:
            first = bad[0]
            result.update(
                check="arithmetic", verdict="fail",
                evidence=(f"{first['equation']} is false: "
                          f"{first['lhs']} != {first['rhs']}"),
            )
        elif not equations:
            result.update(check="arithmetic",
                          evidence="no checkable arithmetic equation in the answer")
        elif not declared["ok"]:
            result.update(check="frame", evidence=declared["reason"])
        else:
            mutation = _mutated_equation(equations[0])
            mutated_bad = ([equation for equation in find_equations(mutation)
                            if not equation["ok"]] if mutation else [])
            if mutated_bad:
                result.update(
                    check="arithmetic", verdict="pass", mutated_ok=True,
                    evidence=(f"{len(equations)} equation(s) check; "
                              "corrupting the first one fails it"),
                )
            else:
                result.update(check="arithmetic", mutated_ok=False,
                              evidence="equations check, but the mutation still passes")
    elif kind == "choice":
        declared = frame(kind, answer)
        result["detail"]["frame"] = declared
        result.update(check="frame",
                      evidence=("choice is well formed; correctness is not executable"
                                if declared["ok"] else declared["reason"]))
    else:
        result.update(check="none", evidence=f"unknown kind {kind!r}")

    if grounding and result["verdict"] == "unknown":
        grounded = check_grounding(answer)
        if grounded["verdict"] == "pass":
            citation = CITATION.search(answer)
            if citation is None:
                grounded = {**grounded, "verdict": "unknown", "mutated_ok": False,
                            "reason": "no citation available to mutate"}
            else:
                mutated = answer[: citation.start()] + answer[citation.end() :]
                mutated_grounding = check_grounding(mutated)
                caught = mutated_grounding["verdict"] != "pass"
                grounded = {**grounded, "mutated_ok": caught,
                            "mutation_verdict": mutated_grounding["verdict"]}
                if not caught:
                    grounded["verdict"] = "unknown"
                    grounded["reason"] = "the removal of a citation still passes"
        result["grounding"] = grounded
        if grounded["verdict"] == "pass":
            result.update(check="grounding", verdict="pass", method="grounding",
                          mutated_ok=grounded.get("mutated_ok"),
                          evidence="every factual claim carries a citation; "
                                   "removing one fails the check")
    result["latency_ms"] = round((time.perf_counter() - started) * 1000, 3)
    return result
