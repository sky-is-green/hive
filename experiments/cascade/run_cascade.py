#!/usr/bin/env python
"""Cascade runner — local generation + validation, optional API fallback.

The Strata-era default: the local endpoint answers every task (no
pre-generation router), a local validator checks the answer, and the API is
**off unless** ``--api-fallback`` names an OpenAI-compatible endpoint — then a
rejected answer escalates there.  Tiny-Jev is the default validator
(``--judge tiny-jev``, ~40 ms/check); ``--judge none`` skips validation and
``--judge local`` uses Intern-Decision-4B/JEV.

``--judge deterministic`` is the gold-free stack the cascade line measured:
the arithmetic ``consistency`` verifier plus ``code_exec`` against the task's
own tests, both reported as findings instead of a probability.  With it,
``--repair`` (default ``auto``) re-asks the local model once, greedily, with
the finding quoted, before the human gate or the API — so the pipeline is
generate -> detect -> repair -> escalate.

The measurement configuration the benchmark used is still available:
``--router api --judge api --router-mode measure`` records the frontier
router's decision and an API comparison answer for every task, which is what
threshold fitting consumes.

Per task the runner records the local answer and outcome, the validation
verdict, any fallback answer, and the checker labels.

Artifacts: ``<out>/<run-name>/report.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import requests

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from harness.cascade import (  # noqa: E402
    Judgement,
    JudgementBroker,
    Policy,
    RouteDecision,
    TelemetryLog,
    default_registry,
    plan_request,
    result_for,
)
from harness.cascade.registry import CandidateRegistry  # noqa: E402
from harness.cascade.policy import verifier_rejects  # noqa: E402
from harness.cascade.learned_gate import (  # noqa: E402
    DEFAULT_HEAD,
    KNOWN_HEADS,
    LearnedGate,
)

from experiments.cascade.checkers import check, extract_json  # noqa: E402
from experiments.cascade.cascade import (  # noqa: E402 - the gold-free runtime guard
    canonical,
    detect,
    runtime_spec,
)
from experiments.cascade.escalation_lane import (  # noqa: E402
    EscalationLane,
    in_lane,
    lane_record,
)
from experiments.cascade.repair_pass import (  # noqa: E402
    DRAFT_HEAD,
    evidence_prompt,
    generic_prompt,
)

# ---------------------------------------------------------------- providers

SIDECAR = os.environ.get("CASCADE_SIDECAR", "http://127.0.0.1:8765")
SCION_PORT = int(os.environ.get("CASCADE_SCION_PORT", "1234"))
SCION_BASE = os.environ.get("CASCADE_SCION_BASE", f"http://127.0.0.1:{SCION_PORT}/v1")
API_URL = os.environ.get(
    "CASCADE_API_URL", "https://opencode.ai/zen/go/v1/chat/completions"
)
API_MODEL = os.environ.get("CASCADE_API_MODEL", "deepseek-v4.1-flash")
AUTH_PATH = Path.home() / ".local" / "share" / "opencode" / "auth.json"
PRICING = {"input": 0.15 / 1e6, "output": 0.60 / 1e6}  # USD per token

SCION_SYSTEM = (
    "You are a precise assistant. Think if you need to, then give the final "
    "answer. For code questions return only the requested Python function in a "
    "single code block."
)
ANSWER_SYSTEM = (
    "You are a precise assistant. Answer the user's task. For code questions "
    "return only the requested Python function in a single code block."
)
ROUTER_SYSTEM = (
    "You route requests in a local-first cascade. LOCAL is a 35B-parameter "
    "ternary-quantized mixture-of-experts model running on consumer hardware: "
    "fast and cheap, but weak on multi-step reasoning and precise coding. API "
    "is a frontier model: slower and paid, much stronger on those tasks. "
    'Decide whether LOCAL can plausibly answer the task correctly without '
    'tools. Reply with JSON only: {"route":"local"|"api","confidence":0-1,'
    '"reason":"<short>"}'
)
VERIFIER_SYSTEM = (
    "You are a strict answer verifier. You receive a TASK and a CANDIDATE "
    "answer. Decide whether the candidate is correct and complete; check the "
    "work, do not just trust its confidence. Reply with JSON only: "
    '{"verdict":"accept"|"reject","confidence":0-1,"reason":"<short>"}'
)


class Usage:
    """Token + cost accounting, split by stage tag."""

    def __init__(self) -> None:
        self.by_tag: dict[str, dict[str, float]] = {}

    def add(self, tag: str, usage: Optional[dict]) -> None:
        if not usage:
            return
        row = self.by_tag.setdefault(tag, {"calls": 0, "input": 0.0, "output": 0.0})
        row["calls"] += 1
        row["input"] += float(usage.get("prompt_tokens") or 0)
        row["output"] += float(usage.get("completion_tokens") or 0)

    def cost(self, tag: Optional[str] = None) -> float:
        rows = [self.by_tag[tag]] if tag else list(self.by_tag.values())
        return sum(
            r["input"] * PRICING["input"] + r["output"] * PRICING["output"] for r in rows
        )

    def summary(self) -> dict[str, Any]:
        return {
            tag: {**row, "cost_usd": round(self.cost(tag), 6)}
            for tag, row in sorted(self.by_tag.items())
        } | {"total_cost_usd": round(self.cost(), 6)}


def _post(url: str, payload: dict, headers: dict, timeout: int) -> tuple[dict, float]:
    t0 = time.time()
    resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    elapsed = (time.time() - t0) * 1000.0
    if resp.status_code >= 400:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
    return resp.json(), elapsed


def _api_key() -> str:
    data = json.loads(AUTH_PATH.read_text(encoding="utf-8"))
    return data["opencode-go"]["key"]


class Frontier:
    """An OpenAI-compatible paid tier: retries, usage accounting.

    The measurement roles (router / verifier / comparison answer) use the
    configured API (``CASCADE_API_*``).  A user's fallback endpoint is the same
    class with ``base_url``/``model``/``key`` overridden and the ``fallback``
    tag, so its spend lands on its own usage line.
    """

    def __init__(
        self,
        usage: Usage,
        session: Optional[str] = None,
        *,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        key: Optional[str] = None,
    ) -> None:
        self.usage = usage
        self.session = session or f"cascade-pilot-{uuid.uuid4().hex[:12]}"
        self.base_url = base_url or API_URL
        self.model = model or API_MODEL
        # A gateway (LiteLLM) in front of the provider takes its own bearer
        # token; fall back to the OpenCode auth store when calling direct.  A
        # user-supplied fallback endpoint may need no key at all.
        if key:
            self.key = key
        elif os.environ.get("CASCADE_API_KEY"):
            self.key = os.environ["CASCADE_API_KEY"]
        elif base_url:
            self.key = "none"
        else:
            self.key = _api_key()
        # Per-role virtual keys: route / verify / escalate each get their own
        # spend line at the gateway.  The call tag names the role.
        self.role_keys = {
            "router": os.environ.get("CASCADE_ROUTER_KEY", ""),
            "verifier": os.environ.get("CASCADE_VERIFIER_KEY", ""),
            "answer": os.environ.get("CASCADE_ESCALATION_KEY", ""),
        }

    def chat(self, messages: list[dict], *, tag: str, max_tokens: int) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        headers = {
            "authorization": f"Bearer {self.role_keys.get(tag) or self.key}",
            "x-opencode-session": self.session,
            "content-type": "application/json",
            "user-agent": "curl/8.9.1",
        }
        last: Optional[Exception] = None
        budget = max_tokens
        for attempt in range(3):
            try:
                data, ms = _post(self.base_url, payload | {"max_tokens": budget}, headers, timeout=180)
                # count usage only for a response we can actually use: a
                # retried call would otherwise double-count the tokens
                choice = data["choices"][0]
                self.usage.add(tag, data.get("usage"))
                msg = choice["message"]
                content = msg.get("content") or ""
                finish = choice.get("finish_reason")
                time.sleep(0.2)
                if content.strip() or finish != "length":
                    return {
                        "content": content,
                        "reasoning": msg.get("reasoning_content") or "",
                        "latency_ms": ms,
                        "finish_reason": finish,
                        "tag": tag,
                    }
                budget *= 2  # reasoning consumed the whole budget: retry bigger
            except Exception as exc:  # noqa: BLE001 - retry then surface
                last = exc
                time.sleep(1.5 * (attempt + 1))
        if last is not None:
            raise RuntimeError(f"frontier call failed ({tag}): {last}")
        return {"content": "", "reasoning": "", "latency_ms": 0.0,
                "finish_reason": "length", "tag": tag}


class HumanGate:
    """The last checkpoint before a paid API call.

    The cascade has already generated a local answer, the verifier has
    rejected it, and the automatic repair round (if configured) could not
    clear the finding.  Instead of silently escalating, show the failure to
    the person at the terminal and offer exactly two outcomes:

    * ``solve`` — provide a corrected answer, or a hint for one local retry.
      A ``keep`` shortcut is accepted only when the task's own checker already
      passes (i.e. the verifier is the thing that is wrong).
    * ``escalate`` — send the task to the configured API.  The human's reason
      is recorded and included as context in the API prompt.

    The first trigger is deliberately verbose: the task, the local answer, the
    verifier's verdict and reason, the checker result, and the API target are
    all printed.  Later triggers are compact.  ``input_fn``/``print_fn`` are
    injectable so the prompt can be tested without a terminal.
    """

    def __init__(
        self,
        *,
        api_target: str,
        api_model: str,
        input_fn=input,
        print_fn=print,
        max_answer_chars: int = 2400,
    ) -> None:
        self.api_target = api_target
        self.api_model = api_model
        self._input = input_fn
        self._print = print_fn
        self.max_answer_chars = max_answer_chars
        self.asked = 0

    # -- rendering -----------------------------------------------------

    def _checker_label(self, checker_ok: Optional[bool]) -> str:
        if checker_ok is True:
            return "PASSED (the verifier may be a false reject)"
        if checker_ok is False:
            return "FAILED"
        return "not available"

    def _show(
        self,
        *,
        task: dict[str, Any],
        local_answer: str,
        verdict: Any,
        checker_ok: Optional[bool],
        first: bool,
        still_failing: bool,
        repair: Optional[dict[str, Any]] = None,
        attempts: int = 0,
    ) -> None:
        print_ = self._print
        print_("=" * 78)
        if first:
            print_("HUMAN GATE — local answer rejected; the next step is the hosted API")
            print_("This is the first trigger. Local generation and the verifier both ran;")
            print_("the cascade would have called the API automatically. You choose now:")
            print_("  [s]olve locally  — correct the answer, or give the model a hint")
            print_("  [e]scalate to API — pay for the frontier answer")
            print_("-" * 78)
        else:
            print_("HUMAN GATE — still unresolved before the API call")
        print_(f"task:   {task.get('id', '?')}  bucket={task.get('bucket', '?')}")
        prompt = str(task.get("prompt", "") or "")
        if prompt:
            if len(prompt) > 1600:
                prompt = prompt[:1600] + "\n... [truncated]"
            print_("task prompt:")
            print_(textwrap.indent(prompt, "  "))
        if attempts > 1 or still_failing:
            print_(f"attempt: {attempts}  (the previous human answer did not pass the checker)")
        reason = ""
        try:
            reason = str((verdict.payload or {}).get("reason", ""))
        except AttributeError:
            reason = ""
        body = local_answer or ""
        if len(body) > self.max_answer_chars:
            body = body[: self.max_answer_chars] + "\n... [truncated; full answer is in the report]"
        print_("local answer:")
        print_(textwrap.indent(body or "(empty)", "  "))
        if repair and first:
            attempt = str(repair.get("text", "") or "")
            if len(attempt) > self.max_answer_chars:
                attempt = attempt[: self.max_answer_chars] + (
                    "\n... [truncated; full attempt is in the report]"
                )
            finding = str(repair.get("detail", "") or repair.get("finding", ""))
            print_("automatic repair: one re-attempt with the finding quoted"
                   + (f" ({finding})" if finding else ""))
            if repair.get("still_flagged"):
                print_(f"repair still flagged by: {', '.join(repair['still_flagged'])}")
            print_("repair attempt:")
            print_(textwrap.indent(attempt or "(empty)", "  "))
        print_(
            f"verifier: {getattr(verdict, 'decision', '?')} "
            f"(confidence {float(getattr(verdict, 'confidence', 0.0) or 0.0):.3f})"
        )
        if reason:
            print_(f"verifier reason: {reason}")
        print_(f"checker:  {self._checker_label(checker_ok)}")
        print_(f"API:      {self.api_target}  model={self.api_model}")
        print_("The API call is paid and logged; the local path is free.")
        print_("-" * 78)

    # -- prompt --------------------------------------------------------

    def ask(
        self,
        *,
        task: dict[str, Any],
        local_answer: str,
        verdict: Any,
        checker_ok: Optional[bool],
        still_failing: bool = False,
        repair: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Show the evidence and return one human decision.

        Returns a dict with ``choice`` (``solve``/``escalate``), ``mode``
        (``answer``/``hint``/``keep``/``escalate``), and the ``answer``,
        ``hint``, and ``reason`` fields.  Raises ``KeyboardInterrupt`` if the
        human aborts; EOF is treated as escalation so a misconfigured
        non-interactive run cannot hang forever.
        """
        self.asked += 1
        first = self.asked == 1
        self._show(
            task=task,
            local_answer=local_answer,
            verdict=verdict,
            checker_ok=checker_ok,
            first=first,
            still_failing=still_failing,
            repair=repair,
            attempts=self.asked,
        )

        while True:
            try:
                choice = self._input("Choice [s]olve / [e]scalate: ").strip().lower()
            except EOFError:
                self._print("No interactive input; escalating to the API.")
                return {
                    "choice": "escalate",
                    "mode": "escalate",
                    "answer": "",
                    "hint": "",
                    "reason": "human gate reached EOF",
                }

            if choice in ("s", "solve", "1"):
                prompt = (
                    "Paste a corrected answer, or a hint prefixed 'hint:' "
                    "(blank = back"
                )
                if checker_ok is True:
                    prompt += ", 'keep' = keep the local answer"
                prompt += "): "
                try:
                    answer = self._input(prompt).strip()
                except EOFError:
                    answer = ""
                if not answer:
                    continue
                if checker_ok is True and answer.lower() in ("keep", "k"):
                    return {
                        "choice": "solve",
                        "mode": "keep",
                        "answer": "",
                        "hint": "",
                        "reason": "checker already passes",
                    }
                if answer.lower().startswith("hint:"):
                    hint = answer[len("hint:") :].strip()
                    if not hint:
                        self._print("The hint was empty; try again.")
                        continue
                    try:
                        why = self._input(
                            "What was missing? (optional; Enter for none): "
                        ).strip()
                    except EOFError:
                        why = ""
                    return {
                        "choice": "solve",
                        "mode": "hint",
                        "answer": "",
                        "hint": hint,
                        "reason": why,
                    }
                try:
                    why = self._input(
                        "What was wrong / what did you change? (optional; Enter for none): "
                    ).strip()
                except EOFError:
                    why = ""
                return {
                    "choice": "solve",
                    "mode": "answer",
                    "answer": answer,
                    "hint": "",
                    "reason": why,
                }

            if choice in ("e", "escalate", "2"):
                try:
                    why = self._input(
                        "Why can't local solve this? (optional; Enter for none): "
                    ).strip()
                except EOFError:
                    why = ""
                return {
                    "choice": "escalate",
                    "mode": "escalate",
                    "answer": "",
                    "hint": "",
                    "reason": why,
                }

            self._print("Please answer 's' to solve locally or 'e' to escalate to the API.")


class ScionTier:
    """The local tier: the llama-server hivebench spawned for the stack."""

    def __init__(self, base: str = SCION_BASE, *, max_tokens: int = 768,
                 temperature: float = 0.0,
                 reasoning_effort: str = "",
                 reasoning_budget_tokens: int = 0) -> None:
        self.base = base.rstrip("/")
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.reasoning_effort = reasoning_effort
        self.reasoning_budget_tokens = int(reasoning_budget_tokens)

    def alive(self) -> bool:
        try:
            r = requests.get(f"{self.base}/models", timeout=5)
            return r.status_code == 200
        except Exception:  # noqa: BLE001
            return False

    def _recover(self) -> None:
        """End a wedged engine so the next request starts a fresh one (#267)."""
        root = self.base[: -len("/v1")] if self.base.endswith("/v1") else self.base
        try:
            requests.post(f"{root}/unload", timeout=60)
        except Exception:  # noqa: BLE001 - the retried request is the real test
            pass
        time.sleep(2)

    def _post_local(self, payload: dict, *, timeout: float) -> tuple[dict, float]:
        """POST the local chat call, surviving engine wedges and restarts.

        Strata answers ``#267`` ("an earlier window never finished on the
        GPU; restart the engine") as an HTTP 400 and restarts its engine on
        the next request after a crash; a long benchmark must not lose a
        label (or the run) to either.  Genuine client errors are re-raised at
        once.
        """
        last: Optional[Exception] = None
        for attempt in range(4):
            try:
                return _post(
                    f"{self.base}/chat/completions", payload,
                    {"content-type": "application/json"}, timeout=timeout,
                )
            except Exception as exc:  # noqa: BLE001 - retry engine failures
                last = exc
                text = str(exc)
                recoverable = (
                    "#267" in text
                    or "restart the engine" in text
                    or "engine stopped unexpectedly" in text
                    or not text.startswith("HTTP 4")
                )
                if not recoverable or attempt == 3:
                    raise
                print(f"[cascade] local call failed ({exc}); recovering ...",
                      file=sys.stderr, flush=True)
                self._recover()
        raise last if last else RuntimeError("local call failed")

    def chat(self, messages: list[dict], *, max_tokens: Optional[int] = None,
             temperature: Optional[float] = None) -> dict:
        budget = int(max_tokens or self.max_tokens)
        payload = {
            "model": "scion",
            "messages": messages,
            "temperature": self.temperature if temperature is None else float(temperature),
            "max_tokens": budget,
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        if self.reasoning_budget_tokens:
            payload["reasoning_budget_tokens"] = self.reasoning_budget_tokens
        data, ms = self._post_local(payload, timeout=900)
        choice = data["choices"][0]
        msg = choice["message"]
        content = msg.get("content") or ""
        usage = data.get("usage") or {}
        finish = choice.get("finish_reason")
        # A reasoning model that ran out of budget before answering: retry once.
        if not content.strip() and int(usage.get("completion_tokens") or 0) >= budget:
            payload["max_tokens"] = max(2048, budget * 2)
            data, ms = self._post_local(payload, timeout=1200)
            choice = data["choices"][0]
            msg = choice["message"]
            content = msg.get("content") or ""
            usage = data.get("usage") or {}
            finish = choice.get("finish_reason")
        return {
            "content": content,
            "reasoning": msg.get("reasoning_content") or "",
            "latency_ms": ms,
            "finish_reason": finish,
            "usage": usage,
        }


# ---------------------------------------------------------------- pipeline


def answer_prompt(
    task: dict,
    *,
    hint: str = "",
    reviewer_note: str = "",
) -> list[dict]:
    messages = [
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user", "content": task["prompt"]},
    ]
    if hint:
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Reviewer hint: {hint}\n"
                    "Use this hint to correct the answer to the task above."
                ),
            }
        )
    if reviewer_note:
        messages.append(
            {
                "role": "user",
                "content": (
                    "Human reviewer note (why the local tier could not solve "
                    f"this): {reviewer_note}"
                ),
            }
        )
    return messages


def record_human_reason(stats: dict[str, Any], reason: str) -> None:
    """Keep a compact frequency table of why humans solved/escalated."""
    reason = (reason or "").strip()
    if not reason:
        return
    key = reason[:160]
    stats["reasons"][key] = int(stats["reasons"].get(key, 0)) + 1


# ---------------------------------------------------------------- D3 + repair

def deterministic_verdict(judgement: Judgement, task: dict, candidate: str, rs: dict):
    """The gold-free D3 verdict: reject when a detector fires, accept otherwise.

    ``rs`` is the task's runtime spec (see ``cascade.runtime_spec``): the
    answer key is not in it, so neither this verdict nor anything downstream
    can read gold.
    """
    t0 = time.time()
    findings = detect(candidate, rs)
    ms = (time.time() - t0) * 1000.0
    return result_for(
        judgement,
        "reject" if findings else "accept",
        0.0 if findings else 1.0,
        latency_ms=ms,
        payload={
            "judge": "deterministic",
            "detectors": [f["detector"] for f in findings],
            "findings": findings,
            "reason": finding_digest(findings),
        },
    )


def finding_digest(findings: list[dict[str, Any]]) -> str:
    """A one-line, human-readable digest of the detectors' findings."""
    parts: list[str] = []
    for f in findings:
        detector = str(f.get("detector", "?"))
        if detector == "consistency":
            equations = f.get("finding") or []
            first = equations[0] if equations else {}
            parts.append(
                f"consistency: {first.get('equation', 'an asserted step')} is false"
            )
        elif detector == "learned_gate":
            d = f.get("detail") or {}
            parts.append(
                f"learned gate: score {float(d.get('p', 0.0)):.3f} below the frozen "
                f"threshold {float(d.get('threshold', 0.5)):.2f} — doubtful"
            )
        else:
            text = " ".join(str(f.get("finding", "")).split())
            parts.append(f"{detector}: {text[:240] or 'flagged'}")
    return "; ".join(parts)


def gate_finding(p: float, ms: float, threshold: float, head: str) -> dict[str, Any]:
    """The learned gate's rejection finding (gold-free: question+candidate only)."""
    return {
        "detector": "learned_gate",
        "finding": (
            f"the frozen self-check scored this answer's correctness confidence "
            f"at {p:.3f}, below the {threshold:.2f} gate threshold; its "
            f"arithmetic and the task's own checks found no definite failure"
        ),
        "detail": {"p": p, "ms": ms, "threshold": threshold, "head": head},
    }


def _new_d3_judgement(task_id: str) -> Judgement:
    return Judgement(id=f"verify-{task_id}", request_id=task_id, generation_id=0,
                     role="D3", kind="verify")


def apply_learned_gate(
    gate: LearnedGate,
    task_id: str,
    task: dict,
    candidate: str,
    verdict: Any,
    threshold: float,
) -> tuple[Any, dict[str, Any]]:
    """Blind-spot-only learned gate: score a silent-draft, fire below threshold.

    Called for a deterministic verdict only, and only when the detectors are
    silent (the verdict accepts): that is the measured blind spot.  A fired
    gate is one more rejection finding — findings, digest and ladder all
    behave exactly like a detector's.
    """
    if getattr(verdict, "decision", "") == "reject":
        return verdict, {"scored": False, "fired": None}
    p, ms = gate.score(str(task.get("prompt", "")), candidate)
    head = getattr(getattr(gate, "head", None), "name", "")
    old_payload = dict(verdict.payload or {})
    row: dict[str, Any] = {"scored": True, "fired": p < threshold, "p": p,
                           "ms": ms, "head": head, "threshold": threshold}
    gate_payload = {"p": p, "ms": ms, "threshold": threshold, "head": head}
    latency = float(getattr(verdict, "latency_ms", 0.0) or 0.0) + ms
    if p >= threshold:
        # silent: the verdict is unchanged in meaning (same decision and
        # confidence) — but re-issued so the D3 row's latency and payload
        # include the score honestly.
        fresh = result_for(
            _new_d3_judgement(task_id), verdict.decision, verdict.confidence,
            latency_ms=latency, payload={**old_payload, "gate": gate_payload},
        )
        return fresh, row
    findings = list(old_payload.get("findings") or [])
    findings.append(gate_finding(p, ms, threshold, head))
    fresh = result_for(
        _new_d3_judgement(task_id), "reject", 0.0,
        latency_ms=latency,
        payload={
            "judge": old_payload.get("judge") or "deterministic",
            "detectors": [f["detector"] for f in findings],
            "findings": findings,
            "reason": finding_digest(findings),
            "gate": gate_payload,
        },
    )
    return fresh, row


def gate_repair_prompt(row: dict, finding: dict[str, Any]) -> str:
    """The learned-gate repair prompt: p vs threshold quoted, nothing else.

    No gold, no checker output — the finding is the score itself and the ask
    is an internal-inconsistency recheck, the failure mode the blind spot is
    made of (a wrong answer the detectors could not see).
    """
    d = finding.get("detail") or {}
    detail = (
        f"An internal consistency self-check scored this draft's correctness "
        f"confidence at {float(d.get('p', 0.0)):.3f}, below the "
        f"{float(d.get('threshold', 0.5)):.2f} gate threshold, while the "
        f"arithmetic and the task's own checks found no definite failure. "
        f"Reread the reasoning line by line looking for an internal "
        f"inconsistency (a step that contradicts another step, or a final "
        f"answer the steps do not actually support), redo the work "
        f"correctly, and give the final answer in the same format."
    )
    return row["prompt"].rstrip() + "\n\n" + DRAFT_HEAD.format(row["candidate"].strip()) + detail


def repair_user_text(task: dict, candidate: str, findings: list[dict[str, Any]]
                     ) -> tuple[str, str]:
    """The repair prompt: the finding is quoted, never the expected answer.

    With structured findings (``--judge deterministic``) the measured
    evidence prompt is used; a learned-gate rejection (the deterministic
    stack silent by construction) gets its own template; without one it
    falls back to the generic recheck prompt.
    """
    row = {"prompt": str(task.get("prompt", "")), "candidate": candidate or ""}
    if findings:
        if all(f.get("detector") == "learned_gate" for f in findings):
            return "learned-gate", gate_repair_prompt(row, findings[0])
        found = {f["detector"]: f["finding"] for f in findings}
        return evidence_prompt(row, found)
    return "generic", generic_prompt(row)


def attempt_repair(
    scion: "ScionTier",
    task: dict,
    candidate: str,
    findings: list[dict[str, Any]],
    rs: dict,
    *,
    max_tokens: Optional[int] = None,
    rescore: Optional[Any] = None,
) -> dict[str, Any]:
    """One greedy local re-attempt with the finding quoted, then re-detect.

    Mirrors the measured repair arm: one round, temperature 0, the finding
    quoted and never the expected answer; a truncated or byte-identical
    attempt is never accepted as the answer; an attempt that clears the
    detectors is re-framed by ``canonical()`` for the task's interface.

    ``rescore`` is the learned gate's strict re-verify: when a gate finding
    is the reason the draft was rejected, a repaired answer that clears the
    detectors is re-scored and the finding clears only if the score clears
    (a persistent flag stays flagged).  It is only called when the detectors
    are silent on the repair — the only case where the gate is the remaining
    objection.
    """
    detail, user_text = repair_user_text(task, candidate, findings)
    messages = [
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user", "content": user_text},
    ]
    t0 = time.time()
    generations = 1
    out = scion.chat(messages, max_tokens=max_tokens, temperature=0.0)
    if out.get("finish_reason") == "length":
        budget = max(2048, int(max_tokens or 0) * 2)
        out = scion.chat(messages, max_tokens=budget, temperature=0.0)
        generations = 2
    text = out.get("content") or ""
    after = detect(text, rs)
    record: dict[str, Any] = {
        "detail": detail,
        "finding": finding_digest(findings),
        "ms": round((time.time() - t0) * 1000.0, 1),
        "completion_tokens": int((out.get("usage") or {}).get("completion_tokens") or 0),
        "chars": len(text),
        "generations": generations,
        "finish_reason": out.get("finish_reason"),
        "same_as_draft": text.strip() == (candidate or "").strip(),
        "still_flagged": [f["detector"] for f in after],
        "truncated": out.get("finish_reason") == "length",
        "canonical": canonical(rs.get("kind"), text) if not after else None,
        "cleared": False,
        "gate_rescored": False,
        "gate_p2": None,
        "gate_cleared": None,
        "text": text,
    }
    if rescore is not None and not after and not record["truncated"]:
        scored = rescore(text)
        record["gate_rescored"] = True
        record["gate_p2"] = scored["p"]
        record["gate_cleared"] = scored["p"] >= scored["threshold"]
    if not after and not record["truncated"] and record["canonical"]:
        if record["gate_rescored"] and not record["gate_cleared"]:
            record["stopped"] = (
                f"the learned gate still scores the repair below threshold "
                f"(p={record['gate_p2']:.3f})"
            )
        else:
            record["cleared"] = True
    if not after and not record["cleared"] and "stopped" not in record:
        record["stopped"] = (
            "the repair was truncated" if record["truncated"]
            else "the repair produced no consumable final answer"
        )
    return record


def truncation_verdict(verdict: Any, task_id: str, judge: str, budget: int):
    """Fold the harness's own truncation finding into the verdict.

    A generation that hit the token cap even after the doubled retry has no
    final answer to accept, so the verdict is a reject whatever the detectors
    said — the truncation rule, in the runner.
    """
    findings = [
        *((verdict.payload or {}).get("findings") or []),
        {"detector": "truncated",
         "finding": f"still cut off at {budget} tokens; there is no final answer to accept"},
    ]
    return result_for(
        Judgement(id=f"verify-{task_id}", request_id=task_id, generation_id=0,
                  role="D3", kind="verify"),
        "reject", 0.0,
        latency_ms=float(getattr(verdict, "latency_ms", 0.0) or 0.0),
        payload={
            "judge": judge,
            "detectors": [f["detector"] for f in findings],
            "findings": findings,
            "reason": finding_digest(findings),
        },
    )


def repair_enabled(args: argparse.Namespace) -> bool:
    """The repair leg runs when asked, or by default with the deterministic judge."""
    return args.repair == "on" or (
        args.repair == "auto" and args.judge == "deterministic"
    )


def open_gate(args: argparse.Namespace, err=sys.stderr) -> Optional[LearnedGate]:
    """The learned gate behind ``--gate`` — or None, with the reason printed.

    The gate is the deterministic stack's third trigger: it is meaningless
    without ``--judge deterministic``.  ``scion`` (path A) is a future source
    that needs the serve restart (M5 of LEARNED-GATE-PLAN.md); a failed head
    load turns the optional trigger off rather than losing the run.
    """
    if args.gate == "off":
        return None
    if args.gate == "scion":
        print("[cascade] --gate scion (path A) is not wired yet — it needs the "
              "serve /embedding restart (M5 of LEARNED-GATE-PLAN.md); gate off",
              file=err)
        return None
    if args.judge != "deterministic":
        print("[cascade] --gate needs --judge deterministic (the gate is the "
              "deterministic stack's third trigger); gate off", file=err)
        return None
    try:
        gate = LearnedGate(head=args.gate_head)
    except Exception as exc:  # noqa: BLE001 - an optional trigger must not kill the run
        print(f"[cascade] --gate disabled (head load failed: {exc})", file=err)
        return None
    print(f"[cascade] learned gate on: head {args.gate_head} at "
          f"{args.gate_threshold:.2f} (frozen r2 heads; the 0.8B backbone "
          "loads on the first silent-draft score)", file=err, flush=True)
    return gate


def route_prompt(task: dict) -> list[dict]:
    return [
        {"role": "system", "content": ROUTER_SYSTEM},
        {"role": "user", "content": task["prompt"]},
    ]


def verify_prompt(task: dict, candidate: str) -> list[dict]:
    return [
        {"role": "system", "content": VERIFIER_SYSTEM},
        {"role": "user", "content": f"TASK:\n{task['prompt']}\n\nCANDIDATE:\n{candidate}"},
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="first N tasks only")
    parser.add_argument("--tasks", default=str(Path(__file__).with_name("tasks.json")))
    parser.add_argument("--split", default="",
                        help="only tasks with this 'split' (bench sets; empty = all)")
    parser.add_argument("--benchmark", default="",
                        help="only tasks with this 'benchmark' (bench sets)")
    parser.add_argument(
        "--out", default=str(REPO / "experiments" / "cascade" / "runs"),
        help="artifact root",
    )
    parser.add_argument(
        "--run-name", default="",
        help="explicit run directory name under --out (the API pre-allocates it)",
    )
    parser.add_argument("--scion-base", default=SCION_BASE)
    parser.add_argument("--local-max-tokens", type=int, default=768,
                        help="local answer budget; a reasoning-only answer retries doubled")
    parser.add_argument("--local-temperature", type=float, default=0.0,
                        help="local sampling temperature (0 = greedy, the serving "
                             "default); 0.7 reproduces the sampled cascade regime")
    parser.add_argument("--local-reasoning-effort", default="",
                        choices=("", "none", "low", "medium", "high"),
                        help="local thinking level; empty = the engine default")
    parser.add_argument("--local-reasoning-budget-tokens", type=int, default=0,
                        help="hard local thinking cap (0 = none)")
    parser.add_argument(
        "--router", choices=("none", "jev", "api"), default="none",
        help="C1: none (default; always answer locally), local Tiny-Jev, or "
             "the frontier API prompt",
    )
    parser.add_argument(
        "--router-threshold", type=float, default=0.90,
        help="Tiny-Jev: route local when P(local answers correctly) >= this",
    )
    parser.add_argument("--router-model", default="lostargon/Tiny-Jev-1.7B")
    parser.add_argument(
        "--judge",
        choices=("none", "tiny-jev", "local", "api", "harness", "deterministic"),
        default="tiny-jev",
        help="D3 validation: none, local Tiny-Jev (default, in-process), local "
             "Intern-Decision-4B/JEV, the frontier API prompt, the harness "
             "validator service at --validator-url, or deterministic (the "
             "gold-free consistency + code_exec stack, with findings)",
    )
    parser.add_argument(
        "--repair", choices=("auto", "on", "off"), default="auto",
        help="after a verifier reject, re-ask the local model once, greedily, "
             "with the finding quoted, before the human gate or the API: auto "
             "(default) is on for --judge deterministic and off otherwise",
    )
    parser.add_argument(
        "--gate", choices=("off", "intern-0.8b", "scion"), default="off",
        help="the learned gate (a third trigger beside the deterministic "
             "detectors, blind-spot-only): intern-0.8b is the frozen r2 "
             "readout head over the intern-decision-0.8B CPU backbone; scion "
             "(path A, the serve's own embedding) is not wired yet",
    )
    parser.add_argument(
        "--gate-head", default=DEFAULT_HEAD, choices=KNOWN_HEADS,
        help=f"the frozen r2 head the gate applies (default {DEFAULT_HEAD}, "
             "the plan's precision-first choice; L0.5 is the recall dial)",
    )
    parser.add_argument(
        "--gate-threshold", type=float, default=0.5,
        help="fire the gate when the head's score falls below this (frozen "
             "at 0.5 by the plan; no per-run tuning)",
    )
    parser.add_argument("--judge-model", default="",
                        help="empty = the default model for the chosen judge")
    parser.add_argument("--judge-device", default="cpu",
                        help="cpu (default; safe beside the resident engine) or "
                             "cuda when the GPUs are free")
    parser.add_argument(
        "--validator-url",
        default=os.environ.get("CASCADE_VALIDATOR_URL",
                               "http://127.0.0.1:8765/v1/cascade/validate"),
        help="harness validator endpoint used by --judge harness",
    )
    parser.add_argument(
        "--judge-threshold", type=float, default=0.5,
        help="accept the candidate when P(correct) >= this",
    )
    parser.add_argument(
        "--router-mode", choices=("measure", "gate"), default="gate",
        help="gate (default): a router 'api' decision skips local generation "
             "and the verifier; measure: record the route but always generate "
             "locally and fetch the API comparison answer",
    )
    parser.add_argument(
        "--api-fallback", default="",
        help="OpenAI-compatible base URL used ONLY when validation rejects and "
             "no other API role is configured; empty (default) = API off",
    )
    parser.add_argument("--api-fallback-model", default="",
                        help="model name for the fallback (default: CASCADE_API_MODEL)")
    parser.add_argument("--api-fallback-key", default="",
                        help="bearer key for the fallback (default: CASCADE_API_KEY)")
    parser.add_argument(
        "--human-gate", choices=("auto", "on", "off"), default="auto",
        help="ask a human before the API fallback: auto (default) is on only "
             "when stdin is a TTY and --api-fallback is configured; on forces "
             "the prompt; off preserves the non-interactive benchmark flow",
    )
    parser.add_argument(
        "--no-escalation-lane", action="store_true",
        help="do not write escalation.jsonl next to the report (the harvest "
             "of every ladder-resolved item: task+findings+resolution; "
             "training-side only, the runtime never reads it)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    spec = json.loads(Path(args.tasks).read_text(encoding="utf-8"))
    tasks = [
        task for task in spec["tasks"]
        if (not args.split or task.get("split") == args.split)
        and (not args.benchmark or task.get("benchmark") == args.benchmark)
    ]
    if args.limit:
        tasks = tasks[: args.limit]
    if not tasks:
        print("no tasks selected", file=sys.stderr)
        return 2

    # The loop's runtime view of each task: `runtime_spec` drops the answer
    # key, so the deterministic verifier and the repair leg (and nothing they
    # call) can read it.
    runtime_specs: dict[str, dict[str, Any]] = {}
    for task in tasks:
        try:
            runtime_specs[task["id"]] = runtime_spec(task)
        except Exception as exc:  # noqa: BLE001 - a malformed checker is not a runtime input
            runtime_specs[task["id"]] = {"kind": None, "error": str(exc)[:200]}

    scion = ScionTier(
        args.scion_base,
        max_tokens=args.local_max_tokens,
        temperature=args.local_temperature,
        reasoning_effort=args.local_reasoning_effort,
        reasoning_budget_tokens=args.local_reasoning_budget_tokens,
    )
    if not scion.alive():
        print(
            f"local tier not reachable at {scion.base} — apply the "
            "'strata-125b' stack through hivebench "
            "(POST /v1/stacks/strata-125b/apply), start the server yourself, "
            "or point --scion-base at a running endpoint.",
            file=sys.stderr,
        )
        return 2

    usage = Usage()
    measure = args.router_mode == "measure"
    # The frontier (paid tier) is needed whenever a task can be answered by
    # the API: the measure mode always compares, an api router/judge always
    # uses it, and router-mode "gate" uses it for every router decision that
    # routes to the API (the default router-mode is "gate").
    needs_api = (measure or args.router == "api" or args.judge == "api"
                 or args.router_mode == "gate")
    frontier = Frontier(usage) if needs_api else None
    fallback = None
    if args.api_fallback:
        fallback = Frontier(
            usage,
            base_url=args.api_fallback,
            model=args.api_fallback_model or None,
            key=args.api_fallback_key or None,
        )
    human_gate: Optional[HumanGate] = None
    if args.human_gate == "on" or (
        args.human_gate == "auto" and sys.stdin.isatty()
    ):
        if fallback is None:
            if args.human_gate == "on":
                print(
                    "[cascade] --human-gate on needs --api-fallback; gate disabled",
                    file=sys.stderr,
                )
        elif measure:
            if args.human_gate == "on":
                print(
                    "[cascade] --human-gate is not used in measure mode; gate disabled",
                    file=sys.stderr,
                )
        else:
            human_gate = HumanGate(
                api_target=fallback.base_url,
                api_model=fallback.model,
            )

    human_stats: dict[str, Any] = {
        "asked": 0,
        "solve_attempts": 0,
        "solved": 0,
        "escalated": 0,
        "kept": 0,
        "first_triggered": False,
        "reasons": {},
    }

    telemetry = TelemetryLog()
    policy = Policy(
        accept_confidence=args.judge_threshold,
        human_gate=human_gate is not None,
    )
    broker = JudgementBroker()
    gate = open_gate(args)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out) / (args.run_name.strip() or stamp)
    out_dir.mkdir(parents=True, exist_ok=True)
    lane = (
        EscalationLane.disabled()
        if args.no_escalation_lane
        else EscalationLane(out_dir / "escalation.jsonl")
    )

    def router_handler(batch):
        results = []
        for j in batch:
            task = j.payload["task"]
            try:
                out = frontier.chat(route_prompt(task), tag="router", max_tokens=120)
                parsed = extract_json(out["content"]) or extract_json(out["reasoning"]) or {}
                route = str(parsed.get("route", "")).lower()
                if route not in ("local", "api"):
                    route = "api"
                results.append(
                    result_for(
                        j, route, float(parsed.get("confidence") or 0.0),
                        latency_ms=out["latency_ms"],
                        payload={"reason": parsed.get("reason", "")},
                    )
                )
            except Exception as exc:  # noqa: BLE001
                results.append(
                    result_for(j, "api", 0.0, payload={"error": str(exc)[:200]})
                )
        return results

    def verify_handler(batch):
        results = []
        for j in batch:
            task = j.payload["task"]
            try:
                out = frontier.chat(
                    verify_prompt(task, j.payload["candidate"]),
                    tag="verifier", max_tokens=400,
                )
                parsed = extract_json(out["content"]) or extract_json(out["reasoning"]) or {}
                verdict = str(parsed.get("verdict", "")).lower()
                if verdict not in ("accept", "reject"):
                    verdict = "reject"  # unparseable = do not trust
                results.append(
                    result_for(
                        j, verdict, float(parsed.get("confidence") or 0.0),
                        latency_ms=out["latency_ms"],
                        payload={"reason": parsed.get("reason", "")},
                    )
                )
            except Exception as exc:  # noqa: BLE001
                results.append(result_for(j, "reject", 0.0, payload={"error": str(exc)[:200]}))
        return results

    broker.register_handler("C1", router_handler)
    broker.register_handler("D3", verify_handler)

    if args.router == "jev":
        # Local C1: Tiny-Jev's `choice` primitive, one forward pass per task.
        from experiments.cascade.router_eval import TinyJev

        print(f"[cascade] loading local router {args.router_model} ...")
        jev = TinyJev(args.router_model)

        def jev_router_handler(batch):
            results = []
            for j in batch:
                task = j.payload["task"]
                state = {"task": task["prompt"], "bucket": task["bucket"]}
                t0 = time.time()
                with jev.torch.no_grad():
                    res = jev.model.choice(
                        jev.tok, state, "Which model should handle this task?",
                        {"local": "the local model answers correctly",
                         "api": "the API model is needed"},
                    )
                p_local = float((res.get("probabilities") or {}).get("local", 0.0))
                ms = (time.time() - t0) * 1000.0
                decision = "local" if p_local >= args.router_threshold else "api"
                results.append(
                    result_for(j, decision, p_local, latency_ms=ms,
                               payload={"p_local": p_local, "router": "tiny-jev"})
                )
            return results

        broker.register_handler("C1", jev_router_handler)

    verify_enabled = args.judge != "none"
    if args.judge in ("tiny-jev", "local") and args.judge_device != "cpu" and scion.alive():
        print(
            "[cascade] WARNING: a torch judge on the GPUs while the local engine "
            "is resident creates a second ROCm context; this host can hang "
            "(harness/models.py).  Use --judge-device cpu (the default) unless "
            "the GPUs are actually free.",
            file=sys.stderr, flush=True,
        )
    if args.judge == "harness":
        # The harness owns the validator: one service for any engine, on CPU.
        def harness_verify_handler(batch):
            results = []
            for j in batch:
                try:
                    resp = requests.post(
                        args.validator_url,
                        json={
                            "task": j.payload["task"],
                            "candidate": j.payload["candidate"],
                            "threshold": policy.accept_confidence,
                        },
                        timeout=180,
                    )
                    if resp.status_code >= 400:
                        raise RuntimeError(
                            f"HTTP {resp.status_code}: {resp.text[:200]}")
                    data = resp.json()
                    p = float(data["p_correct"])
                    accept = bool(data.get("accept", p >= policy.accept_confidence))
                    results.append(
                        result_for(j, "accept" if accept else "reject", p,
                                   latency_ms=float(data.get("ms") or 0.0),
                                   payload={"judge": "harness",
                                            "model": data.get("model")})
                    )
                except Exception as exc:  # noqa: BLE001
                    results.append(
                        result_for(j, "reject", 0.0, payload={"error": str(exc)[:200]})
                    )
            return results

        broker.register_handler("D3", harness_verify_handler)

    if args.judge == "tiny-jev":
        # Local D3, advisory: Tiny-Jev's `noul` head, one forward pass (~40 ms).
        from experiments.cascade.decision import TinyJevJudge

        model_id = args.judge_model or TinyJevJudge.DEFAULT_MODEL
        print(f"[cascade] loading local validator {model_id} ...")
        judge = TinyJevJudge(model_id, device=args.judge_device)

        def tiny_jev_verify_handler(batch):
            results = []
            for j in batch:
                try:
                    p, ms = judge.verdict(j.payload["task"], j.payload["candidate"])
                    decision = "accept" if p >= policy.accept_confidence else "reject"
                    results.append(
                        result_for(j, decision, p, latency_ms=ms,
                                   payload={"judge": model_id.split("/")[-1]})
                    )
                except Exception as exc:  # noqa: BLE001
                    results.append(
                        result_for(j, "reject", 0.0, payload={"error": str(exc)[:200]})
                    )
            return results

        broker.register_handler("D3", tiny_jev_verify_handler)

    if args.judge == "local":
        # Local D3: Intern-Decision-4B, one masked-next-token forward pass per
        # judgement (calibrated P(candidate is correct)).
        from experiments.cascade.decision import InternDecisionJudge

        model_id = args.judge_model or "internlm/Intern-Decision-4B"
        print(f"[cascade] loading local judge {model_id} ...")
        judge = InternDecisionJudge(model_id, device=args.judge_device)

        def local_verify_handler(batch):
            results = []
            for j in batch:
                try:
                    p, ms = judge.verdict(j.payload["task"], j.payload["candidate"])
                    decision = "accept" if p >= policy.accept_confidence else "reject"
                    results.append(
                        result_for(j, decision, p, latency_ms=ms,
                                   payload={"judge": model_id.split("/")[-1]})
                    )
                except Exception as exc:  # noqa: BLE001
                    results.append(
                        result_for(j, "reject", 0.0, payload={"error": str(exc)[:200]})
                    )
            return results

        broker.register_handler("D3", local_verify_handler)

    repair_on = repair_enabled(args)
    if args.judge == "deterministic":
        # D3: the measured gold-free stack — the arithmetic consistency
        # verifier plus the task's own tests (code_exec).  The finding is the
        # verdict's reason, which is what the repair leg quotes and the human
        # gate shows.
        def deterministic_verify_handler(batch):
            results = []
            for j in batch:
                task = j.payload["task"]
                rs = runtime_specs.get(task["id"]) or {"kind": None}
                try:
                    results.append(
                        deterministic_verdict(j, task, j.payload["candidate"], rs)
                    )
                except Exception as exc:  # noqa: BLE001 - do not trust on error
                    results.append(
                        result_for(j, "reject", 0.0, payload={"error": str(exc)[:200]})
                    )
            return results

        broker.register_handler("D3", deterministic_verify_handler)

    # -- phase A: router batch (skipped in the local-first default) -----
    routes: dict[str, Any] = {}
    if args.router != "none":
        print(f"[cascade] router phase over {len(tasks)} tasks ...")
        for task in tasks:
            broker.submit(
                Judgement(
                    id=f"route-{task['id']}", request_id=task["id"], generation_id=0,
                    role="C1", kind="route", payload={"task": task},
                )
            )
        broker.run_pending()
        routes = {r.request_id: r for r in broker.consume()}

    registry = default_registry()
    records: list[dict[str, Any]] = []
    try:
        for index, task in enumerate(tasks, 1):
            route_result = routes.get(task["id"])
            planned = plan_request(
                RouteDecision(task["bucket"], confidence=(route_result.confidence if route_result else 1.0)),
                out_len=256,
            )
            gated = (
                args.router_mode == "gate"
                and route_result is not None
                and route_result.decision == "api"
            )
            suffix = "  [gate->api]" if gated else ""
            print(f"[{index}/{len(tasks)}] {task['id']}  path={planned.path}{suffix}")

            if gated:
                # The router sent this to the API: skip local generation and the
                # verifier entirely; the API answer is the cascade answer.
                api_direct = frontier.chat(answer_prompt(task), tag="answer", max_tokens=700)
                api_ok = check(task, api_direct["content"])
                telemetry.log(
                    task["id"], planned.path, "E4-api", "api", api_direct["latency_ms"],
                    outcome="ok" if api_ok else "wrong", escalated=True,
                )
                records.append(
                    {
                        "id": task["id"], "bucket": task["bucket"],
                        "planned_path": planned.path,
                        "scion_answer": None, "scion_ok": None, "scion_ms": None,
                        "scion_tokens": 0,
                        "human_answer": None, "human_gate": None,
                        "human_corrected": False,
                        "router_route": route_result.decision,
                        "router_confidence": route_result.confidence,
                        "router_agrees": None,
                        "verdict": "gated", "verdict_confidence": None,
                        "verdict_reason": "router sent this task to the API",
                        "escalated": True, "gated": True,
                        "api_ok": api_ok, "api_answer": api_direct["content"][:4000],
                        "api_finish": api_direct.get("finish_reason"),
                        "api_ms": round(api_direct["latency_ms"], 1),
                        "api_direct_correct": api_ok,
                        "cascade_ok": api_ok,
                        "false_accept": False, "false_reject": False, "corrected": False,
                        "judge": None,
                        "gate": args.gate, "gate_scored": False, "gate_fired": None,
                        "gate_p": None, "gate_cleared": None,
                        "fallback_available": fallback is not None,
                        "flagged": False,
                    }
                )
                continue

            local = scion.chat(answer_prompt(task))
            truncated = local.get("finish_reason") == "length"
            if truncated:
                # The truncation rule: a cap hit is the harness's ceiling, not an
                # answer.  Retry once at twice the budget; if it is still cut off,
                # the verdict below rejects it and it is never served as-is.
                print(f"[cascade] {task['id']}: local answer hit the token cap; "
                      "retrying at double budget", file=sys.stderr, flush=True)
                local = scion.chat(answer_prompt(task),
                                   max_tokens=max(2048, args.local_max_tokens * 2))
                truncated = local.get("finish_reason") == "length"
            local_ok = check(task, local["content"])
            telemetry.log(
                task["id"], planned.path, "E4-local", "dgpu1", local["latency_ms"],
                tokens=int((local["usage"] or {}).get("completion_tokens") or 0),
                outcome="ok" if local_ok else "wrong",
                detail="truncated at 2x budget" if truncated else "",
            )

            if verify_enabled:
                broker.submit(
                    Judgement(
                        id=f"verify-{task['id']}", request_id=task["id"], generation_id=0,
                        role="D3", kind="verify",
                        payload={"task": task, "candidate": local["content"]},
                    )
                )
                broker.run_pending()
                verdicts = {r.request_id: r for r in broker.consume()}
                verdict = verdicts.get(task["id"])
                if verdict is None:
                    verdict = result_for(
                        Judgement(id=f"verify-{task['id']}", request_id=task["id"],
                                  generation_id=0, role="D3", kind="verify"),
                        "reject", 0.0, payload={"error": "no verdict"},
                    )
            else:
                verdict = result_for(
                    Judgement(id=f"verify-{task['id']}", request_id=task["id"],
                              generation_id=0, role="D3", kind="verify"),
                    "accept", 1.0, latency_ms=0.0, payload={"judge": "none"},
                )
            rejected = verifier_rejects(verdict, policy) if verify_enabled else False
            if truncated:
                verdict = truncation_verdict(
                    verdict, task["id"], args.judge,
                    max(2048, args.local_max_tokens * 2),
                )
                rejected = True

            # The learned gate, blind-spot-only: it scores a draft only when the
            # deterministic detectors are silent (the measured blind spot, where
            # the +1..+2 catch lives).  A fired gate is one more rejection
            # finding — findings, digest, repair and ladder all behave as for a
            # detector; a truncated draft is never scored (no answer to score).
            gate_row: Optional[dict[str, Any]] = None
            if gate is not None and not rejected and not truncated:
                try:
                    verdict, gate_row = apply_learned_gate(
                        gate, task["id"], task, local["content"], verdict,
                        args.gate_threshold,
                    )
                except Exception as exc:  # noqa: BLE001 - a failed score is not a verdict
                    gate_row = {"scored": False, "fired": None, "error": str(exc)[:200]}
                    print(f"[cascade] {task['id']}: gate scoring failed "
                          f"({gate_row['error']}); gate skipped", file=sys.stderr)
                if gate_row.get("fired"):
                    rejected = True
            telemetry.log(
                task["id"], planned.path, "D3-verify",
                "api" if args.judge == "api" else args.judge_device,
                verdict.latency_ms,
                outcome="ok" if not rejected else "reject", async_=True,
                detail=(
                    f"gate p={gate_row['p']:.4f}"
                    + (", fired" if gate_row.get("fired") else "")
                ) if gate_row and gate_row.get("scored") else "",
            )

            # The repair leg: one greedy local re-attempt with the finding quoted.
            # It runs before the gate, so the wrong answers it can fix never
            # reach the human or the API; what it cannot clear is what the gate
            # and the fallback are for.
            repair_record: Optional[dict[str, Any]] = None
            repaired_answer: Optional[str] = None
            if rejected and repair_on and not truncated:
                rescore = None
                if gate is not None and any(
                    f.get("detector") == "learned_gate"
                    for f in (verdict.payload or {}).get("findings") or []
                ):
                    def _rescore(candidate_text: str, _task=task):
                        p2, ms2 = gate.score(str(_task["prompt"]), candidate_text)
                        return {"p": p2, "ms": ms2, "threshold": args.gate_threshold}

                    rescore = _rescore
                repair_record = attempt_repair(
                    scion, task, local["content"],
                    list((verdict.payload or {}).get("findings") or []),
                    runtime_specs.get(task["id"]) or {"kind": None},
                    max_tokens=args.local_max_tokens,
                    rescore=rescore,
                )
                if repair_record["cleared"]:
                    repaired_answer = repair_record["canonical"]
                    rejected = False
                telemetry.log(
                    task["id"], planned.path, "D3-repair", "dgpu1",
                    repair_record["ms"],
                    outcome="ok" if repair_record["cleared"] else "flagged",
                    tokens=repair_record["completion_tokens"], async_=True,
                    detail=repair_record["finding"][:200],
                )

            # Escalate only when there is somewhere to escalate to.  With the API
            # fallback unset (the default), a reject is recorded and the local
            # answer is returned; measure mode uses the measurement API instead.
            escalate = rejected and (measure or fallback is not None)
            human_decision: Optional[dict[str, Any]] = None
            human_attempts = 0
            human_gate_first = False
            human_checked_answer: Optional[str] = None
            display_answer = local["content"]
            display_checker_ok: Optional[bool] = local_ok

            if escalate and human_gate is not None and fallback is not None:
                while True:
                    human_attempts += 1
                    was_first = not human_stats["first_triggered"]
                    human_decision = human_gate.ask(
                        task=task,
                        local_answer=display_answer,
                        verdict=verdict,
                        checker_ok=display_checker_ok,
                        still_failing=human_attempts > 1,
                        repair=repair_record,
                    )
                    human_stats["asked"] += 1
                    if was_first:
                        human_stats["first_triggered"] = True
                        human_gate_first = True
                    reason = str(human_decision.get("reason", "") or "")
                    record_human_reason(human_stats, reason)
                    telemetry.log(
                        task["id"], planned.path, "HUMAN", "human", 0.0,
                        outcome="ok" if human_decision["choice"] == "solve" else "escalate",
                        detail=(
                            f"mode={human_decision['mode']}"
                            + (f"; {reason}" if reason else "")
                        )[:300],
                    )

                    if human_decision["choice"] == "escalate":
                        human_stats["escalated"] += 1
                        break

                    human_stats["solve_attempts"] += 1
                    mode = human_decision["mode"]
                    if mode == "keep":
                        human_stats["kept"] += 1
                        human_stats["solved"] += 1
                        human_checked_answer = local["content"]
                        escalate = False
                        break
                    if mode == "hint":
                        print(
                            "[cascade] retrying local with the human hint ...",
                            file=sys.stderr, flush=True,
                        )
                        local = scion.chat(answer_prompt(task, hint=human_decision["hint"]))
                        local_ok = check(task, local["content"])
                        display_answer = local["content"]
                        display_checker_ok = local_ok
                        telemetry.log(
                            task["id"], planned.path, "E4-local-retry", "dgpu1",
                            local["latency_ms"],
                            tokens=int((local["usage"] or {}).get("completion_tokens") or 0),
                            outcome="ok" if local_ok else "wrong",
                            detail="human hint",
                        )
                        if local_ok:
                            human_stats["solved"] += 1
                            human_checked_answer = local["content"]
                            escalate = False
                            break
                        continue

                    # mode == "answer"
                    human_checked_answer = str(human_decision["answer"])
                    human_ok = check(task, human_checked_answer)
                    display_answer = human_checked_answer
                    display_checker_ok = human_ok
                    if human_ok:
                        human_stats["solved"] += 1
                        escalate = False
                        break
                    # The human's answer failed the checker; show the gate again.
                    continue

            api_direct = None
            if measure:
                api_direct = frontier.chat(answer_prompt(task), tag="answer", max_tokens=700)
            elif escalate:
                reviewer_note = (
                    str(human_decision.get("reason", "") or "")
                    if human_decision is not None else ""
                )
                api_direct = fallback.chat(
                    answer_prompt(task, reviewer_note=reviewer_note),
                    tag="fallback", max_tokens=700,
                )
            api_ok = check(task, api_direct["content"]) if api_direct is not None else None

            if human_decision is not None and not escalate:
                if human_decision["mode"] in ("keep", "hint"):
                    cascade_answer = local["content"]
                    cascade_ok = local_ok
                else:
                    cascade_answer = human_checked_answer or local["content"]
                    cascade_ok = check(task, cascade_answer)
            else:
                if escalate:
                    cascade_answer = api_direct["content"]
                    cascade_ok = api_ok
                elif repaired_answer is not None:
                    # the repair cleared the detectors: the canonical re-framing
                    # is the served answer (the draft was wrong)
                    cascade_answer = repaired_answer
                    cascade_ok = check(task, repaired_answer)
                else:
                    cascade_answer = local["content"]
                    cascade_ok = local_ok
            flagged = rejected and not measure and fallback is None

            if escalate:
                telemetry.log(
                    task["id"], planned.path,
                    "E4-api" if measure else "E4-fallback",
                    "api", api_direct["latency_ms"],
                    outcome="ok" if api_ok else "wrong", escalated=True,
                    detail=(
                        f"human reason: {human_decision.get('reason', '')}"
                        if human_decision is not None
                        and human_decision.get("choice") == "escalate"
                        else ""
                    )[:300],
                )

            human_record = None
            if human_decision is not None:
                human_record = {
                    "asked": True,
                    "first": human_gate_first,
                    "choice": human_decision["choice"],
                    "mode": human_decision["mode"],
                    "reason": str(human_decision.get("reason", "") or ""),
                    "hint": str(human_decision.get("hint", "") or ""),
                    "human_answer": human_checked_answer,
                    "attempts": human_attempts,
                    "accepted_locally": not escalate,
                    "api_target": (human_gate.api_target if human_gate else None),
                    "api_model": (human_gate.api_model if human_gate else None),
                }

            records.append(
                {
                    "id": task["id"],
                    "bucket": task["bucket"],
                    "planned_path": planned.path,
                    "scion_answer": local["content"][:4000],
                    "scion_ok": local_ok,
                    "scion_ms": round(local["latency_ms"], 1),
                    "scion_tokens": int((local["usage"] or {}).get("completion_tokens") or 0),
                    "truncated": truncated,
                    "human_answer": human_checked_answer,
                    "human_gate": human_record,
                    "repair": repair_record,
                    "repaired_answer": repaired_answer,
                    "router_route": (route_result.decision if route_result else None),
                    "router_confidence": (route_result.confidence if route_result else None),
                    "router_agrees": ((route_result.decision == "local") == local_ok)
                    if route_result
                    else None,
                    "verdict": verdict.decision,
                    "verdict_confidence": verdict.confidence,
                    "verdict_reason": verdict.payload.get("reason", ""),
                    "findings": list((verdict.payload or {}).get("findings") or []),
                    "escalated": escalate,
                    "gated": False,
                    "api_ok": api_ok,
                    "api_answer": (api_direct["content"][:4000] if api_direct else None),
                    "api_finish": (api_direct.get("finish_reason") if api_direct else None),
                    "api_ms": (round(api_direct["latency_ms"], 1) if api_direct else None),
                    "api_direct_correct": api_ok,
                    "cascade_ok": cascade_ok,
                    "false_accept": (not escalate) and (not cascade_ok),
                    "false_reject": escalate and local_ok,
                    "corrected": escalate and local_ok is False and api_ok,
                    "human_corrected": bool(
                        human_decision is not None
                        and not escalate
                        and human_decision["mode"] in ("answer", "hint")
                        and cascade_ok
                    ),
                    "judge": args.judge,
                    "gate": args.gate,
                    "gate_scored": bool(gate_row and gate_row.get("scored")),
                    "gate_fired": (gate_row.get("fired") if gate_row else None),
                    "gate_p": (gate_row.get("p") if gate_row else None),
                    "gate_cleared": (
                        repair_record.get("gate_cleared") if repair_record else None
                    ),
                    "fallback_available": fallback is not None,
                    "flagged": flagged,
                }
            )

            if in_lane(records[-1]):
                try:
                    lane.append(
                        lane_record(
                            records[-1], task,
                            run=(args.run_name.strip() or stamp),
                            draft=local["content"],
                            cascade_answer=cascade_answer,
                            findings=records[-1]["findings"],
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - never kill the run over the harvest
                    print(
                        f"[cascade] escalation lane write failed ({exc})",
                        file=sys.stderr,
                    )

    finally:
        # the lane handle must be closed even if a task raised
        lane.close()
    # -- aggregate -----------------------------------------------------
    n = len(records)
    measured = [r for r in records if r["scion_ok"] is not None]
    gated = [r for r in records if r.get("gated")]
    api_rows = [r for r in records if r["api_ok"] is not None]
    accepted = [r for r in records if not r["escalated"]]

    def mean(rows, key):
        values = [r[key] for r in rows if r.get(key) is not None]
        return round(sum(values) / len(values), 4) if values else None

    summary = {
        "tasks": n,
        "gated": len(gated),
        "gate_rate": round(len(gated) / n, 4),
        "scion_accuracy": mean(measured, "scion_ok"),
        "api_accuracy": mean(api_rows, "api_ok"),
        "cascade_accuracy": round(sum(r["cascade_ok"] for r in records) / n, 4),
        "escalation_rate": round(sum(r["escalated"] for r in records) / n, 4),
        "accept_rate": round(len(accepted) / n, 4),
        "false_accept_rate": mean(measured, "false_accept"),
        "accepted_and_wrong": sum(r["false_accept"] for r in measured),
        "false_rejects": sum(r["false_reject"] for r in measured),
        "corrections": sum(r["corrected"] for r in measured),
        "repairs_attempted": sum(1 for r in records if r.get("repair")),
        "repairs_cleared": sum(1 for r in records if (r.get("repair") or {}).get("cleared")),
        "repairs_still_flagged": sum(
            1 for r in records if r.get("repair") and not r["repair"]["cleared"]
        ),
        "gate_scored": sum(1 for r in records if r.get("gate_scored")),
        "gate_fires": sum(1 for r in records if r.get("gate_fired")),
        "gate_cleared": sum(
            1 for r in records if r.get("gate_cleared") is True
        ),
        "router_agreement": mean(measured, "router_agrees"),
        "scion_mean_ms": mean(measured, "scion_ms"),
        "scion_mean_tokens": mean(measured, "scion_tokens"),
        "api_mean_ms": mean(api_rows, "api_ms"),
        "fallback_available": fallback is not None,
        "flagged_no_fallback": sum(1 for r in records if r.get("flagged")),
        "human_corrected": sum(1 for r in measured if r.get("human_corrected")),
        "human_gate": {
            "enabled": human_gate is not None,
            "mode": args.human_gate,
            "asked": human_stats["asked"],
            "solve_attempts": human_stats["solve_attempts"],
            "solved": human_stats["solved"],
            "kept": human_stats["kept"],
            "escalated": human_stats["escalated"],
            "first_triggered": human_stats["first_triggered"],
            "reasons": human_stats["reasons"],
        },
        "usage": usage.summary(),
        "telemetry": telemetry.summary(),
    }

    report = {
        "spec": spec["name"],
        "scion_base": scion.base,
        "router": args.router,
        "judge": args.judge,
        "repair": args.repair,
        "gate": args.gate,
        "gate_head": args.gate_head,
        "gate_threshold": args.gate_threshold,
        "escalation_lane": {
            "path": (str(lane.path) if lane.path else None),
            "count": lane.count,
        },
        "local_temperature": args.local_temperature,
        "router_mode": args.router_mode,
        "api_fallback": args.api_fallback or None,
        "human_gate": args.human_gate,
        "api_model": (frontier.model if frontier is not None
                      else (fallback.model if fallback is not None else None)),
        "summary": summary,
        "records": records,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")

    print("\n=== cascade pilot ===")
    header = f"{'task':26} {'bucket':10} {'scion':>5} {'api':>4} {'route':>5} {'verdict':>8} {'esc':>4} {'cascade':>7}"
    print(header)
    for r in records:
        print(
            f"{r['id']:26} {r['bucket']:10} {str(r['scion_ok']):>5} {str(r['api_ok']):>4} "
            f"{str(r['router_route']):>5} {r['verdict']:>8} {str(r['escalated']):>4} "
            f"{str(r['cascade_ok']):>7}"
        )
    print("\nsummary:", json.dumps({k: v for k, v in summary.items() if k not in ('usage', 'telemetry')}, indent=1))
    print("usage:", json.dumps(summary["usage"], indent=1))
    print("report:", out_dir / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
