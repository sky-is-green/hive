#!/usr/bin/env python
"""Cascade runner — local generation + validation, optional API fallback.

The Strata-era default: the local endpoint answers every task (no
pre-generation router), a local validator checks the answer, and the API is
**off unless** ``--api-fallback`` names an OpenAI-compatible endpoint — then a
rejected answer escalates there.  Tiny-Jev is the default validator
(``--judge tiny-jev``, ~40 ms/check); ``--judge none`` skips validation and
``--judge local`` uses Intern-Decision-4B/JEV.

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

from experiments.cascade.checkers import check, extract_json  # noqa: E402

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
                self.usage.add(tag, data.get("usage"))
                choice = data["choices"][0]
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


class ScionTier:
    """The local tier: the llama-server hivebench spawned for the stack."""

    def __init__(self, base: str = SCION_BASE, *, max_tokens: int = 768,
                 reasoning_effort: str = "",
                 reasoning_budget_tokens: int = 0) -> None:
        self.base = base.rstrip("/")
        self.max_tokens = int(max_tokens)
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

    def chat(self, messages: list[dict], *, max_tokens: Optional[int] = None) -> dict:
        budget = int(max_tokens or self.max_tokens)
        payload = {
            "model": "scion",
            "messages": messages,
            "temperature": 0,
            "max_tokens": budget,
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        if self.reasoning_budget_tokens:
            payload["reasoning_budget_tokens"] = self.reasoning_budget_tokens
        data, ms = self._post_local(payload, timeout=900)
        msg = data["choices"][0]["message"]
        content = msg.get("content") or ""
        usage = data.get("usage") or {}
        # A reasoning model that ran out of budget before answering: retry once.
        if not content.strip() and int(usage.get("completion_tokens") or 0) >= budget:
            payload["max_tokens"] = max(2048, budget * 2)
            data, ms = self._post_local(payload, timeout=1200)
            msg = data["choices"][0]["message"]
            content = msg.get("content") or ""
            usage = data.get("usage") or {}
        return {
            "content": content,
            "reasoning": msg.get("reasoning_content") or "",
            "latency_ms": ms,
            "usage": usage,
        }


# ---------------------------------------------------------------- pipeline


def answer_prompt(task: dict) -> list[dict]:
    return [
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user", "content": task["prompt"]},
    ]


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
        "--judge", choices=("none", "tiny-jev", "local", "api"), default="tiny-jev",
        help="D3 validation: none, local Tiny-Jev (default, ~40 ms), local "
             "Intern-Decision-4B/JEV, or the frontier API prompt",
    )
    parser.add_argument("--judge-model", default="",
                        help="empty = the default model for the chosen judge")
    parser.add_argument("--judge-device", default="cpu",
                        help="cpu (default; safe beside the resident engine) or "
                             "cuda when the GPUs are free")
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

    scion = ScionTier(
        args.scion_base,
        max_tokens=args.local_max_tokens,
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
    needs_api = measure or args.router == "api" or args.judge == "api"
    frontier = Frontier(usage) if needs_api else None
    fallback = None
    if args.api_fallback:
        fallback = Frontier(
            usage,
            base_url=args.api_fallback,
            model=args.api_fallback_model or None,
            key=args.api_fallback_key or None,
        )
    telemetry = TelemetryLog()
    policy = Policy(accept_confidence=args.judge_threshold)
    broker = JudgementBroker()

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
    if args.judge_device != "cpu" and scion.alive():
        print(
            "[cascade] WARNING: a torch judge on the GPUs while the local engine "
            "is resident creates a second ROCm context; this host can hang "
            "(harness/models.py).  Use --judge-device cpu (the default) unless "
            "the GPUs are actually free.",
            file=sys.stderr, flush=True,
        )
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
                    "fallback_available": fallback is not None,
                    "flagged": False,
                }
            )
            continue

        local = scion.chat(answer_prompt(task))
        local_ok = check(task, local["content"])
        telemetry.log(
            task["id"], planned.path, "E4-local", "dgpu1", local["latency_ms"],
            tokens=int((local["usage"] or {}).get("completion_tokens") or 0),
            outcome="ok" if local_ok else "wrong",
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
        telemetry.log(
            task["id"], planned.path, "D3-verify",
            "api" if args.judge == "api" else args.judge_device,
            verdict.latency_ms,
            outcome="reject" if rejected else "accept", async_=True,
        )

        # Escalate only when there is somewhere to escalate to.  With the API
        # fallback unset (the default), a reject is recorded and the local
        # answer is returned; measure mode uses the measurement API instead.
        escalate = rejected and (measure or fallback is not None)
        api_direct = None
        if measure:
            api_direct = frontier.chat(answer_prompt(task), tag="answer", max_tokens=700)
        elif escalate:
            api_direct = fallback.chat(answer_prompt(task), tag="fallback", max_tokens=700)
        api_ok = check(task, api_direct["content"]) if api_direct is not None else None

        cascade_answer = api_direct["content"] if escalate else local["content"]
        cascade_ok = api_ok if escalate else local_ok
        flagged = rejected and not measure and fallback is None

        if escalate:
            telemetry.log(
                task["id"], planned.path,
                "E4-api" if measure else "E4-fallback",
                "api", api_direct["latency_ms"],
                outcome="ok" if api_ok else "wrong", escalated=True,
            )

        records.append(
            {
                "id": task["id"],
                "bucket": task["bucket"],
                "planned_path": planned.path,
                "scion_answer": local["content"][:4000],
                "scion_ok": local_ok,
                "scion_ms": round(local["latency_ms"], 1),
                "scion_tokens": int((local["usage"] or {}).get("completion_tokens") or 0),
                "router_route": (route_result.decision if route_result else None),
                "router_confidence": (route_result.confidence if route_result else None),
                "router_agrees": ((route_result.decision == "local") == local_ok)
                if route_result
                else None,
                "verdict": verdict.decision,
                "verdict_confidence": verdict.confidence,
                "verdict_reason": verdict.payload.get("reason", ""),
                "escalated": escalate,
                "gated": False,
                "api_ok": api_ok,
                "api_answer": (api_direct["content"][:4000] if api_direct else None),
                "api_finish": (api_direct.get("finish_reason") if api_direct else None),
                "api_ms": (round(api_direct["latency_ms"], 1) if api_direct else None),
                "api_direct_correct": api_ok,
                "cascade_ok": cascade_ok,
                "false_accept": (not escalate) and (not local_ok),
                "false_reject": escalate and local_ok,
                "corrected": escalate and local_ok is False and api_ok,
                "judge": args.judge,
                "fallback_available": fallback is not None,
                "flagged": flagged,
            }
        )

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
        "router_agreement": mean(measured, "router_agrees"),
        "scion_mean_ms": mean(measured, "scion_ms"),
        "scion_mean_tokens": mean(measured, "scion_tokens"),
        "api_mean_ms": mean(api_rows, "api_ms"),
        "fallback_available": fallback is not None,
        "flagged_no_fallback": sum(1 for r in records if r.get("flagged")),
        "usage": usage.summary(),
        "telemetry": telemetry.summary(),
    }

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out) / (args.run_name.strip() or stamp)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "spec": spec["name"],
        "scion_base": scion.base,
        "router": args.router,
        "judge": args.judge,
        "router_mode": args.router_mode,
        "api_fallback": args.api_fallback or None,
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
