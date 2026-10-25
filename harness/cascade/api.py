"""``/v1/cascade/*`` router factory — the console's contract.

Thin HTTP handlers over :mod:`harness.cascade` and the pilot/oracle run
artifacts.  This is the seam that lets any console (the DSH panel, the Studio
page, a bare ``curl``) drive the cascade engine without re-implementing it:
the router wraps ``plan_request``, the registry, the oracle fronts and the
runner under ``experiments/cascade/``.

Route table:

============================  ====  ========================================
Path                          Verb  Contract
============================  ====  ========================================
``/v1/cascade/roles``         GET   role taxonomy A1–H1 (data)
``/v1/cascade/paths``         GET   paths P0–P6 + the route table
``/v1/cascade/registry``      GET   candidate catalog + role coverage
``/v1/cascade/validator``     GET   local validator status (model/device/threshold)
``/v1/cascade/validate``      POST  score one candidate answer (P(correct))
``/v1/cascade/plan``          GET   ``plan_request`` for bucket/confidence/out_len
``/v1/cascade/frontier``      GET   per-role Pareto fronts from an oracle file
``/v1/cascade/senses/status`` GET   sensory encoder/ingress state
``/v1/cascade/senses``        POST  classify, store and encode one upload
``/v1/cascade/run``           POST  launch a pilot/oracle run (background)
``/v1/cascade/runs``          GET   run index, newest first
``/v1/cascade/report/{run}``  GET   one run's ``report.json``
============================  ====  ========================================

Malformed input is 4xx, never 500.  ``run`` launches a subprocess, so its
``tasks``/``run_name`` inputs are strictly sanitised and confined to the
cascade experiment directory.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional

from fastapi import APIRouter, Body, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from harness.cascade.oracle import CandidateMeasurement, frontier_by_role
from harness.cascade.paths import PATHS, ROUTE_TABLE
from harness.cascade.registry import CandidateRegistry, default_registry
from harness.cascade.roles import ROLES, Device
from harness.cascade.scheduler import LatencyModel, RouteDecision, plan_request
from harness.cascade.senses import (
    EncoderError,
    PayloadTooLarge,
    SensesError,
    UnsupportedModality,
    bucket_for_modality,
)

#: Router prefix (the app mounts the router without adding another prefix).
PREFIX = "/v1/cascade"

REPO_ROOT = Path(__file__).resolve().parents[2]
CASCADE_DIR = REPO_ROOT / "experiments" / "cascade"
DEFAULT_RUNS_ROOT = CASCADE_DIR / "runs"
DEFAULT_RUNNER = CASCADE_DIR / "run_cascade.py"

#: Run directory names the router will create or read.
_RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

Spawner = Callable[[list[str], Path, Path], int]


def _default_spawn(cmd: list[str], cwd: Path, log_path: Path) -> int:
    """Start ``cmd`` detached with stdout+stderr in ``log_path``; return its pid."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "ab") as fh:
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=fh, stderr=subprocess.STDOUT)
    return proc.pid


def _check_run_name(name: str) -> str:
    if not _RUN_NAME_RE.match(name):
        raise HTTPException(
            422, "run name must match [A-Za-z0-9][A-Za-z0-9._-]{0,63}"
        )
    return name


def _resolve_run(runs_root: Path, name: str) -> Path:
    _check_run_name(name)
    target = (runs_root / name).resolve()
    root = runs_root.resolve()
    if target != root and root not in target.parents:
        raise HTTPException(422, "run name escapes the runs root")
    return target


def _run_summary(run_dir: Path) -> dict[str, Any]:
    """One run-index row; ``status`` is ``done`` when the report exists."""
    report_path = run_dir / "report.json"
    row: dict[str, Any] = {
        "run": run_dir.name,
        "path": str(run_dir),
        "created": time.strftime(
            "%Y-%m-%dT%H:%M:%S", time.localtime(run_dir.stat().st_mtime)
        ),
        "status": "done" if report_path.is_file() else "running",
    }
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            summary = report.get("summary") or {}
            row.update(
                {
                    "spec": report.get("spec"),
                    "tasks": summary.get("tasks"),
                    "scion_accuracy": summary.get("scion_accuracy"),
                    "api_accuracy": summary.get("api_accuracy"),
                    "cascade_accuracy": summary.get("cascade_accuracy"),
                    "escalation_rate": summary.get("escalation_rate"),
                    "cost_usd": (summary.get("usage") or {}).get("total_cost_usd"),
                }
            )
        except (OSError, ValueError):
            row["status"] = "unreadable"
    return row



_SECRET_FLAGS = ("--api-fallback-key", "--api-key")


def _redact_cmd(cmd: list[str]) -> list[str]:
    """Copy an argv with secret-bearing values replaced by ``***``."""
    out = list(cmd)
    for i, arg in enumerate(out):
        if arg in _SECRET_FLAGS and i + 1 < len(out):
            out[i + 1] = "***"
    return out


class CascadeRunRequest(BaseModel):
    """Every knob the cascade runner accepts. All optional: the runner's
    defaults apply for anything omitted, and the harness-wide baseline
    (the Global tab's Quality block) fills what the caller leaves out."""

    tasks: str = Field("tasks-hard.json", description="Task bundle file name in the cascade dir.")
    run_name: str = Field("", description="Output directory name (default: timestamped).")
    limit: int = Field(0, ge=0, description="Run only the first N tasks (0 = all).")
    judge: Optional[str] = Field(None, description="Verifier: deterministic | learned.")
    judge_threshold: Optional[float] = Field(None, ge=0, le=1, description="Accept when P(correct) >= this.")
    repair: Optional[str] = Field(None, description="Repair leg: auto | on | off.")
    gate: Optional[str] = Field(None, description="Learned gate: off | intern-0.8b | scion.")
    gate_head: Optional[str] = Field(None, description="Gate head: L0.5 | L0.75 | L1 | ridge-concat | mlp-concat.")
    gate_threshold: Optional[float] = Field(None, ge=0, le=1, description="Fire the gate when P(correct) < this.")
    human_gate: Optional[str] = Field(None, description="Human gate before API escalation: auto | on | off.")
    router: Optional[str] = Field(None, description="Router model: local | jev.")
    router_mode: Optional[str] = Field(None, description="gate | measure.")
    local_temperature: Optional[float] = Field(None, description="Sampling temperature for local drafts.")
    local_max_tokens: Optional[int] = Field(None, ge=1, description="Token budget per draft.")
    local_reasoning_effort: Optional[str] = Field(None, description="none | low | medium | high.")
    local_reasoning_budget_tokens: Optional[int] = Field(None, ge=0, description="Reasoning token budget (0 = auto).")
    scion_base: Optional[str] = Field(None, description="Local model server base URL (the local tier).")
    api_fallback: Optional[str] = Field(None, description="OpenAI-compatible endpoint for escalated items.")
    api_fallback_model: Optional[str] = Field(None, description="Model name for the API fallback.")
    api_fallback_key: Optional[str] = Field(None, description="Bearer key for the API fallback.")
    no_escalation_lane: bool = Field(False, description="Skip writing escalation.jsonl (lean runs).")


def create_router(
    *,
    runs_root: Any = None,
    runner: Any = None,
    tasks_root: Any = None,
    spawn: Optional[Spawner] = None,
    registry: Optional[CandidateRegistry] = None,
    validator: Any = None,
    senses: Any = None,
    defaults: Optional[Callable[[], dict]] = None,
) -> APIRouter:
    """Return the ``/v1/cascade`` router.

    ``runs_root``/``tasks_root``/``runner`` are injectable so tests (and the
    app) can point the router at a tmp tree; ``spawn`` replaces the subprocess
    launcher in tests; ``validator`` is a
    :class:`harness.cascade.validator.ValidatorService`-like object (lazy in
    production, a fake in tests); ``senses`` is a
    :class:`harness.cascade.senses.SensesService` (unconfigured by default so
    the API never fabricates embeddings); ``defaults`` returns the harness-wide
    baseline (the Global tab's Quality block) that a run starts from
    unless it says otherwise.
    """
    router = APIRouter(prefix=PREFIX, tags=["cascade"])
    runs = Path(runs_root) if runs_root is not None else DEFAULT_RUNS_ROOT
    tasks_dir = Path(tasks_root) if tasks_root is not None else CASCADE_DIR
    runner_path = Path(runner) if runner is not None else DEFAULT_RUNNER
    launcher: Spawner = spawn or _default_spawn
    catalog = registry or default_registry()

    def baseline() -> dict:
        """The Global tab's quality baseline (empty when there is none)."""
        try:
            value = defaults() if defaults else {}
        except Exception:  # noqa: BLE001 - a broken baseline must not stop a run
            value = {}
        return dict(value or {})

    @router.get("/roles")
    def roles() -> dict[str, Any]:
        """The role taxonomy (A1, B1, C1, D3, E4, ...) with descriptions."""
        return {"count": len(ROLES), "roles": [role.to_dict() for role in ROLES]}

    @router.get("/paths")
    def paths() -> dict[str, Any]:
        """The route table: path ids, escalation paths, and expected costs."""
        return {
            "paths": [path.to_dict() for path in PATHS],
            "route_table": dict(ROUTE_TABLE),
        }

    @router.get("/senses/status")
    def senses_status() -> dict[str, Any]:
        """Sensory ingress state; no encoder is loaded by this call."""
        if senses is None:
            return {
                "configured": False,
                "loaded": False,
                "note": "no sensory encoder attached to this sidecar",
            }
        return {"configured": True, **senses.status()}

    @router.post("/senses")
    async def sense_upload(file: UploadFile = File(...)) -> dict[str, Any]:
        """Classify, store and encode one upload; suggest a cascade path.

        This is the A2/P6 front door.  It does not generate an answer: it
        returns the embedding and the scheduler's path preview so a caller
        can hand the stored content to the vision generator.
        """
        if senses is None:
            raise HTTPException(503, "no sensory encoder attached to this sidecar")

        limit = int(senses.max_bytes)
        declared = getattr(file, "size", None)
        if declared is not None and int(declared) > limit:
            raise HTTPException(413, f"upload exceeds the {limit}-byte limit")
        try:
            data = await file.read(limit + 1)
        except OSError as exc:
            raise HTTPException(400, f"could not read upload: {exc}") from exc
        if len(data) > limit:
            raise HTTPException(413, f"upload exceeds the {limit}-byte limit")

        try:
            sense = await run_in_threadpool(
                partial(
                    senses.ingest,
                    data=data,
                    filename=file.filename,
                    content_type=file.content_type,
                )
            )
        except PayloadTooLarge as exc:
            raise HTTPException(413, str(exc)) from exc
        except UnsupportedModality as exc:
            raise HTTPException(415, str(exc)) from exc
        except EncoderError as exc:
            raise HTTPException(503, str(exc)) from exc
        except SensesError as exc:
            raise HTTPException(422, str(exc)) from exc

        bucket = bucket_for_modality(sense.modality)
        path = ROUTE_TABLE.get(bucket) if bucket else None
        plan = (
            plan_request(RouteDecision(bucket=bucket, confidence=1.0)).to_dict()
            if bucket and path
            else None
        )
        out = sense.to_dict()
        out.update(
            {
                "bucket": bucket,
                "path": path,
                "plan": plan,
                "routed": path is not None,
            }
        )
        if path is None:
            out["note"] = (
                "encoded and stored; no cascade path is defined for this "
                "modality yet"
            )
        return out

    @router.get("/registry")
    def registry_view() -> dict[str, Any]:
        """The candidate catalog: every model candidate with the roles it covers."""
        candidates = [c.to_dict() for c in catalog]
        return {
            "count": len(candidates),
            "candidates": candidates,
            "missing_roles": list(catalog.missing_roles([r.id for r in ROLES])),
        }

    @router.get("/validator")
    def validator_status() -> dict[str, Any]:
        """Validator service state (model, device, loaded)."""
        if validator is None:
            return {"configured": False, "loaded": False,
                    "note": "no validator attached to this sidecar"}
        return {"configured": True, **validator.status()}

    @router.post("/validate")
    def validate(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        """Validate a candidate answer against a task with the validator model."""
        if validator is None:
            raise HTTPException(503, "no validator attached to this sidecar")
        task = body.get("task")
        candidate = body.get("candidate")
        if not isinstance(task, dict) or not str(task.get("prompt") or "").strip():
            raise HTTPException(422, "task must be an object with a non-empty prompt")
        if not isinstance(candidate, str) or not candidate.strip():
            raise HTTPException(422, "candidate must be a non-empty string")
        try:
            return validator.verdict(task, candidate, threshold=body.get("threshold"))
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001 - a broken model is a 503
            raise HTTPException(503, f"validator failed: {exc}") from exc

    @router.get("/plan")
    def plan(
        bucket: str = Query("qa"),
        confidence: float = Query(1.0, ge=0.0, le=1.0),
        out_len: int = Query(0, ge=0, le=1_000_000),
    ) -> dict[str, Any]:
        """Plan the route for a bucket + confidence + output length."""
        decision = RouteDecision(bucket=bucket, confidence=confidence)
        result = plan_request(decision, out_len=out_len)
        return {
            "request": {
                "bucket": bucket,
                "confidence": confidence,
                "out_len": out_len,
            },
            "plan": result.to_dict(),
            "latency_model": LatencyModel.from_brief().to_dict(),
        }

    @router.get("/frontier")
    def frontier(file: str = Query("oracle.json")) -> dict[str, Any]:
        """Frontier measurements per role (quality vs latency curves)."""
        # A basename only: the query cannot escape the cascade directory.
        name = Path(file).name
        if not name.endswith(".json"):
            raise HTTPException(422, "oracle file must be a .json in the cascade dir")
        path = tasks_dir / name
        if not path.is_file():
            return {
                "file": name,
                "measurements": [],
                "frontiers": {},
                "note": "no oracle measurements yet",
            }
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise HTTPException(400, f"unreadable oracle file {name!r}: {exc}") from exc
        rows_raw = raw.get("measurements", raw) if isinstance(raw, dict) else raw
        if not isinstance(rows_raw, list):
            raise HTTPException(400, "oracle file must be a list of measurements")
        try:
            rows = [
                CandidateMeasurement(
                    candidate=str(r["candidate"]),
                    role=str(r["role"]),
                    device=Device(str(r["device"])),
                    quality=float(r["quality"]),
                    latency_ms=float(r["latency_ms"]),
                    unit=str(r.get("unit", "call")),
                    ctx=int(r.get("ctx", 0)),
                    precision=str(r.get("precision", "bf16")),
                    bytes_gb=(None if r.get("bytes_gb") is None else float(r["bytes_gb"])),
                    notes=str(r.get("notes", "")),
                )
                for r in rows_raw
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(400, f"malformed measurement: {exc}") from exc
        fronts = frontier_by_role(rows)
        return {
            "file": name,
            "measurements": [r.to_dict() for r in rows],
            "frontiers": {role: [r.to_dict() for r in front] for role, front in fronts.items()},
        }

    @router.post("/run")
    def launch(req: CascadeRunRequest) -> dict[str, Any]:
        body = req.model_dump()
        """Launch a cascade run: tasks, run name, limit, and every runner knob."""
        # the harness-wide baseline (the Global tab's Quality block) fills
        # the knobs the caller left alone; a run that names its own wins
        defaults_map = baseline()
        for key, value in defaults_map.items():
            if body.get(key) is None and value is not None:
                body[key] = value
        tasks_name = str(body.get("tasks") or "tasks-hard.json")
        if Path(tasks_name).name != tasks_name or not tasks_name.endswith(".json"):
            raise HTTPException(422, "tasks must be a .json file name in the cascade dir")
        tasks_path = tasks_dir / tasks_name
        if not tasks_path.is_file():
            raise HTTPException(404, f"no such task file: {tasks_name}")
        if not runner_path.is_file():
            raise HTTPException(409, f"runner not found at {runner_path}")
        limit = int(body.get("limit") or 0)
        if limit < 0:
            raise HTTPException(422, "limit must be >= 0")
        run_name = str(
            body.get("run_name") or time.strftime("cascade-%Y%m%d-%H%M%S")
        )
        _check_run_name(run_name)
        run_dir = runs / run_name
        if run_dir.exists():
            raise HTTPException(409, f"run {run_name!r} already exists")
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, str(runner_path),
            "--tasks", str(tasks_path),
            "--out", str(runs),
            "--run-name", run_name,
        ]
        if limit:
            cmd += ["--limit", str(limit)]

        # Cascade knobs (all optional; runner defaults apply when omitted).
        _str_opt = (
            ("judge", "--judge"),
            ("repair", "--repair"),
            ("gate", "--gate"),
            ("gate_head", "--gate-head"),
            ("human_gate", "--human-gate"),
            ("router", "--router"),
            ("router_mode", "--router-mode"),
            ("local_reasoning_effort", "--local-reasoning-effort"),
            ("scion_base", "--scion-base"),
            ("api_fallback", "--api-fallback"),
            ("api_fallback_model", "--api-fallback-model"),
            ("api_fallback_key", "--api-fallback-key"),
        )
        for key, flag in _str_opt:
            val = body.get(key)
            if val not in (None, ""):
                cmd += [flag, str(val)]
        _num_opt = (
            ("gate_threshold", "--gate-threshold", float),
            ("local_temperature", "--local-temperature", float),
            ("local_max_tokens", "--local-max-tokens", int),
            ("local_reasoning_budget_tokens", "--local-reasoning-budget-tokens", int),
            ("judge_threshold", "--judge-threshold", float),
        )
        for key, flag, cast in _num_opt:
            val = body.get(key)
            if val not in (None, ""):
                try:
                    cmd += [flag, str(cast(val))]
                except (TypeError, ValueError) as exc:
                    raise HTTPException(422, f"bad value for {key}: {exc}") from exc
        if body.get("no_escalation_lane"):
            cmd.append("--no-escalation-lane")

        pid = launcher(cmd, REPO_ROOT, run_dir / "run_stdout.log")
        return {
            "ok": True,
            "run": run_name,
            "run_dir": str(run_dir),
            "log": str(run_dir / "run_stdout.log"),
            "pid": pid,
            # the argv is echoed for transparency, but secrets never are
            "cmd": _redact_cmd(cmd),
        }

    @router.get("/runs")
    def runs_index() -> dict[str, Any]:
        """List run bundles available for viewing."""
        if not runs.is_dir():
            return {"runs": []}
        rows = [
            _run_summary(child)
            for child in runs.iterdir()
            if child.is_dir() and _RUN_NAME_RE.match(child.name)
        ]
        rows.sort(key=lambda r: r["created"], reverse=True)
        return {"runs": rows}

    @router.get("/report/{run}")
    def report(run: str) -> dict[str, Any]:
        """Return the run_report.json bundle for a run directory."""
        target = _resolve_run(runs, run)
        path = target / "report.json"
        if not path.is_file():
            raise HTTPException(404, f"no report for run {run!r}")
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise HTTPException(400, f"unreadable report for {run!r}: {exc}") from exc

    return router
