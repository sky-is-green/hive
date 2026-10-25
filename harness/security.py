"""Security policy, tool grants, and the approval gateway core.

This is the enforceable part of the tool boundary: tool metadata, the policy
tier, session/global grants, reason checking, headless denial, private-network
gating, pending approvals, and append-only audit.  Tool implementations are
injected into :meth:`ApprovalGateway.execute`; the gateway never treats a
tool's own claims as authority.

The bundled DSH runtime has no pre-execution approval hook yet.  Until that
adapter lands, this module is the boundary for Hive/MCP/browser adapters and
the future DSH proxy, not a claim that the current agent loop is already
wrapped.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

from harness.state_store import AuditLog, JsonStore, StatePaths

TIERS: tuple[str, ...] = ("strict", "guarded", "trusted", "open")
DEFAULT_TIER = "guarded"
APPROVAL_TTL_S = 600.0
REASON_MIN_CHARS = 8
GENERIC_REASONS = frozenset({
    "", "because", "do it", "test", "testing", "please", "run it", "yes",
    "n/a", "na", "none", "ok", "okay", "go", "execute",
})
_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,160}$")
_SOURCE = frozenset({"hive", "dsh", "mcp", "adapter"})
_NETWORK = frozenset({"none", "public", "private"})
_FILESYSTEM = frozenset({"none", "read", "workspace", "host"})
_RISK = frozenset({"low", "medium", "high"})
_SANDBOX = frozenset({"required", "optional", "unsupported"})

DEFAULT_SECURITY = {
    "schema_version": 1,
    "tier": DEFAULT_TIER,
    "master": {
        "approval_prompts": True,
        "sandbox": True,
        "private_network": True,
    },
    "headless": {"read_only": "auto", "side_effect": "deny"},
    "updated_at": None,
    "updated_by": "local-user",
}

DEFAULT_GRANTS = {"schema_version": 1, "grants": []}


class SecurityError(RuntimeError):
    """Security state or a decision could not be used safely."""


class InvalidTier(SecurityError):
    """The requested tier is not one Hive understands."""


class ApprovalError(SecurityError):
    """An approval request is missing, expired, or malformed."""


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(
        timespec="seconds").replace("+00:00", "Z")


def _safe(value: Any, what: str) -> str:
    text = str(value or "").strip()
    if not _SAFE.match(text):
        raise SecurityError(
            f"invalid {what} {value!r}: use 1-160 letters, digits, '.', '_', "
            "'-', ':', '/', '@' or '+'")
    return text


def _safe_reason(reason: str | None) -> bool:
    text = str(reason or "").strip()
    if len(text) < REASON_MIN_CHARS:
        return False
    return text.lower() not in GENERIC_REASONS


def _json_hash(value: Any) -> str:
    try:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                             default=str)
    except (TypeError, ValueError):
        payload = repr(value)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ToolSpec:
    """The metadata a tool source must declare; unknown fields are safe."""

    name: str
    source: str = "hive"
    version: str = "1.0.0"
    description: str = ""
    read_only: bool = False
    side_effect: bool | None = None
    reason_required: bool | None = None
    network: str = "none"
    filesystem: str = "none"
    risk: str = "high"
    sandbox: str = "required"

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _safe(self.name, "tool name"))
        object.__setattr__(self, "source",
                           self.source if self.source in _SOURCE else "adapter")
        object.__setattr__(self, "version", _safe(self.version, "tool version"))
        object.__setattr__(self, "network",
                           self.network if self.network in _NETWORK else "private")
        object.__setattr__(
            self, "filesystem",
            self.filesystem if self.filesystem in _FILESYSTEM else "host")
        object.__setattr__(self, "risk",
                           self.risk if self.risk in _RISK else "high")
        object.__setattr__(
            self, "sandbox",
            self.sandbox if self.sandbox in _SANDBOX else "required")

    @property
    def identity(self) -> str:
        return f"{self.source}:{self.name}@{self.version}"

    @property
    def is_side_effect(self) -> bool:
        if self.side_effect is not None:
            return bool(self.side_effect)
        return not bool(self.read_only)

    @property
    def needs_reason(self) -> bool:
        if self.reason_required is not None:
            return bool(self.reason_required)
        return self.is_side_effect

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "source": self.source, "version": self.version,
            "description": self.description, "read_only": self.read_only,
            "side_effect": self.side_effect,
            "reason_required": self.reason_required, "network": self.network,
            "filesystem": self.filesystem, "risk": self.risk,
            "sandbox": self.sandbox, "identity": self.identity,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ToolSpec":
        if not isinstance(raw, Mapping):
            raise SecurityError("tool metadata must be an object")
        return cls(
            name=str(raw.get("name") or ""),
            source=str(raw.get("source") or "adapter"),
            version=str(raw.get("version") or "1.0.0"),
            description=str(raw.get("description") or ""),
            read_only=bool(raw.get("read_only", False)),
            side_effect=raw.get("side_effect") if isinstance(
                raw.get("side_effect"), bool) else None,
            reason_required=raw.get("reason_required") if isinstance(
                raw.get("reason_required"), bool) else None,
            network=str(raw.get("network") or "private"),
            filesystem=str(raw.get("filesystem") or "host"),
            risk=str(raw.get("risk") or "high"),
            sandbox=str(raw.get("sandbox") or "required"),
        )


@dataclass(frozen=True)
class ToolRequest:
    session_id: str
    action: str
    reason: str = ""
    args: Mapping[str, Any] = field(default_factory=dict)
    turn_id: str | None = None
    timeout_s: float = APPROVAL_TTL_S

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id",
                           _safe(self.session_id, "session id"))
        object.__setattr__(self, "action", str(self.action or "")[:4096])
        object.__setattr__(self, "reason", str(self.reason or "")[:4096])
        object.__setattr__(self, "args", dict(self.args or {}))
        object.__setattr__(self, "timeout_s",
                           max(1.0, min(3600.0, float(self.timeout_s))))


class SecurityStore:
    """The tier and master switches; missing/corrupt means Guarded."""

    def __init__(self, state: StatePaths | None = None) -> None:
        self.state = (state or StatePaths.default()).ensure()
        self._store = JsonStore(self.state.security_file(),
                                default=DEFAULT_SECURITY)

    def read(self) -> dict[str, Any]:
        raw = self._store.read()
        out = json.loads(json.dumps(DEFAULT_SECURITY))
        out.update({k: raw[k] for k in ("schema_version", "tier", "master",
                                        "headless", "updated_at", "updated_by")
                    if k in raw})
        if out.get("tier") not in TIERS:
            out["tier"] = DEFAULT_TIER
        return out

    def status(self) -> dict[str, Any]:
        return self.read()

    def _audit(self, **fields: Any) -> None:
        AuditLog(self.state.audit_file()).append(
            {"event": "security.policy", **fields})

    def set_tier(self, tier: str, *, updated_by: str = "local-user") -> dict[str, Any]:
        if tier not in TIERS:
            raise InvalidTier(f"tier must be one of {', '.join(TIERS)}")
        result = self._store.update(lambda state: {
            **state, "tier": tier, "updated_at": _utc(),
            "updated_by": updated_by,
        })
        self._audit(tier=tier, updated_by=updated_by)
        return result

    def set_master(self, **switches: Any) -> dict[str, Any]:
        allowed = set(DEFAULT_SECURITY["master"])
        unknown = set(switches) - allowed
        if unknown:
            raise SecurityError(
                f"unknown master switch(es): {', '.join(sorted(unknown))}")

        def update(state: dict[str, Any]) -> dict[str, Any]:
            master = dict(state.get("master") or {})
            master.update({key: bool(value) for key, value in switches.items()})
            return {**state, "master": master, "updated_at": _utc()}

        result = self._store.update(update)
        self._audit(master=switches)
        return result


class SessionGrants:
    """In-memory grants; they disappear with the process/session."""

    def __init__(self) -> None:
        self._grants: dict[tuple[str, str], dict[str, Any]] = {}

    def grant(self, session_id: str, identity: str, *,
              reason: str = "") -> dict[str, Any]:
        row = {
            "tool_identity": identity, "scope": "session",
            "session_id": session_id, "granted_at": _utc(),
            "last_used_at": None, "expires_at": None, "reason": reason,
        }
        self._grants[(session_id, identity)] = row
        return dict(row)

    def has(self, session_id: str, identity: str) -> bool:
        return (session_id, identity) in self._grants

    def revoke(self, session_id: str, identity: str | None = None) -> int:
        keys = [key for key in self._grants
                if key[0] == session_id
                and (identity is None or key[1] == identity)]
        for key in keys:
            del self._grants[key]
        return len(keys)

    def list(self, session_id: str | None = None) -> list[dict[str, Any]]:
        return [dict(row) for (sid, _), row in sorted(self._grants.items())
                if session_id is None or sid == session_id]


class GrantStore:
    """Global grants are durable; session grants are explicitly in memory."""

    def __init__(self, state: StatePaths | None = None, *,
                 sessions: SessionGrants | None = None) -> None:
        self.state = (state or StatePaths.default()).ensure()
        self.sessions = sessions or SessionGrants()
        self._store = JsonStore(self.state.grants_file(),
                                default=DEFAULT_GRANTS)

    def _rows(self) -> list[dict[str, Any]]:
        rows = self._store.read().get("grants") or []
        return [dict(row) for row in rows if isinstance(row, dict)]

    def _expired(self, row: Mapping[str, Any], now: float) -> bool:
        value = row.get("expires_at")
        if value is None:
            return False
        try:
            if isinstance(value, (int, float)):
                return float(value) <= now
            return datetime.fromisoformat(str(value).replace("Z", "+00:00")) \
                .timestamp() <= now
        except (TypeError, ValueError):
            return True

    def has(self, identity: str, session_id: str | None = None) -> bool:
        if session_id and self.sessions.has(session_id, identity):
            return True
        now = time.time()
        return any(row.get("tool_identity") == identity
                   and not self._expired(row, now)
                   for row in self._rows())

    def grant(self, identity: str, *, scope: str, session_id: str | None,
              reason: str = "", display_name: str = "") -> dict[str, Any]:
        if scope not in ("global", "session"):
            raise SecurityError("scope must be global or session")
        if scope == "session":
            if not session_id:
                raise SecurityError("session grant needs a session id")
            return self.sessions.grant(session_id, identity, reason=reason)
        row = {
            "tool_identity": identity, "scope": "global",
            "display_name": display_name or identity.split("@", 1)[0],
            "granted_at": _utc(), "last_used_at": None,
            "expires_at": None, "reason": reason,
        }

        def update(state: dict[str, Any]) -> dict[str, Any]:
            rows = [r for r in (state.get("grants") or [])
                    if isinstance(r, dict)
                    and r.get("tool_identity") != identity]
            rows.append(row)
            return {"schema_version": 1, "grants": rows}

        self._store.update(update)
        AuditLog(self.state.audit_file()).append(
            {"event": "tool.grant", "scope": "global",
             "tool_identity": identity, "reason": reason})
        return dict(row)

    def revoke(self, identity: str | None = None, *,
               session_id: str | None = None) -> dict[str, int]:
        removed_session = (self.sessions.revoke(session_id, identity)
                           if session_id else 0)
        removed_global = 0
        if identity is not None:
            def update(state: dict[str, Any]) -> dict[str, Any]:
                nonlocal removed_global
                before = [r for r in (state.get("grants") or [])
                          if isinstance(r, dict)]
                after = [r for r in before
                         if r.get("tool_identity") != identity]
                removed_global = len(before) - len(after)
                return {"schema_version": 1, "grants": after}
        else:
            def update(state: dict[str, Any]) -> dict[str, Any]:
                nonlocal removed_global
                removed_global = len([r for r in (state.get("grants") or [])
                                      if isinstance(r, dict)])
                return {"schema_version": 1, "grants": []}

        if identity is not None or removed_session == 0:
            self._store.update(update)
        AuditLog(self.state.audit_file()).append(
            {"event": "tool.revoke", "tool_identity": identity,
             "session_id": session_id, "global_removed": removed_global})
        return {"session": removed_session, "global": removed_global}

    def list(self, session_id: str | None = None) -> list[dict[str, Any]]:
        now = time.time()
        rows = [row for row in self._rows() if not self._expired(row, now)]
        rows += self.sessions.list(session_id)
        return rows


@dataclass
class _Pending:
    request_id: str
    spec: ToolSpec
    request: ToolRequest
    created_at: float
    expires_at: float


class GatewayHub:
    """Process-local gateways that share session grants and pending requests."""

    def __init__(self, state: StatePaths | None = None) -> None:
        self.state = (state or StatePaths.default()).ensure()
        self.security = SecurityStore(self.state)
        self.grants = GrantStore(self.state, sessions=SessionGrants())
        self._gateways: dict[str, ApprovalGateway] = {}

    def gateway(self, session_id: str, *,
                headless: bool = False) -> "ApprovalGateway":
        key = str(session_id)
        gateway = self._gateways.get(key)
        if gateway is None:
            gateway = ApprovalGateway(
                self.state, session_id=key, security=self.security,
                grants=self.grants, headless=headless)
            self._gateways[key] = gateway
        else:
            gateway.headless = bool(headless)
        return gateway

    def pending(self, session_id: str | None = None) -> list[dict[str, Any]]:
        if session_id is not None:
            return self.gateway(session_id).pending()
        rows: list[dict[str, Any]] = []
        for gateway in self._gateways.values():
            rows.extend(gateway.pending())
        return rows

    def decide(self, session_id: str, request_id: str, choice: str, *,
               scope: str = "session") -> dict[str, Any]:
        return self.gateway(session_id).decide(
            request_id, choice, scope=scope)

    def decision(self, session_id: str,
                 request_id: str) -> dict[str, Any] | None:
        return self.gateway(session_id).decision(request_id)


class ApprovalGateway:
    """Evaluate and optionally execute one tool call.

    ``execute`` is the only execution path in this class.  A pending request
    returns without running the executor; the caller must decide and retry.
    """

    def __init__(self, state: StatePaths | None = None, *,
                 session_id: str, security: SecurityStore | None = None,
                 grants: GrantStore | None = None,
                 headless: bool = False,
                 now: Callable[[], float] = time.time) -> None:
        self.state = (state or StatePaths.default()).ensure()
        self.session_id = _safe(session_id, "session id")
        self.security = security or SecurityStore(self.state)
        self.grants = grants or GrantStore(self.state)
        self.headless = bool(headless)
        self._now = now
        self._pending: dict[str, _Pending] = {}
        self._decisions: dict[str, dict[str, Any]] = {}

    def _audit(self, event: str, **fields: Any) -> None:
        AuditLog(self.state.tool_audit_file()).append(
            {"event": event, **fields})

    def _deny(self, spec: ToolSpec, request: ToolRequest, reason: str,
              tier: str) -> dict[str, Any]:
        self._audit("tool.denied", session_id=self.session_id,
                    tool_identity=spec.identity, action=request.action,
                    reason=request.reason, args_sha256=_json_hash(request.args),
                    decision=reason, tier=tier)
        return {"status": "denied", "reason": reason,
                "tool_identity": spec.identity, "session_id": self.session_id}

    def authorize(self, spec: ToolSpec | Mapping[str, Any],
                  request: ToolRequest, *,
                  headless: bool | None = None) -> dict[str, Any]:
        spec = spec if isinstance(spec, ToolSpec) else ToolSpec.from_dict(spec)
        if request.session_id != self.session_id:
            raise SecurityError("request session does not match the gateway")
        headless = self.headless if headless is None else bool(headless)
        policy = self.security.read()
        tier = str(policy.get("tier") or DEFAULT_TIER)
        reason_ok = _safe_reason(request.reason)

        if spec.is_side_effect and spec.needs_reason and not reason_ok:
            return self._deny(spec, request, "missing reason", tier)
        if not spec.is_side_effect:
            self._audit("tool.approved", session_id=self.session_id,
                        tool_identity=spec.identity, action=request.action,
                        args_sha256=_json_hash(request.args),
                        decision="read-only", tier=tier)
            return {"status": "approved", "reason": "read-only",
                    "tool_identity": spec.identity, "session_id": self.session_id}
        if headless:
            return self._deny(spec, request, "headless side-effect", tier)

        granted = self.grants.has(spec.identity, self.session_id)
        private = spec.network == "private"
        if granted and tier in ("guarded", "trusted", "open"):
            self._audit("tool.approved", session_id=self.session_id,
                        tool_identity=spec.identity, action=request.action,
                        args_sha256=_json_hash(request.args),
                        decision="grant", tier=tier)
            return {"status": "approved", "reason": "grant",
                    "tool_identity": spec.identity, "session_id": self.session_id}
        if tier == "open" and not private:
            self._audit("tool.approved", session_id=self.session_id,
                        tool_identity=spec.identity, action=request.action,
                        args_sha256=_json_hash(request.args),
                        decision="open tier", tier=tier)
            return {"status": "approved", "reason": "open tier",
                    "tool_identity": spec.identity, "session_id": self.session_id}

        request_id = f"req-{uuid.uuid4().hex[:20]}"
        now = self._now()
        expires = now + request.timeout_s
        self._pending[request_id] = _Pending(
            request_id, spec, request, now, expires)
        self._audit("tool.pending", request_id=request_id,
                    session_id=self.session_id, tool_identity=spec.identity,
                    action=request.action, reason=request.reason,
                    risk=spec.risk, network=spec.network,
                    filesystem=spec.filesystem, sandbox=spec.sandbox,
                    args_sha256=_json_hash(request.args), tier=tier,
                    expires_at=expires)
        return {
            "status": "pending", "request_id": request_id,
            "expires_at": expires, "tool_identity": spec.identity,
            "session_id": self.session_id, "tier": tier,
            "risk": spec.risk, "network": spec.network,
            "filesystem": spec.filesystem, "sandbox": spec.sandbox,
            "reason": request.reason, "action": request.action,
            "args": dict(request.args),
        }

    def decide(self, request_id: str, choice: str, *,
               scope: str = "session") -> dict[str, Any]:
        pending = self._pending.pop(request_id, None)
        if pending is None:
            raise ApprovalError(f"no pending request {request_id!r}")
        now = self._now()
        if now >= pending.expires_at:
            self._audit("tool.expired", request_id=request_id,
                        session_id=self.session_id,
                        tool_identity=pending.spec.identity)
            result = {"status": "denied", "reason": "timeout",
                      "request_id": request_id}
            self._decisions[request_id] = result
            return result
        choice = str(choice or "").strip().lower()
        if choice == "deny":
            self._audit("tool.denied", request_id=request_id,
                        session_id=self.session_id,
                        tool_identity=pending.spec.identity,
                        reason="user denied")
            result = {"status": "denied", "reason": "user denied",
                      "request_id": request_id}
            self._decisions[request_id] = result
            return result
        if choice not in ("once", "session", "always"):
            raise ApprovalError("choice must be once, session, always, or deny")
        if choice == "session":
            self.grants.grant(pending.spec.identity, scope="session",
                              session_id=self.session_id,
                              reason=pending.request.reason)
        elif choice == "always":
            if scope not in ("global", "session"):
                raise ApprovalError("always requires scope=global or scope=session")
            self.grants.grant(pending.spec.identity, scope=scope,
                              session_id=self.session_id,
                              reason=pending.request.reason)
        self._audit("tool.approved", request_id=request_id,
                    session_id=self.session_id,
                    tool_identity=pending.spec.identity,
                    decision=choice, grant_scope=scope if choice == "always" else None)
        result = {"status": "approved", "reason": choice,
                  "request_id": request_id,
                  "tool_identity": pending.spec.identity}
        self._decisions[request_id] = result
        return result

    def decision(self, request_id: str) -> dict[str, Any] | None:
        """The settled decision for a request, if the hook is still waiting."""
        return self._decisions.get(request_id)

    def pending(self) -> list[dict[str, Any]]:
        now = self._now()
        expired = [request_id for request_id, item in self._pending.items()
                   if now >= item.expires_at]
        for request_id in expired:
            self._pending.pop(request_id, None)
        return [{
            "request_id": item.request_id,
            "session_id": self.session_id,
            "tool_identity": item.spec.identity,
            "action": item.request.action,
            "reason": item.request.reason,
            "args_sha256": _json_hash(item.request.args),
            "created_at": item.created_at,
            "expires_at": item.expires_at,
            "risk": item.spec.risk,
            "network": item.spec.network,
            "filesystem": item.spec.filesystem,
            "sandbox": item.spec.sandbox,
        } for item in self._pending.values()]

    def execute(self, spec: ToolSpec | Mapping[str, Any],
                request: ToolRequest,
                executor: Callable[[ToolRequest], Any], *,
                headless: bool | None = None) -> dict[str, Any]:
        decision = self.authorize(spec, request, headless=headless)
        if decision["status"] != "approved":
            return decision
        try:
            result = executor(request)
        except Exception as exc:  # noqa: BLE001 - a tool error is structured
            self._audit("tool.result", session_id=self.session_id,
                        tool_identity=decision.get("tool_identity"),
                        status="error", error=str(exc)[:1000])
            return {**decision, "status": "error", "error": str(exc)}
        self._audit("tool.result", session_id=self.session_id,
                    tool_identity=decision.get("tool_identity"),
                    status="ok", result_sha256=_json_hash(result))
        return {**decision, "result": result}
