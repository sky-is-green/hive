"""Durable activity ledger, tripwire rules, and emergency cease control.

Hive's agent stream already carries shaped activity events.  This module makes
them inspectable and actionable:

* :class:`ActivityLedger` appends every event to a JSONL file and publishes it
  to live subscribers.  The file is the replay/audit source; the subscribers
  are the Studio's live feed and the external sentinel.
* :class:`ActivityRules` matches events against operator-supplied tripwires.
  Rules are disabled by default; a rule may warn, revoke session grants, or
  cease the turn.
* :class:`CeaseStore` is the durable "this conversation may not use tools"
  flag.  A cease is not a pause: the runtime is force-stopped, session grants
  are revoked, and the flag must be cleared explicitly.
* :class:`ActivityHub` ties the three together and exposes the callback the
  app uses to kill the runtime and its sandbox containers.

The ledger is deliberately separate from the approval audit.  Approval records
deliberately hash arguments; the activity ledger stores the concrete tool input
and output because the user asked to see exactly what the agent did.  It is
still redacted through :func:`harness.state_store.redact` and lives under the
AI user's state directory.
"""

from __future__ import annotations

import json
import queue
import re
import threading
from collections import deque
from datetime import datetime, timezone
from itertools import count
from typing import Any, Callable, Iterable, Mapping, Optional

from harness.state_store import AuditLog, JsonStore, StatePaths, redact

DEFAULT_RULES: dict[str, Any] = {
    "schema_version": 1,
    # Off by default: an automatic stop is a policy decision the operator
    # should make explicitly.  The examples in docs/ACTIVITY-CONTROL.md show
    # conservative rules to enable.
    "enabled": False,
    "rules": [],
}

DEFAULT_CEASE: dict[str, Any] = {
    "schema_version": 1,
    "blocks": [],
}

_RULE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")
_ACTIONS = ("warn", "revoke", "cease")
_FIELDS = ("any", "text", "args", "result", "output", "action")
_MATCH_KEYS = frozenset({"type", "phase", "tool", "field", "regex", "contains"})
_MAX_REGEX = 2000


class ActivityError(RuntimeError):
    """An activity rule, ledger record, or cease request is malformed."""


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(
        timespec="seconds").replace("+00:00", "Z")


def _month() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m")


def _event_text(event: Mapping[str, Any], field: str) -> str:
    """Canonical text a rule searches for one event field."""
    if field == "args":
        value = event.get("args")
    elif field == "result":
        value = event.get("result")
    elif field == "output":
        value = event.get("output")
    elif field == "text":
        value = event.get("text")
    elif field == "action":
        value = event.get("action")
    else:
        value = None
    if field != "any" and value is None:
        return ""
    if field == "any":
        value = event
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


def _validate_rule(raw: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ActivityError("each rule must be an object")
    rule_id = str(raw.get("id") or "").strip()
    if not _RULE_ID.match(rule_id):
        raise ActivityError(
            "rule id must be 1-81 letters, digits, '.', '_' or '-'")
    action = str(raw.get("action") or "warn").strip().lower()
    if action not in _ACTIONS:
        raise ActivityError(f"rule action must be one of {', '.join(_ACTIONS)}")
    match = raw.get("match")
    if not isinstance(match, Mapping) or not match:
        raise ActivityError(f"rule {rule_id!r} needs a non-empty match object")
    unknown = set(match) - _MATCH_KEYS
    if unknown:
        raise ActivityError(
            f"rule {rule_id!r} has unknown match keys: {', '.join(sorted(unknown))}")
    normalized_match: dict[str, str] = {}
    for key, value in match.items():
        text = str(value or "").strip()
        if not text:
            continue
        if key == "field":
            if text not in _FIELDS:
                raise ActivityError(
                    f"rule {rule_id!r} field must be one of {', '.join(_FIELDS)}")
        if key == "regex":
            if len(text) > _MAX_REGEX:
                raise ActivityError(
                    f"rule {rule_id!r} regex is longer than {_MAX_REGEX} chars")
            try:
                re.compile(text)
            except re.error as exc:
                raise ActivityError(
                    f"rule {rule_id!r} has an invalid regex: {exc}") from exc
        normalized_match[key] = text
    if not normalized_match:
        raise ActivityError(f"rule {rule_id!r} match is empty")
    return {
        "id": rule_id,
        "action": action,
        "enabled": bool(raw.get("enabled", True)),
        "match": normalized_match,
        "reason": str(raw.get("reason") or "").strip()[:500],
        "note": str(raw.get("note") or "").strip()[:1000],
    }


class ActivityRules:
    """Operator-editable tripwire rules under ``settings/activity-rules.json``."""

    def __init__(self, state: StatePaths | None = None) -> None:
        self.state = (state or StatePaths.default()).ensure()
        self._store = JsonStore(self.state.settings_file("activity-rules"),
                                default=DEFAULT_RULES)
        self._cache: dict[str, Any] | None = None
        self._cache_mtime: int | None = None

    def _mtime(self) -> int | None:
        try:
            return self._store.path.stat().st_mtime_ns
        except OSError:
            return None

    def read(self) -> dict[str, Any]:
        mtime = self._mtime()
        if self._cache is not None and mtime == self._cache_mtime:
            return self._cache
        raw = self._store.read()
        out = {
            "schema_version": 1,
            "enabled": bool(raw.get("enabled", False)),
            "rules": [],
        }
        for item in raw.get("rules") or []:
            try:
                out["rules"].append(_validate_rule(item))
            except ActivityError:
                # A corrupt rule must not disable the whole feed.  Keep the
                # good rules and let the operator see the count.
                continue
        out["invalid"] = max(
            0, len(raw.get("rules") or []) - len(out["rules"]))
        self._cache = out
        self._cache_mtime = mtime
        return out

    def write(self, rules: Iterable[Mapping[str, Any]], *,
              enabled: bool | None = None) -> dict[str, Any]:
        normalized = [_validate_rule(rule) for rule in rules]
        payload = {
            "schema_version": 1,
            "enabled": True if enabled is None else bool(enabled),
            "rules": normalized,
        }
        self._store.write(payload)
        self._cache = {**payload, "invalid": 0}
        self._cache_mtime = self._mtime()
        AuditLog(self.state.audit_file()).append({
            "event": "activity.rules",
            "enabled": payload["enabled"],
            "rules": [rule["id"] for rule in normalized],
        })
        return payload

    def set_enabled(self, enabled: bool) -> dict[str, Any]:
        current = self.read()
        return self.write(current["rules"], enabled=enabled)

    def matches(self, event: Mapping[str, Any]) -> list[dict[str, Any]]:
        current = self.read()
        if not current["enabled"]:
            return []
        hits: list[dict[str, Any]] = []
        for rule in current["rules"]:
            if not rule["enabled"]:
                continue
            match = rule["match"]
            if match.get("type") not in (None, "*") \
                    and event.get("type") != match["type"]:
                continue
            if match.get("phase") not in (None, "*") \
                    and event.get("phase") != match["phase"]:
                continue
            if match.get("tool") not in (None, "*") \
                    and event.get("tool") != match["tool"]:
                continue
            field = match.get("field") or "any"
            text = _event_text(event, field)
            if not text:
                continue
            regex = match.get("regex")
            contains = match.get("contains")
            if regex and not re.search(regex, text, flags=re.IGNORECASE | re.DOTALL):
                continue
            if contains and contains.lower() not in text.lower():
                continue
            if not regex and not contains:
                # Type/tool/phase-only rules are valid but should be rare;
                # require at least one content predicate for safety.
                continue
            hits.append({"rule": rule, "field": field})
        return hits


def _session_family(session_id: str) -> str:
    """Strip a runtime generation suffix (``agent-c1-g3`` -> ``agent-c1``)."""
    return re.sub(r"-g\d+$", "", str(session_id or ""))


class CeaseStore:
    """Durable per-conversation/per-session tool block."""

    def __init__(self, state: StatePaths | None = None) -> None:
        self.state = (state or StatePaths.default()).ensure()
        self._store = JsonStore(self.state.settings_file("activity-cease"),
                                default=DEFAULT_CEASE)

    def _rows(self) -> list[dict[str, Any]]:
        rows = self._store.read().get("blocks") or []
        return [dict(row) for row in rows if isinstance(row, dict)]

    def block(self, *, session_id: str = "", conversation_id: str = "",
              reason: str = "", source: str = "user",
              rule: str = "") -> dict[str, Any]:
        session_id = str(session_id or "").strip()[:200]
        conversation_id = str(conversation_id or "").strip()[:200]
        if not session_id and not conversation_id:
            raise ActivityError("cease needs a session or conversation id")
        row = {
            "session_id": session_id,
            "conversation_id": conversation_id,
            "reason": str(reason or "cease requested")[:1000],
            "source": str(source or "user")[:80],
            "rule": str(rule or "")[:120],
            "created_at": _utc(),
        }

        def update(state: dict[str, Any]) -> dict[str, Any]:
            family = _session_family(session_id)
            rows = [item for item in (state.get("blocks") or [])
                    if isinstance(item, dict)
                    and not (
                        (session_id and (
                            item.get("session_id") == session_id
                            or (_session_family(item.get("session_id"))
                                == family and family)))
                        or (conversation_id
                            and item.get("conversation_id") == conversation_id)
                    )]
            rows.append(row)
            return {"schema_version": 1, "blocks": rows}

        self._store.update(update)
        AuditLog(self.state.audit_file()).append({
            "event": "activity.cease.block", **row,
        })
        return dict(row)

    def clear(self, *, session_id: str = "",
              conversation_id: str = "") -> int:
        session_id = str(session_id or "").strip()[:200]
        conversation_id = str(conversation_id or "").strip()[:200]
        if not session_id and not conversation_id:
            raise ActivityError("resume needs a session or conversation id")
        removed = {"count": 0}

        def update(state: dict[str, Any]) -> dict[str, Any]:
            family = _session_family(session_id)
            before = [item for item in (state.get("blocks") or [])
                      if isinstance(item, dict)]
            after = [item for item in before
                     if not (
                         (session_id and (
                             item.get("session_id") == session_id
                             or (_session_family(item.get("session_id"))
                                 == family and family)))
                         or (conversation_id
                             and item.get("conversation_id") == conversation_id)
                     )]
            removed["count"] = len(before) - len(after)
            return {"schema_version": 1, "blocks": after}

        self._store.update(update)
        AuditLog(self.state.audit_file()).append({
            "event": "activity.cease.clear",
            "session_id": session_id,
            "conversation_id": conversation_id,
            "removed": removed["count"],
        })
        return removed["count"]

    def is_blocked(self, *, session_id: str = "",
                   conversation_id: str = "") -> bool:
        session_id = str(session_id or "").strip()[:200]
        conversation_id = str(conversation_id or "").strip()[:200]
        family = _session_family(session_id)
        return any(
            (session_id and (
                row.get("session_id") == session_id
                or (_session_family(row.get("session_id")) == family
                    and family)))
            or (conversation_id
                and row.get("conversation_id") == conversation_id)
            for row in self._rows()
        )

    def list(self) -> list[dict[str, Any]]:
        return self._rows()


class ActivityLedger:
    """Append-only JSONL activity history plus an in-process live feed."""

    def __init__(self, state: StatePaths | None = None, *,
                 max_memory: int = 2000) -> None:
        self.state = (state or StatePaths.default()).ensure()
        self._lock = threading.Lock()
        self._events: deque[dict[str, Any]] = deque(maxlen=max_memory)
        self._subscribers: set[queue.Queue] = set()
        self._seq = count(1)

    def path(self, month: str | None = None):
        return self.state.audit_dir / f"activity-{month or _month()}.jsonl"

    def record(self, event: Mapping[str, Any], *, conversation_id: str = "",
               session_id: str = "", source: str = "agent") -> dict[str, Any]:
        if not isinstance(event, Mapping):
            raise ActivityError("activity event must be an object")
        row = dict(event)
        row.setdefault("ts", _utc())
        row["seq"] = next(self._seq)
        row["conversation_id"] = str(conversation_id or "")
        row["session_id"] = str(session_id or "")
        row["source"] = str(source or "agent")
        row = redact(row)
        AuditLog(self.path()).append(row)
        self._publish(row)
        return row

    def _publish(self, row: dict[str, Any]) -> None:
        with self._lock:
            self._events.append(row)
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            try:
                subscriber.put_nowait(row)
            except queue.Full:
                # A slow viewer must not back up the agent.  Drop the oldest
                # item in its queue and keep the newest event.
                try:
                    subscriber.get_nowait()
                    subscriber.put_nowait(row)
                except (queue.Empty, queue.Full):
                    pass

    def recent(self, conversation_id: str | None = None, *,
               limit: int = 200) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 2000))
        with self._lock:
            rows = list(self._events)
        if conversation_id:
            rows = [row for row in rows
                    if row.get("conversation_id") == conversation_id]
        return rows[-limit:]

    def read_file(self, conversation_id: str | None = None, *,
                  limit: int = 500) -> list[dict[str, Any]]:
        """Read the newest records from the current month's file."""
        path = self.path()
        if not path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
        if conversation_id:
            rows = [row for row in rows
                    if row.get("conversation_id") == conversation_id]
        return rows[-max(1, min(int(limit), 5000)):]

    def subscribe(self) -> queue.Queue:
        subscriber: queue.Queue = queue.Queue(maxsize=1000)
        with self._lock:
            self._subscribers.add(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: queue.Queue) -> None:
        with self._lock:
            self._subscribers.discard(subscriber)


class ActivityHub:
    """Record events, evaluate tripwires, and perform an emergency cease."""

    def __init__(self, state: StatePaths | None = None, *,
                 ledger: ActivityLedger | None = None,
                 rules: ActivityRules | None = None,
                 cease: CeaseStore | None = None,
                 grants: object | None = None) -> None:
        self.state = (state or StatePaths.default()).ensure()
        self.ledger = ledger or ActivityLedger(self.state)
        self.rules = rules or ActivityRules(self.state)
        self.cease = cease or CeaseStore(self.state)
        #: The app's live GrantStore (session grants are in memory).  When it
        #: is not wired yet, a fresh store still handles durable grants.
        self.grants = grants
        self._on_cease: Optional[Callable[..., dict[str, Any]]] = None
        self._active: dict[str, str] = {}
        self._sinks: dict[str, Callable[[dict[str, Any]], None]] = {}
        self._lock = threading.Lock()

    # -- live session wiring ------------------------------------------------
    def bind(self, conversation_id: str, session_id: str,
             sink: Callable[[dict[str, Any]], None] | None = None) -> None:
        with self._lock:
            self._active[conversation_id] = session_id
            if sink is not None:
                self._sinks[session_id] = sink

    def unbind(self, conversation_id: str, session_id: str) -> None:
        with self._lock:
            if self._active.get(conversation_id) == session_id:
                self._active.pop(conversation_id, None)
            self._sinks.pop(session_id, None)

    def session_for(self, conversation_id: str) -> str:
        with self._lock:
            return self._active.get(conversation_id, "")

    def is_blocked(self, *, conversation_id: str = "",
                   session_id: str = "") -> bool:
        return self.cease.is_blocked(
            conversation_id=conversation_id, session_id=session_id)

    # -- event path ---------------------------------------------------------
    def record(self, event: Mapping[str, Any], *, conversation_id: str = "",
               session_id: str = "", source: str = "agent",
               evaluate: bool = True) -> dict[str, Any]:
        row = self.ledger.record(
            event, conversation_id=conversation_id,
            session_id=session_id, source=source)
        if not evaluate:
            return row
        for hit in self.rules.matches(row):
            rule = hit["rule"]
            self.ledger.record({
                "type": "tripwire",
                "rule": rule["id"],
                "action": rule["action"],
                "reason": rule.get("reason") or "",
                "hit_field": hit["field"],
                "hit": row,
            }, conversation_id=conversation_id, session_id=session_id,
                source="tripwire")
            if rule["action"] == "revoke":
                self.revoke_session(session_id)
            elif rule["action"] == "cease":
                self.cease_now(
                    conversation_id=conversation_id,
                    session_id=session_id,
                    reason=rule.get("reason") or f"tripwire {rule['id']}",
                    source="tripwire",
                    rule=rule["id"],
                    revoke_grants=True,
                )
                break
        return row

    # -- cease / resume -----------------------------------------------------
    def register_cease_callback(
            self, callback: Callable[..., dict[str, Any]]) -> None:
        self._on_cease = callback

    def revoke_session(self, session_id: str) -> dict[str, int]:
        from harness.security import GrantStore

        if not session_id:
            return {"session": 0, "global": 0}
        store = self.grants if self.grants is not None else GrantStore(self.state)
        return store.revoke_session(session_id)

    def cease_now(self, *, conversation_id: str = "", session_id: str = "",
                  reason: str = "cease requested", source: str = "user",
                  rule: str = "", revoke_grants: bool = True,
                  revoke_all: bool = False,
                  force: bool = True) -> dict[str, Any]:
        conversation_id = str(conversation_id or "").strip()[:200]
        session_id = str(session_id or "").strip()[:200]
        if not session_id and conversation_id:
            session_id = self.session_for(conversation_id)
        if not session_id and not conversation_id:
            raise ActivityError("cease needs a session or conversation id")

        block = self.cease.block(
            session_id=session_id, conversation_id=conversation_id,
            reason=reason, source=source, rule=rule)
        actions: dict[str, Any] = {"block": block}

        # Tell the live SSE stream immediately.  The stop callback can spend
        # time killing containers; the user should see the cease frame first.
        # This is a raw sink, not the recording path, so it cannot recurse.
        with self._lock:
            sink = self._sinks.get(session_id)
        if sink is not None:
            try:
                sink({"type": "cease", "reason": reason,
                      "source": source, "rule": rule})
            except Exception:  # noqa: BLE001 - the block is the contract
                pass

        if self._on_cease is not None:
            try:
                actions["stop"] = self._on_cease(
                    conversation_id=conversation_id,
                    session_id=session_id,
                    reason=reason,
                    force=force,
                )
            except Exception as exc:  # noqa: BLE001 - the block is the contract
                actions["stop_error"] = str(exc)

        if revoke_all:
            from harness.security import GrantStore

            store = self.grants if self.grants is not None else GrantStore(self.state)
            actions["grants"] = store.revoke(None)
        elif revoke_grants and session_id:
            actions["grants"] = self.revoke_session(session_id)

        record = self.ledger.record({
            "type": "cease",
            "reason": reason,
            "source": source,
            "rule": rule,
            "actions": actions,
        }, conversation_id=conversation_id, session_id=session_id,
            source="cease")
        return {"ok": True, "block": block, "actions": actions,
                "event": record}

    def resume(self, *, conversation_id: str = "", session_id: str = "",
               source: str = "user") -> dict[str, Any]:
        conversation_id = str(conversation_id or "").strip()[:200]
        session_id = str(session_id or "").strip()[:200]
        if not session_id and conversation_id:
            session_id = self.session_for(conversation_id)
        removed = self.cease.clear(
            session_id=session_id, conversation_id=conversation_id)
        record = self.ledger.record({
            "type": "resume",
            "source": source,
            "removed": removed,
        }, conversation_id=conversation_id, session_id=session_id,
            source="resume")
        return {"ok": True, "removed": removed, "event": record}

    def status(self) -> dict[str, Any]:
        return {
            "rules": self.rules.read(),
            "blocks": self.cease.list(),
            "recent": self.ledger.recent(limit=20),
        }
