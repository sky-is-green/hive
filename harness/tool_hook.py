"""Fail-closed PreToolUse bridge from DSH to the Hive approval gateway.

DSH's hook bridge runs this as a command hook.  The hook posts the concrete
tool call to the local Hive gateway, then waits for a decision.  Exit 0 means
the tool may run; exit 2 blocks it and the reason on stderr is shown to the
model.  If Hive is unreachable or the decision times out, the call is blocked.

This is deliberately a small stdlib script: it must work inside the runtime
subprocess without importing the Studio or any model code.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.net_policy import UrlError, classify_url, origin_of

DEFAULT_URL = "http://127.0.0.1:8765"
DEFAULT_TIMEOUT_S = 300.0
READ_ONLY_TOOLS = frozenset({
    "read", "read_file", "readimage", "read_image", "list", "list_directory",
    "grep", "glob", "search", "fs_search", "todo", "skill", "web_search",
    "websearch", "web_fetch", "webfetch", "fetch", "browse", "http_get",
    "get_url", "inspect", "status",
})

#: Delegation tools whose child must start in a dedicated scratch workspace.
SUBAGENT_TOOLS = frozenset({"subagent", "subagent_fork"})
SUBAGENT_WORKSPACE_ROOT = Path("/srv/hive-ai/workspaces")


class HookError(RuntimeError):
    """The bridge could not obtain a trustworthy decision."""


def _tool_name(payload: dict[str, Any]) -> str:
    for key in ("tool_name", "toolName", "tool"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise HookError("hook payload has no tool name")


def _input(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("tool_input") or payload.get("toolInput") or {}
    return value if isinstance(value, dict) else {"value": value}


def _action(name: str, args: dict[str, Any]) -> str:
    for key in ("command", "cmd", "path", "file", "url", "query"):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return f"{name}: {value.strip()[:1200]}"
    try:
        detail = json.dumps(args, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        detail = str(args)
    return f"{name}: {detail[:1200]}"


def _urls_from_args(args: dict[str, Any]) -> list[str]:
    found: list[str] = []
    for key in ("url", "uri", "href", "target", "address", "urls"):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            found.append(value.strip())
        elif isinstance(value, list):
            found.extend(str(item).strip() for item in value
                         if isinstance(item, str) and item.strip())
    return found


def _url_is_private(url: str) -> bool:
    try:
        return classify_url(url).private
    except (UrlError, ValueError):
        return True


def _origin(url: str) -> str:
    try:
        return origin_of(url)
    except UrlError:
        return ""


def _resource_for(name: str, args: dict[str, Any]) -> str:
    low = name.lower().replace("-", "_")
    if not any(token in low for token in ("web", "fetch", "http", "browse")):
        return ""
    origins = [_origin(url) for url in _urls_from_args(args)]
    origins = sorted({origin for origin in origins if origin})
    if not origins:
        return ""
    return ",".join(origins)[:240]


def _network_for(name: str, args: dict[str, Any]) -> str:
    low = name.lower().replace("-", "_")
    web = any(token in low for token in ("web", "fetch", "http", "browse"))
    if not web:
        return "none"
    urls = _urls_from_args(args)
    if urls:
        return ("private" if any(_url_is_private(url) for url in urls)
                else "public")
    # Search-style calls carry a query, not an arbitrary destination.
    if any(key in args for key in ("query", "q", "search", "prompt")):
        return "public"
    # A web tool with no visible destination is not assumed public.
    return "private"


def _spec(name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    low = name.lower().replace("-", "_")
    if low.startswith("mcp__"):
        # MCP tool names are mcp__<server>__<raw>.  Until verified metadata is
        # registered, do not trust the name to imply read-only or public.
        return {
            "name": name, "source": "mcp", "version": "1.0.0",
            "description": f"MCP tool {name} (unregistered metadata)",
            "read_only": False, "side_effect": True, "reason_required": True,
            "network": "private", "filesystem": "host", "risk": "high",
            "sandbox": "required",
        }
    read_only = low in READ_ONLY_TOOLS
    network = _network_for(name, args or {})
    return {
        "name": name, "source": "dsh", "version": "1.0.0",
        "description": f"DSH tool {name}",
        "read_only": read_only,
        "side_effect": not read_only,
        "reason_required": not read_only,
        "network": network,
        "filesystem": "none" if read_only else "workspace",
        "risk": "high" if network == "private" else (
            "low" if read_only else ("high" if low in (
                "bash", "shell", "code_runtime", "subagent", "write_file")
                else "medium")),
        "sandbox": "optional" if read_only else "required",
    }


def _session(payload: dict[str, Any]) -> str:
    for key in ("session_id", "sessionId", "conversation_id"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:160]
    return "dsh-hook"


def subagent_workspace_error(name: str, args: dict[str, Any]) -> str:
    """Require an existing scratch workspace for every delegated child.

    The child's session cwd is its sandbox workspace root.  A missing or
    outside cwd would inherit the parent's ``/srv/hive-ai`` root, so the hook
    refuses before the approval gateway is asked.  The parent creates the
    directory with its own confined bash first.
    """
    if name not in SUBAGENT_TOOLS:
        return ""
    raw = args.get("cwd")
    if not isinstance(raw, str) or not raw.strip():
        return ("subagent calls need cwd under "
                f"{SUBAGENT_WORKSPACE_ROOT}; create a scratch directory first")
    try:
        target = Path(raw).expanduser().resolve()
        root = Path(SUBAGENT_WORKSPACE_ROOT).resolve()
    except (OSError, RuntimeError, ValueError):
        return f"subagent cwd is not usable: {raw!r}"
    try:
        relative = target.relative_to(root)
    except ValueError:
        return f"subagent cwd must be under {root}: {target}"
    if not relative.parts:
        return f"subagent cwd must be a subdirectory of {root}"
    if not target.is_dir():
        return f"subagent cwd does not exist: {target}; create it first"
    return ""


def _request(payload: dict[str, Any]) -> dict[str, Any]:
    name = _tool_name(payload)
    args = _input(payload)
    explicit = (payload.get("reason") or args.get("reason")
                or args.get("justification"))
    reason = (str(explicit).strip() if explicit
              else f"DSH requested {name}; review the displayed action")
    return {
        "session_id": _session(payload),
        "tool": _spec(name, args),
        "action": _action(name, args),
        "reason": reason,
        "args": args,
        "resource": _resource_for(name, args),
        "headless": os.environ.get("HIVE_TOOL_HEADLESS", "").strip().lower()
        in ("1", "true", "yes"),
        "timeout_s": float(os.environ.get("HIVE_TOOL_HOOK_TIMEOUT", "600")),
    }


def _http(url: str, method: str, body: dict[str, Any] | None = None
          ) -> dict[str, Any]:
    headers = {"content-type": "application/json"}
    token = os.environ.get("HARNESS_TOKEN", "").strip()
    if token:
        headers["x-splinter-token"] = token
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=4) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise HookError(f"Hive gateway returned HTTP {exc.code}") from exc
    except OSError as exc:
        raise HookError(f"Hive gateway is unreachable: {exc}") from exc
    try:
        result = json.loads(raw)
    except ValueError as exc:
        raise HookError("Hive gateway returned invalid JSON") from exc
    if not isinstance(result, dict):
        raise HookError("Hive gateway returned a non-object")
    return result


def _post(url: str, body: dict[str, Any]) -> dict[str, Any]:
    return _http(url, "POST", body)


def _get(url: str) -> dict[str, Any]:
    return _http(url, "GET")


def decide(payload: dict[str, Any], *,
           post: Callable[[str, dict[str, Any]], dict[str, Any]] = _post,
           get: Callable[[str], dict[str, Any]] = _get,
           sleep: Callable[[float], None] = time.sleep,
           timeout_s: float | None = None) -> dict[str, Any]:
    """Send one hook call and wait for its decision."""
    problem = subagent_workspace_error(_tool_name(payload), _input(payload))
    if problem:
        return {"status": "denied", "reason": problem}
    base = os.environ.get("HIVE_TOOL_URL", DEFAULT_URL).rstrip("/")
    request = _request(payload)
    result = post(f"{base}/v1/tools/call", request)
    status = str(result.get("status") or "")
    if status == "approved":
        return {"status": "approved", "request_id": result.get("request_id")}
    if status == "denied":
        return {"status": "denied", "reason": result.get("reason") or "denied"}
    if status != "pending":
        raise HookError(f"unknown gateway status {status!r}")
    request_id = str(result.get("request_id") or "")
    if not request_id:
        raise HookError("gateway returned pending without a request id")
    limit = float(timeout_s if timeout_s is not None
                  else os.environ.get("HIVE_TOOL_HOOK_TIMEOUT", DEFAULT_TIMEOUT_S))
    deadline = time.monotonic() + max(1.0, limit)
    while time.monotonic() < deadline:
        sleep(0.5)
        try:
            decided = get(
                f"{base}/v1/tools/decision?"
                + urllib.parse.urlencode({
                    "session_id": request["session_id"],
                    "request_id": request_id,
                }))
        except HookError:
            continue  # transient; the hard deadline remains
        if str(decided.get("status") or "") == "approved":
            return {"status": "approved", "request_id": request_id}
        if str(decided.get("status") or "") == "denied":
            return {"status": "denied", "reason": decided.get("reason")
                    or "denied", "request_id": request_id}
    raise HookError("approval timed out")


def main(argv: list[str] | None = None) -> int:
    try:
        payload = json.load(sys.stdin)
        if not isinstance(payload, dict):
            raise HookError("hook payload must be an object")
        outcome = decide(payload)
    except (HookError, OSError, ValueError, TypeError) as exc:
        print(f"blocked: {exc}", file=sys.stderr)
        return 2
    if outcome.get("status") == "approved":
        return 0
    print(f"blocked: {outcome.get('reason') or 'denied'}", file=sys.stderr)
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
