"""A small, explicit HTTP fetcher with per-hop SSRF checks and file gates.

This is not a browser: it does not run JavaScript, keep cookies, or reuse a
profile.  It is the safe public/private network primitive that a browser
adapter can sit on later.  Redirects are followed manually so every hop is
classified before a socket is opened.
"""

from __future__ import annotations

import hashlib
import http.client
import mimetypes
import ssl
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urljoin

from harness.net_policy import UrlError, classify_url, origin_of, redirect_url
from harness.state_store import AuditLog, JsonStore, StatePaths

DEFAULT_POLICY: dict[str, Any] = {
    "schema_version": 1,
    "max_bytes": 2_000_000,
    "max_upload_bytes": 8_000_000,
    "max_redirects": 5,
    "timeout_s": 20,
    "allow_public": True,
    "allow_downloads": False,
    "allow_uploads": False,
    "allowed_content_types": [
        "text/html", "text/plain", "text/markdown", "text/xml",
        "application/json", "application/xml", "application/xhtml+xml",
        "application/ld+json",
    ],
    "updated_at": None,
}
DEFAULT_GRANTS = {"schema_version": 1, "grants": []}
ACCESS = ("public", "private")
SCOPES = ("global", "session")
SAFE_REQUEST_HEADERS = frozenset({
    "accept", "accept-language", "content-type", "user-agent",
})
SAFE_RESPONSE_HEADERS = frozenset({
    "content-type", "content-length", "last-modified", "etag",
})
_BLOCKED_HEADERS = frozenset({
    "authorization", "cookie", "host", "proxy-authorization",
    "proxy-connection", "te", "trailer", "transfer-encoding",
    "upgrade", "via", "x-forwarded-for", "x-forwarded-host",
    "x-forwarded-proto", "x-real-ip",
})


class WebFetchError(RuntimeError):
    """A fetch was refused or failed with a structured reason."""

    def __init__(self, message: str, *, code: str = "web_fetch_error",
                 status: int = 400, detail: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.status = int(status)
        self.detail = dict(detail or {})


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(
        timespec="seconds").replace("+00:00", "Z")


class WebFetchPolicy:
    """Read/write the fetch limits and the download/upload switches."""

    def __init__(self, state: StatePaths | None = None) -> None:
        self.state = state or StatePaths.default()
        self.store = JsonStore(self.state.settings_file("browser"),
                               default=DEFAULT_POLICY)

    @staticmethod
    def _normalize(data: Mapping[str, Any]) -> dict[str, Any]:
        out = {**DEFAULT_POLICY, **data}
        try:
            out["max_bytes"] = max(1, min(50_000_000, int(out["max_bytes"])))
            out["max_upload_bytes"] = max(
                1, min(100_000_000, int(out["max_upload_bytes"])))
            out["max_redirects"] = max(0, min(10, int(out["max_redirects"])))
            out["timeout_s"] = max(1, min(120, int(out["timeout_s"])))
        except (TypeError, ValueError) as exc:
            raise WebFetchError("fetch limits must be integers",
                                code="bad_policy") from exc
        out["allow_public"] = bool(out["allow_public"])
        out["allow_downloads"] = bool(out["allow_downloads"])
        out["allow_uploads"] = bool(out["allow_uploads"])
        types = out.get("allowed_content_types")
        if not isinstance(types, list):
            types = DEFAULT_POLICY["allowed_content_types"]
        out["allowed_content_types"] = [str(item).strip().lower()
                                        for item in types
                                        if str(item).strip()]
        return out

    def read(self) -> dict[str, Any]:
        return self._normalize({**DEFAULT_POLICY, **self.store.read()})

    def update(self, **fields: Any) -> dict[str, Any]:
        data = self.read()
        for key, value in fields.items():
            if value is not None:
                data[key] = value
        data["updated_at"] = _utc()
        data = self._normalize(data)
        self.store.write(data)
        return data

    def download_dir(self) -> Path:
        path = self.state.browser_dir / "downloads"
        path.mkdir(parents=True, exist_ok=True)
        try:
            path.chmod(0o700)
        except OSError:
            pass
        return path

    def upload_dir(self) -> Path:
        path = self.state.browser_dir / "uploads"
        path.mkdir(parents=True, exist_ok=True)
        try:
            path.chmod(0o700)
        except OSError:
            pass
        return path


_SHARED_SESSIONS: dict[tuple[str, str, str, str], dict[str, Any]] = {}


class BrowserGrantStore:
    """Global durable grants plus process-local session grants, by origin."""

    def __init__(self, state: StatePaths | None = None) -> None:
        self.state = state or StatePaths.default()
        self._root = str(Path(self.state.root).expanduser().resolve())
        self._store = JsonStore(self.state.browser_grants_file(),
                                default=DEFAULT_GRANTS)
        self._sessions = _SHARED_SESSIONS

    @staticmethod
    def _origin(origin: str) -> str:
        try:
            return origin_of(str(origin or "").strip())
        except UrlError as exc:
            raise WebFetchError(str(exc), code="bad_origin") from exc

    def _expired(self, row: Mapping[str, Any], now: float) -> bool:
        value = row.get("expires_at")
        if value is None:
            return False
        try:
            if isinstance(value, (int, float)):
                return float(value) <= now
            return datetime.fromisoformat(
                str(value).replace("Z", "+00:00")).timestamp() <= now
        except (TypeError, ValueError):
            return True

    def grant(self, origin: str, *, access: str, scope: str = "session",
              session_id: str | None = None, reason: str = "",
              expires_at: Any = None) -> dict[str, Any]:
        origin = self._origin(origin)
        access = str(access or "").strip().lower()
        if access not in ACCESS:
            raise WebFetchError("access must be public or private",
                                code="bad_access")
        if scope not in SCOPES:
            raise WebFetchError("scope must be global or session",
                                code="bad_scope")
        row = {
            "origin": origin, "access": access, "scope": scope,
            "session_id": session_id if scope == "session" else None,
            "granted_at": _utc(), "expires_at": expires_at, "reason": reason,
        }
        if scope == "session":
            if not session_id:
                raise WebFetchError("session grant needs a session_id",
                                    code="missing_session")
            self._sessions[(self._root, str(session_id), origin, access)] = row
        else:
            def update(state: dict[str, Any]) -> dict[str, Any]:
                rows = [r for r in (state.get("grants") or [])
                        if isinstance(r, dict)
                        and not (r.get("origin") == origin
                                 and r.get("access") == access)]
                rows.append(row)
                return {"schema_version": 1, "grants": rows}

            self._store.update(update)
        AuditLog(self.state.audit_file()).append(
            {"event": "web.grant", **row})
        return dict(row)

    def has(self, origin: str, access: str,
            session_id: str | None = None) -> bool:
        try:
            origin = self._origin(origin)
        except WebFetchError:
            return False
        access = str(access or "").strip().lower()
        if session_id and (self._root, str(session_id), origin,
                           access) in self._sessions:
            return True
        now = time.time()
        return any(row.get("origin") == origin
                   and row.get("access") == access
                   and not self._expired(row, now)
                   for row in self._rows())

    def _rows(self) -> list[dict[str, Any]]:
        return [dict(row) for row in
                (self._store.read().get("grants") or [])
                if isinstance(row, dict)]

    def revoke(self, origin: str | None = None, *,
               session_id: str | None = None) -> dict[str, int]:
        canonical = None
        if origin is not None:
            canonical = self._origin(origin)
        session_keys = [key for key in self._sessions
                        if key[0] == self._root
                        and (not session_id or key[1] == session_id)
                        and (canonical is None or key[2] == canonical)]
        for key in session_keys:
            del self._sessions[key]
        removed_global = 0

        def update(state: dict[str, Any]) -> dict[str, Any]:
            nonlocal removed_global
            before = [r for r in (state.get("grants") or [])
                      if isinstance(r, dict)]
            after = [r for r in before
                     if canonical is not None and r.get("origin") != canonical]
            if canonical is None:
                after = []
            removed_global = len(before) - len(after)
            return {"schema_version": 1, "grants": after}

        self._store.update(update)
        AuditLog(self.state.audit_file()).append(
            {"event": "web.revoke", "origin": canonical,
             "session_id": session_id, "global_removed": removed_global})
        return {"session": len(session_keys), "global": removed_global}

    def list(self, session_id: str | None = None) -> list[dict[str, Any]]:
        now = time.time()
        rows = [row for row in self._rows()
                if not self._expired(row, now)]
        rows += [dict(row) for (root, sid, _origin, _access), row
                 in sorted(self._sessions.items())
                 if root == self._root
                 and (session_id is None or sid == session_id)]
        return rows


def _safe_headers(raw: Mapping[str, Any] | None) -> dict[str, str]:
    headers: dict[str, str] = {}
    for key, value in (raw or {}).items():
        name = str(key).strip().lower()
        if not name:
            continue
        if name in _BLOCKED_HEADERS:
            raise WebFetchError(f"request header {name!r} is not allowed",
                                code="blocked_header")
        if name not in SAFE_REQUEST_HEADERS:
            raise WebFetchError(
                f"request header {name!r} is not in the safe allowlist",
                code="blocked_header")
        headers[name] = str(value).strip()
    return headers


def _authorize(info, *, session_id: str | None, allow_private: bool,
               grants: BrowserGrantStore, policy: WebFetchPolicy,
               hop: int) -> None:
    data = policy.read()
    access = "private" if info.private else "public"
    if access == "public":
        if data["allow_public"] or grants.has(
                info.origin, "public", session_id):
            return
        raise WebFetchError(
            "public fetching is disabled by policy",
            code="public_blocked", status=403,
            detail={"origin": info.origin})
    if grants.has(info.origin, "private", session_id):
        return
    if hop == 0 and allow_private and session_id:
        grants.grant(info.origin, access="private", scope="session",
                     session_id=session_id,
                     reason="explicit private fetch approval")
        return
    raise WebFetchError(
        "private origin needs an explicit grant",
        code="private_origin", status=403,
        detail={"origin": info.origin, "url": info.url,
                "reason": info.reason,
                "requires_grant": True,
                "next": "retry with allow_private=true and a session_id"})


def _request(info, method: str, body: bytes | None,
             headers: Mapping[str, str], timeout: float,
             max_bytes: int) -> tuple[int, dict[str, str], bytes]:
    context = ssl.create_default_context() if info.scheme == "https" else None
    conn = (http.client.HTTPSConnection(
                info.host, info.port, timeout=timeout, context=context)
            if info.scheme == "https"
            else http.client.HTTPConnection(
                info.host, info.port, timeout=timeout))
    request_headers = {
        "user-agent": "hive-fetch/0.2",
        "accept": "*/*",
        "accept-encoding": "identity",
        "connection": "close",
        **headers,
    }
    if body is not None and "content-type" not in request_headers:
        request_headers["content-type"] = "application/octet-stream"
    try:
        conn.request(method, info.request_path, body=body,
                     headers=request_headers)
        response = conn.getresponse()
        status = int(response.status)
        response_headers = {key.lower(): value
                            for key, value in response.getheaders()}
        data = response.read(max_bytes + 1)
    except (OSError, http.client.HTTPException, ssl.SSLError) as exc:
        raise WebFetchError(f"request failed: {exc}", code="transport",
                            status=502) from exc
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass
    if len(data) > max_bytes:
        raise WebFetchError(
            f"response exceeds the {max_bytes}-byte limit",
            code="too_large", status=413,
            detail={"limit_bytes": max_bytes})
    return status, response_headers, data


def _write_download(name: str, data: bytes, directory: Path) -> dict[str, Any]:
    raw = str(name or "").strip()
    filename = Path(raw).name.strip()
    if (not filename or filename in (".", "..") or raw != filename):
        raise WebFetchError("download_name must be a plain filename",
                            code="bad_download_name")
    target = (directory / filename).resolve()
    root = directory.resolve()
    if target.parent != root:
        raise WebFetchError("download path escapes the download directory",
                            code="bad_download_name")
    candidate = target
    suffix = 1
    while candidate.exists():
        candidate = root / f"{target.stem}-{suffix}{target.suffix}"
        suffix += 1
        if suffix > 100:
            raise WebFetchError("too many files with that download name",
                                code="download_exists")
    with open(candidate, "xb") as handle:
        handle.write(data)
    try:
        candidate.chmod(0o600)
    except OSError:
        pass
    return {
        "name": candidate.name, "path": str(candidate),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def fetch(url: str, *, method: str = "GET", body: Any = None,
          headers: Mapping[str, Any] | None = None,
          session_id: str | None = None, allow_private: bool = False,
          allow_download: bool = False, download_name: str | None = None,
          upload_path: str | Path | None = None,
          allow_upload: bool = False,
          state: StatePaths | None = None) -> dict[str, Any]:
    """Fetch one URL with per-hop SSRF checks and explicit file gates."""
    state = state or StatePaths.default()
    policy = WebFetchPolicy(state)
    grants = BrowserGrantStore(state)
    data = policy.read()
    method = str(method or "GET").strip().upper()
    if method not in ("GET", "POST"):
        raise WebFetchError("method must GET or POST", code="bad_method")
    if method == "GET" and (body is not None or upload_path):
        raise WebFetchError("GET requests cannot carry a body",
                            code="bad_method")
    if body is not None and upload_path:
        raise WebFetchError("pass either body or upload_path, not both",
                            code="bad_request")
    session = str(session_id or "").strip()[:160] or None
    request_headers = _safe_headers(headers)
    body_bytes: bytes | None = None
    upload_meta: dict[str, Any] | None = None
    if upload_path:
        if not allow_upload or not data["allow_uploads"]:
            raise WebFetchError(
                "uploads need allow_upload=true and a policy that permits them",
                code="upload_blocked", status=403)
        root = policy.upload_dir().resolve()
        source = Path(upload_path).expanduser().resolve()
        if source.parent != root and root not in source.parents:
            raise WebFetchError(
                "upload_path must be inside the configured upload directory",
                code="upload_outside_root", status=403,
                detail={"root": str(root)})
        if not source.is_file():
            raise WebFetchError("upload_path does not exist",
                                code="upload_missing")
        body_bytes = source.read_bytes()
        if len(body_bytes) > data["max_upload_bytes"]:
            raise WebFetchError(
                f"upload exceeds the {data['max_upload_bytes']}-byte limit",
                code="upload_too_large", status=413)
        upload_meta = {"name": source.name, "bytes": len(body_bytes),
                       "sha256": hashlib.sha256(body_bytes).hexdigest()}
        request_headers.setdefault(
            "content-type",
            mimetypes.guess_type(source.name)[0] or "application/octet-stream")
    elif body is not None:
        if isinstance(body, str):
            body_bytes = body.encode("utf-8")
        elif isinstance(body, (bytes, bytearray)):
            body_bytes = bytes(body)
        else:
            raise WebFetchError("body must be text or bytes", code="bad_body")
        if len(body_bytes) > data["max_upload_bytes"]:
            raise WebFetchError(
                f"request body exceeds the {data['max_upload_bytes']}-byte limit",
                code="upload_too_large", status=413)
    if allow_download and not data["allow_downloads"]:
        raise WebFetchError(
            "downloads are disabled by policy", code="download_blocked",
            status=403)
    if allow_download and not download_name:
        raise WebFetchError(
            "download_name is required for an explicit download",
            code="download_name_required", status=400)

    current = str(url or "").strip()
    current_method = method
    current_body = body_bytes
    response_headers: dict[str, str] = {}
    payload = b""
    status = 0
    final_url = current
    for hop in range(int(data["max_redirects"]) + 1):
        try:
            info = classify_url(current)
        except UrlError as exc:
            raise WebFetchError(str(exc), code="bad_url") from exc
        _authorize(info, session_id=session, allow_private=allow_private,
                   grants=grants, policy=policy, hop=hop)
        status, response_headers, payload = _request(
            info, current_method, current_body, request_headers,
            float(data["timeout_s"]), int(data["max_bytes"]))
        final_url = info.url
        if status in (301, 302, 303, 307, 308):
            location = response_headers.get("location")
            if not location:
                raise WebFetchError("redirect has no Location header",
                                    code="bad_redirect", status=502)
            try:
                current = redirect_url(current, location)
            except UrlError as exc:
                raise WebFetchError(str(exc), code="bad_redirect",
                                    status=502) from exc
            if status in (301, 302, 303) and current_method != "GET":
                current_method = "GET"
                current_body = None
                request_headers.pop("content-type", None)
            continue
        break
    else:
        raise WebFetchError(
            f"too many redirects (limit {data['max_redirects']})",
            code="too_many_redirects", status=502)

    content_type = response_headers.get("content-type", "").split(";", 1)[0] \
        .strip().lower()
    disposition = response_headers.get("content-disposition", "").lower()
    if status in (204, 304) and not payload:
        return {
            "ok": True, "status": status, "url": final_url,
            "content_type": content_type, "body": "", "bytes": 0,
            "upload": upload_meta,
            "headers": {key: value for key, value in response_headers.items()
                        if key in SAFE_RESPONSE_HEADERS},
        }
    if allow_download:
        if not 200 <= status < 400:
            raise WebFetchError(
                f"upstream returned HTTP {status}", code="http_status",
                status=502, detail={"upstream_status": status})
        download = _write_download(download_name or "", payload,
                                   policy.download_dir())
        return {
            "ok": 200 <= status < 400, "status": status, "url": final_url,
            "content_type": content_type, "download": download,
            "upload": upload_meta,
            "headers": {key: value for key, value in response_headers.items()
                        if key in SAFE_RESPONSE_HEADERS},
            "bytes": len(payload),
        }
    if "attachment" in disposition:
        raise WebFetchError(
            "response is an attachment; retry with allow_download=true and "
            "a download_name", code="download_required", status=409)
    allowed = data["allowed_content_types"]
    if content_type not in allowed:
        raise WebFetchError(
            f"content type {content_type or 'missing'!r} is not allowed",
            code="content_type", status=415,
            detail={"content_type": content_type, "allowed": allowed})
    text = payload.decode("utf-8", errors="replace")
    return {
        "ok": 200 <= status < 400, "status": status, "url": final_url,
        "content_type": content_type, "body": text, "bytes": len(payload),
        "upload": upload_meta,
        "headers": {key: value for key, value in response_headers.items()
                    if key in SAFE_RESPONSE_HEADERS},
    }


def status(state: StatePaths | None = None) -> dict[str, Any]:
    state = state or StatePaths.default()
    policy = WebFetchPolicy(state)
    grants = BrowserGrantStore(state)
    return {
        "policy": policy.read(),
        "download_dir": str(policy.download_dir()),
        "upload_dir": str(policy.upload_dir()),
        "grants": grants.list(),
    }
