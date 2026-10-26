"""The secure fetch adapter: per-hop SSRF checks and explicit file gates."""

from __future__ import annotations

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from harness import net_policy, web_fetch
from harness.app import create_app
from harness.net_policy import classify_url
from harness.state_store import StatePaths


class _Handler(BaseHTTPRequestHandler):
    def _serve(self) -> None:
        length = int(self.headers.get("content-length", "0") or 0)
        body = self.rfile.read(length) if length else b""
        self.server.requests.append({  # type: ignore[attr-defined]
            "method": self.command,
            "path": self.path,
            "headers": dict(self.headers),
            "body": body,
        })
        route = self.server.routes.get(self.path)  # type: ignore[attr-defined]
        if route is None:
            status, headers, payload = 404, {"content-type": "text/plain"}, b"no"
        elif callable(route):
            status, headers, payload = route(self)
        else:
            status, headers, payload = route
        self.send_response(status)
        for key, value in headers.items():
            self.send_header(key, value)
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    do_GET = _serve
    do_POST = _serve

    def log_message(self, *args: Any) -> None:
        pass


class _Server:
    def __init__(self) -> None:
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.httpd.requests = []  # type: ignore[attr-defined]
        self.httpd.routes = {}  # type: ignore[attr-defined]
        self.thread = threading.Thread(target=self.httpd.serve_forever,
                                       daemon=True)

    def start(self) -> "_Server":
        self.thread.start()
        return self

    @property
    def origin(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_port}"

    @property
    def requests(self) -> list[dict[str, Any]]:
        return self.httpd.requests  # type: ignore[attr-defined]

    def route(self, path: str, status: int, body: bytes,
              headers: dict[str, str] | None = None) -> None:
        self.httpd.routes[path] = (  # type: ignore[attr-defined]
            status, headers or {"content-type": "text/plain"}, body)

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture()
def server():
    srv = _Server().start()
    try:
        yield srv
    finally:
        srv.close()


@pytest.fixture()
def state(tmp_path):
    return StatePaths(tmp_path / "state").ensure()


@pytest.fixture()
def client(tmp_path):
    app = create_app(
        providers_file=tmp_path / "providers.local.json",
        engines_file=tmp_path / "engines.local.json",
        runs_root=tmp_path / "runs",
        log_dir=str(tmp_path / "logs"),
        state_dir=str(tmp_path / "state"),
    )
    return TestClient(app)


def _grant_private(srv: _Server, state: StatePaths) -> None:
    web_fetch.BrowserGrantStore(state).grant(
        srv.origin, access="private", scope="global",
        reason="test server")


def test_private_literals_and_metadata_are_classified_private():
    for url in (
        "http://127.0.0.1:8765/server",
        "http://[::1]/",
        "http://169.254.169.254/latest/meta-data",
        "http://10.1.2.3/",
        "http://192.168.1.1/",
        "http://localhost:8080/",
        "http://metadata.google.internal/",
    ):
        info = classify_url(url)
        assert info.private is True, url


def test_public_classification_uses_resolved_addresses(monkeypatch):
    monkeypatch.setattr(
        net_policy.socket, "getaddrinfo",
        lambda host, port: [(2, 1, 6, "", ("93.184.216.34", 0))])
    info = classify_url("https://example.com/docs")
    assert info.private is False
    assert info.origin == "https://example.com"


def test_bad_urls_are_rejected():
    with pytest.raises(net_policy.UrlError):
        classify_url("ftp://example.com/file")
    with pytest.raises(net_policy.UrlError):
        classify_url("http://user:pass@example.com/")


def test_private_fetch_needs_an_explicit_grant(server, state):
    server.route("/text", 200, b"hello")
    with pytest.raises(web_fetch.WebFetchError) as exc:
        web_fetch.fetch(server.origin + "/text", session_id="s1",
                        state=state)
    assert exc.value.code == "private_origin"
    assert server.requests == []

    result = web_fetch.fetch(server.origin + "/text", session_id="s1",
                             allow_private=True, state=state)
    assert result["body"] == "hello"
    assert result["status"] == 200
    assert server.requests[0]["path"] == "/text"

    # The explicit approval became a session grant for this exact origin.
    result = web_fetch.fetch(server.origin + "/text", session_id="s1",
                             state=state)
    assert result["body"] == "hello"


def test_public_fetch_does_not_need_a_grant(server, state, monkeypatch):
    monkeypatch.setattr(
        net_policy, "_host_is_private", lambda host: (False, "public"))
    server.route("/text", 200, b"public")
    result = web_fetch.fetch(server.origin + "/text", state=state)
    assert result["body"] == "public"


def test_a_redirect_to_another_private_origin_is_blocked(server, state):
    other = _Server().start()
    try:
        server.route("/start", 302, b"", {
            "content-type": "text/plain",
            "location": other.origin + "/text"})
        other.route("/text", 200, b"other")
        _grant_private(server, state)
        with pytest.raises(web_fetch.WebFetchError) as exc:
            web_fetch.fetch(server.origin + "/start", state=state)
        assert exc.value.code == "private_origin"
        assert exc.value.detail["origin"] == other.origin
        assert other.requests == []

        _grant_private(other, state)
        result = web_fetch.fetch(server.origin + "/start", state=state)
        assert result["body"] == "other"
        assert other.requests
    finally:
        other.close()


def test_a_redirect_to_a_non_http_scheme_is_rejected(server, state):
    _grant_private(server, state)
    server.route("/start", 302, b"", {
        "content-type": "text/plain", "location": "file:///etc/passwd"})
    with pytest.raises(web_fetch.WebFetchError) as exc:
        web_fetch.fetch(server.origin + "/start", state=state)
    assert exc.value.code == "bad_redirect"


def test_a_download_needs_its_own_gate(server, state):
    _grant_private(server, state)
    server.route("/file", 200, b"payload", {
        "content-type": "application/octet-stream",
        "content-disposition": "attachment; filename=file.bin"})
    with pytest.raises(web_fetch.WebFetchError) as exc:
        web_fetch.fetch(server.origin + "/file", state=state)
    assert exc.value.code == "download_required"

    policy = web_fetch.WebFetchPolicy(state)
    policy.update(allow_downloads=True)
    result = web_fetch.fetch(
        server.origin + "/file", allow_download=True,
        download_name="file.bin", state=state)
    download = result["download"]
    assert download["bytes"] == 7
    assert download["sha256"] == hashlib.sha256(b"payload").hexdigest()
    assert Path(download["path"]).read_bytes() == b"payload"
    assert "body" not in result


def test_a_download_name_cannot_escape_the_directory(server, state):
    _grant_private(server, state)
    web_fetch.WebFetchPolicy(state).update(allow_downloads=True)
    server.route("/file", 200, b"payload", {
        "content-type": "application/octet-stream"})
    with pytest.raises(web_fetch.WebFetchError) as exc:
        web_fetch.fetch(server.origin + "/file", allow_download=True,
                        download_name="../escape", state=state)
    assert exc.value.code == "bad_download_name"


def test_upload_requires_a_policy_and_a_contained_path(server, state):
    _grant_private(server, state)
    server.route("/upload", 200, b"accepted")
    policy = web_fetch.WebFetchPolicy(state)
    uploads = policy.upload_dir()
    note = uploads / "note.txt"
    note.write_text("hello upload", encoding="utf-8")

    with pytest.raises(web_fetch.WebFetchError) as exc:
        web_fetch.fetch(server.origin + "/upload", method="POST",
                        upload_path=note, allow_upload=True, state=state)
    assert exc.value.code == "upload_blocked"

    policy.update(allow_uploads=True)
    result = web_fetch.fetch(server.origin + "/upload", method="POST",
                             upload_path=note, allow_upload=True, state=state)
    assert result["upload"]["bytes"] == len(b"hello upload")
    assert server.requests[-1]["body"] == b"hello upload"

    outside = state.root / "outside.txt"
    outside.write_text("no", encoding="utf-8")
    with pytest.raises(web_fetch.WebFetchError) as exc:
        web_fetch.fetch(server.origin + "/upload", method="POST",
                        upload_path=outside, allow_upload=True, state=state)
    assert exc.value.code == "upload_outside_root"


def test_sensitive_headers_are_refused(server, state, monkeypatch):
    monkeypatch.setattr(
        net_policy, "_host_is_private", lambda host: (False, "public"))
    server.route("/text", 200, b"ok")
    with pytest.raises(web_fetch.WebFetchError) as exc:
        web_fetch.fetch(server.origin + "/text",
                        headers={"Cookie": "secret=1"}, state=state)
    assert exc.value.code == "blocked_header"
    with pytest.raises(web_fetch.WebFetchError) as exc:
        web_fetch.fetch(server.origin + "/text",
                        headers={"Authorization": "Bearer x"}, state=state)
    assert exc.value.code == "blocked_header"

    result = web_fetch.fetch(
        server.origin + "/text",
        headers={"Accept-Language": "en"}, state=state)
    assert result["body"] == "ok"
    sent = {key.lower(): value
            for key, value in server.requests[-1]["headers"].items()}
    assert sent.get("accept-language") == "en"
    assert "cookie" not in sent


def test_size_and_content_type_limits(server, state):
    _grant_private(server, state)
    server.route("/big", 200, b"x" * 100)
    web_fetch.WebFetchPolicy(state).update(max_bytes=10)
    with pytest.raises(web_fetch.WebFetchError) as exc:
        web_fetch.fetch(server.origin + "/big", state=state)
    assert exc.value.code == "too_large"

    web_fetch.WebFetchPolicy(state).update(max_bytes=1000)
    server.route("/binary", 200, b"\x00\x01", {
        "content-type": "application/octet-stream"})
    with pytest.raises(web_fetch.WebFetchError) as exc:
        web_fetch.fetch(server.origin + "/binary", state=state)
    assert exc.value.code == "content_type"


def test_status_and_grants_are_listed(state):
    store = web_fetch.BrowserGrantStore(state)
    store.grant("https://example.com", access="private", scope="session",
                session_id="s1", reason="test")
    rows = store.list(session_id="s1")
    assert rows[0]["origin"] == "https://example.com"
    body = web_fetch.status(state)
    assert body["policy"]["max_bytes"] > 0
    assert body["grants"]


def test_api_private_fetch_needs_grant_then_approval(server, client):
    server.route("/text", 200, b"api-hello")
    denied = client.post("/v1/web/fetch", json={
        "url": server.origin + "/text", "session_id": "s1"})
    assert denied.status_code == 403
    assert denied.json()["detail"]["code"] == "private_origin"

    allowed = client.post("/v1/web/fetch", json={
        "url": server.origin + "/text", "session_id": "s1",
        "allow_private": True})
    assert allowed.status_code == 200, allowed.text
    assert allowed.json()["body"] == "api-hello"


def test_api_grant_policy_download_and_revoke(server, client):
    server.route("/file", 200, b"download-me", {
        "content-type": "application/octet-stream",
        "content-disposition": "attachment; filename=file.bin"})
    granted = client.post("/v1/web/grant", json={
        "origin": server.origin, "access": "private", "scope": "global",
        "reason": "api test"})
    assert granted.status_code == 200
    assert client.get("/v1/web/grants").json()["grants"]

    policy = client.post("/v1/web/policy", json={"allow_downloads": True})
    assert policy.status_code == 200
    assert policy.json()["allow_downloads"] is True

    downloaded = client.post("/v1/web/fetch", json={
        "url": server.origin + "/file", "allow_download": True,
        "download_name": "api-file.bin"})
    assert downloaded.status_code == 200, downloaded.text
    assert downloaded.json()["download"]["bytes"] == len(b"download-me")

    status = client.get("/v1/web/status").json()
    assert status["policy"]["allow_downloads"] is True
    removed = client.post("/v1/web/revoke", json={"origin": server.origin})
    assert removed.json()["global"] == 1


def test_api_blocks_sensitive_headers(server, client):
    client.post("/v1/web/grant", json={
        "origin": server.origin, "access": "private", "scope": "global"})
    response = client.post("/v1/web/fetch", json={
        "url": server.origin + "/text",
        "headers": {"Authorization": "Bearer secret"}})
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "blocked_header"
