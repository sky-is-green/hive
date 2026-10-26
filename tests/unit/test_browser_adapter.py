"""The optional browser adapter and its validating proxy."""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from harness import browser_adapter
from harness.app import create_app
from harness.state_store import StatePaths
from harness.web_fetch import WebFetchError


def _browser_ready() -> bool:
    return bool(browser_adapter.status().get("available"))


pytestmark = pytest.mark.skipif(
    not _browser_ready(), reason="Playwright Chromium is not installed")


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.server.hits += 1  # type: ignore[attr-defined]
        route = self.server.routes.get(self.path)  # type: ignore[attr-defined]
        if route is None:
            self.send_response(404)
            self.end_headers()
            return
        status, headers, body = route
        self.send_response(status)
        for key, value in headers.items():
            self.send_header(key, value)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: Any) -> None:
        pass


class _Server:
    def __init__(self) -> None:
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.httpd.hits = 0  # type: ignore[attr-defined]
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
    def hits(self) -> int:
        return self.httpd.hits  # type: ignore[attr-defined]

    def route(self, path: str, status: int, body: bytes,
              headers: dict[str, str] | None = None) -> None:
        self.httpd.routes[path] = (  # type: ignore[attr-defined]
            status, headers or {"content-type": "text/html"}, body)

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture()
def page_server():
    srv = _Server().start()
    srv.route("/", 200, (
        b"<html><head><title>Browser Test</title></head>"
        b"<body><h1>hello browser</h1></body></html>"))
    try:
        yield srv
    finally:
        srv.close()


@pytest.fixture()
def state(tmp_path):
    return StatePaths(tmp_path / "state").ensure()


def test_browse_private_needs_grant_then_approval(page_server, state):
    with pytest.raises(WebFetchError) as exc:
        browser_adapter.browse(page_server.origin + "/", session_id="s1",
                               state=state)
    assert exc.value.code == "private_origin"
    assert page_server.hits == 0

    result = browser_adapter.browse(
        page_server.origin + "/", session_id="s1", allow_private=True,
        screenshot_name="page.png", state=state)
    assert result["title"] == "Browser Test"
    assert "hello browser" in result["text"]
    assert result["screenshot"]["path"].endswith("page.png")
    assert Path(result["screenshot"]["path"]).is_file()
    assert page_server.hits >= 1


def test_redirect_to_a_second_private_origin_is_blocked(state):
    other = _Server().start()
    first = _Server().start()
    try:
        other.route("/secret", 200, b"<html><body>secret</body></html>")
        first.route("/", 302, b"", {
            "content-type": "text/html",
            "location": other.origin + "/secret"})
        with pytest.raises(WebFetchError) as exc:
            browser_adapter.browse(
                first.origin + "/", session_id="s1", allow_private=True,
                state=state)
        assert exc.value.code == "private_origin"
        assert exc.value.detail["blocked"][0]["origin"] == other.origin
        assert other.hits == 0
    finally:
        first.close()
        other.close()


def test_api_browser_status_and_browse(page_server, tmp_path):
    app = create_app(
        providers_file=tmp_path / "providers.local.json",
        engines_file=tmp_path / "engines.local.json",
        runs_root=tmp_path / "runs",
        log_dir=str(tmp_path / "logs"),
        state_dir=str(tmp_path / "state"),
    )
    client = TestClient(app)
    status = client.get("/v1/browser/status").json()
    assert status["available"] is True

    denied = client.post("/v1/browser/browse", json={
        "url": page_server.origin + "/", "session_id": "s1"})
    assert denied.status_code == 403
    assert denied.json()["detail"]["code"] == "private_origin"

    allowed = client.post("/v1/browser/browse", json={
        "url": page_server.origin + "/", "session_id": "s1",
        "allow_private": True})
    assert allowed.status_code == 200, allowed.text
    assert allowed.json()["title"] == "Browser Test"
