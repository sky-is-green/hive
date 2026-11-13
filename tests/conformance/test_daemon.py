"""Phase 2 daemon: native group-1 routes and the sidecar proxy.

The fake sidecar records the exact method/path/body/headers it receives, so the
proxy contract is tested without starting the real Python Studio.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class _Sidecar:
    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def _handler(self):
        requests = self.requests

        class Handler(BaseHTTPRequestHandler):
            def _respond(self, status: int, payload: object) -> None:
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.send_header("x-upstream", "yes")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                requests.append(
                    {
                        "method": "GET",
                        "path": self.path,
                        "body": "",
                        "token": self.headers.get("x-splinter-token", ""),
                    }
                )
                if self.path == "/health":
                    self._respond(200, {"ok": True, "conversations": 7})
                elif self.path == "/v1/status":
                    self._respond(418, {"detail": "teapot"})
                else:
                    self._respond(404, {"detail": "not found"})

            def do_POST(self):
                length = int(self.headers.get("content-length") or 0)
                body = self.rfile.read(length).decode()
                requests.append(
                    {
                        "method": "POST",
                        "path": self.path,
                        "body": body,
                        "token": self.headers.get("x-splinter-token", ""),
                    }
                )
                self._respond(
                    201,
                    {
                        "method": "POST",
                        "path": self.path,
                        "body": body,
                        "token": self.headers.get("x-splinter-token", ""),
                    },
                )

            def log_message(self, *_args):
                pass

        return Handler

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=2)


def _daemon_bin() -> Path:
    raw = os.environ.get("HIVE_DAEMON_BIN")
    path = Path(raw) if raw else REPO / "rust" / "target" / "debug" / "hive-daemon"
    if not path.is_file():
        pytest.skip(
            "hive-daemon not built; run `cargo build --manifest-path rust/Cargo.toml`"
        )
    return path


def _request(
    port: int,
    path: str,
    *,
    method: str = "GET",
    body: bytes | None = None,
    token: str | None = None,
) -> tuple[int, dict, dict]:
    headers = {"content-type": "application/json"}
    if token:
        headers["x-splinter-token"] = token
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=body,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return (
                response.status,
                {key.lower(): value for key, value in response.headers.items()},
                json.loads(response.read() or b"{}"),
            )
    except urllib.error.HTTPError as error:
        return (
            error.code,
            {key.lower(): value for key, value in error.headers.items()},
            json.loads(error.read() or b"{}"),
        )


def _start_daemon(sidecar_port: int, *, token: str | None = None) -> tuple[subprocess.Popen, int]:
    port = _free_port()
    env = dict(os.environ)
    if token:
        env["HARNESS_TOKEN"] = token
    else:
        env.pop("HARNESS_TOKEN", None)
    process = subprocess.Popen(
        [
            str(_daemon_bin()),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--sidecar-url",
            f"http://127.0.0.1:{sidecar_port}",
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.time() + 10
    while time.time() < deadline:
        if process.poll() is not None:
            raise AssertionError(process.stderr.read() if process.stderr else "daemon exited")
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=0.5
            ) as response:
                if response.status == 200:
                    return process, port
        except OSError:
            time.sleep(0.05)
    process.kill()
    raise AssertionError("daemon did not become ready")


def test_group_one_routes_and_proxy_fallback():
    sidecar = _Sidecar()
    process = None
    try:
        process, port = _start_daemon(sidecar.port)

        status, headers, payload = _request(port, "/health")
        assert status == 200
        assert headers["x-hive-owner"] == "rust"
        assert payload == {"ok": True, "conversations": 7}

        status, headers, payload = _request(port, "/v1/setup/status")
        assert status == 410
        assert headers["x-hive-owner"] == "rust"
        assert "Windows-only" in payload["detail"]

        for path in ("/v1/version", "/v1/config"):
            status, headers, payload = _request(port, path)
            assert status == 200, path
            assert headers["x-hive-owner"] == "rust", path
        _, _, version = _request(port, "/v1/version")
        assert version["hive"] == "0.2.0"
        assert "python" in version and "platform" in version
        _, _, config = _request(port, "/v1/config")
        for key in (
            "repo",
            "models_dir",
            "stacks_dir",
            "state_dir",
            "serve_dir",
            "log_dir",
        ):
            assert key in config, key

        status, headers, payload = _request(
            port,
            "/v1/echo?x=1",
            method="POST",
            body=b'{"hello": "world"}',
            token="s3cret",
        )
        assert status == 201
        assert headers["x-hive-owner"] == "python"
        assert headers["x-upstream"] == "yes"
        assert payload["method"] == "POST"
        assert payload["path"] == "/v1/echo?x=1"
        assert payload["body"] == '{"hello": "world"}'
        assert payload["token"] == "s3cret"

        status, headers, payload = _request(port, "/v1/status")
        assert status == 418
        assert headers["x-hive-owner"] == "python"
        assert payload == {"detail": "teapot"}

        status, headers, _ = _request(port, "/v1/missing")
        assert status == 404
        assert headers["x-hive-owner"] == "python"
    finally:
        if process is not None:
            process.terminate()
            process.wait(timeout=5)
        sidecar.close()


def test_token_guard_and_sidecar_health():
    sidecar = _Sidecar()
    process = None
    try:
        process, port = _start_daemon(sidecar.port, token="required")

        status, _, payload = _request(port, "/health")
        assert status == 200
        assert payload["conversations"] == 7

        status, _, payload = _request(port, "/v1/version")
        assert status == 401
        assert payload == {"detail": "invalid or missing token"}

        status, _, _ = _request(port, "/v1/version", token="required")
        assert status == 200

        status, _, payload = _request(
            port, "/v1/echo", method="POST", body=b"ok", token="required"
        )
        assert status == 201
        assert payload["token"] == "required"
    finally:
        if process is not None:
            process.terminate()
            process.wait(timeout=5)
        sidecar.close()


def test_health_reports_a_dead_sidecar_without_failing_liveness():
    process = None
    try:
        process, port = _start_daemon(_free_port())
        status, _, payload = _request(port, "/health")
        assert status == 200
        assert payload["ok"] is False
        assert "error" in payload
    finally:
        if process is not None:
            process.terminate()
            process.wait(timeout=5)
