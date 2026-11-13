"""Phase 2 daemon: native group-1 routes and the sidecar proxy.

The fake sidecar records the exact method/path/body/headers it receives, so the
proxy contract is tested without starting the real Python Studio.
"""

from __future__ import annotations

import http.client
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

import golden_support

REPO = Path(__file__).resolve().parents[2]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class _Sidecar:
    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.stream_closed = threading.Event()
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def _handler(self):
        requests = self.requests
        stream_closed = self.stream_closed

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
                        "request_id": self.headers.get("x-hive-request-id", ""),
                    }
                )
                if self.path == "/health":
                    self._respond(200, {"ok": True, "conversations": 7})
                elif self.path == "/v1/roles":
                    self._respond(200, {"roles": {"face": "fake"}, "from": "sidecar"})
                elif self.path == "/openapi.json":
                    self._respond(
                        200,
                        {
                            "openapi": "3.1.0",
                            "paths": {
                                "/v1/echo": {"post": {"summary": "sidecar echo"}},
                                "/v1/version": {
                                    "get": {"summary": "sidecar version"}
                                },
                            },
                            "components": {
                                "schemas": {"SidecarOnly": {"type": "object"}}
                            },
                        },
                    )
                elif self.path == "/v1/status":
                    self._respond(418, {"detail": "teapot"})
                elif self.path == "/v1/stream":
                    self.send_response(200)
                    self.send_header("content-type", "text/event-stream")
                    self.send_header("cache-control", "no-cache")
                    self.end_headers()
                    for index in range(100):
                        try:
                            self.wfile.write(f"data: {index}\n\n".encode())
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError):
                            stream_closed.set()
                            return
                        time.sleep(0.05)
                    stream_closed.set()
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
                        "request_id": self.headers.get("x-hive-request-id", ""),
                    }
                )
                self._respond(
                    201,
                    {
                        "method": "POST",
                        "path": self.path,
                        "body": body,
                        "token": self.headers.get("x-splinter-token", ""),
                        "request_id": self.headers.get("x-hive-request-id", ""),
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


class _Hub:
    """A tiny Hugging Face-shaped API for the native hub routes."""

    payload = b"GGUF-tiny-model"

    def __init__(self) -> None:
        self.requests: list[str] = []
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def api(self) -> str:
        return f"http://127.0.0.1:{self.port}/api"

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _handler(self):
        hub = self

        class Handler(BaseHTTPRequestHandler):
            def _json(self, payload: object) -> None:
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                hub.requests.append(self.path)
                if self.path.startswith("/api/models?"):
                    self._json(
                        [
                            {
                                "id": "org/tiny",
                                "downloads": 123,
                                "likes": 4,
                                "lastModified": "2026-01-02T03:04:05Z",
                                "createdAt": "2025-12-01T00:00:00Z",
                                "pipeline_tag": "text-generation",
                                "tags": [
                                    "gguf",
                                    "license:mit",
                                    "base_model:org/base",
                                    "text-generation",
                                ],
                            }
                        ]
                    )
                elif self.path == "/api/models/org/tiny/tree/main":
                    self._json(
                        [
                            {"path": "tiny.gguf", "size": 16},
                            {"path": "README.md", "size": 10},
                        ]
                    )
                elif self.path == "/org/tiny/resolve/main/tiny.gguf":
                    self.send_response(200)
                    self.send_header("content-type", "application/octet-stream")
                    self.send_header("content-length", str(len(hub.payload)))
                    self.end_headers()
                    self.wfile.write(hub.payload)
                elif self.path == "/org/tiny/resolve/main/broken.gguf":
                    self.send_response(200)
                    self.send_header("content-length", "100")
                    self.end_headers()
                    self.wfile.write(b"short")
                    self.close_connection = True
                else:
                    self.send_error(404)

            def log_message(self, *_args):
                pass

        return Handler

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=2)


def _raw_request(
    port: int,
    path: str,
    *,
    method: str = "GET",
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 5,
) -> tuple[int, dict, bytes]:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=body,
        headers=headers or {},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return (
                response.status,
                {key.lower(): value for key, value in response.headers.items()},
                response.read(),
            )
    except urllib.error.HTTPError as error:
        return (
            error.code,
            {key.lower(): value for key, value in error.headers.items()},
            error.read(),
        )


def _start_daemon(
    *,
    sidecar_url: str | None = None,
    sidecar_port: int | None = None,
    token: str | None = None,
    env_extra: dict[str, str] | None = None,
    webui: str | None = None,
) -> tuple[subprocess.Popen, int]:
    """Start the daemon against an external sidecar, a managed one, or none."""
    port = _free_port()
    env = dict(os.environ)
    if token:
        env["HARNESS_TOKEN"] = token
    else:
        env.pop("HARNESS_TOKEN", None)
    if env_extra:
        env.update(env_extra)
    args = [
        str(_daemon_bin()),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    if sidecar_url:
        args += ["--sidecar-url", sidecar_url]
    elif sidecar_port is not None:
        args += ["--sidecar-port", str(sidecar_port)]
    else:
        args += ["--no-sidecar"]
    if webui:
        args += ["--webui", webui]
    process = subprocess.Popen(
        args,
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


def _wait_health(
    port: int,
    *,
    managed: bool | None = None,
    pid_not: int | None = None,
    timeout: float = 90,
) -> dict:
    deadline = time.time() + timeout
    last: dict = {}
    while time.time() < deadline:
        status, _, payload = _request(port, "/health")
        last = payload
        assert status == 200
        sidecar = payload.get("sidecar") or {}
        if payload.get("ok") is True:
            if managed is not None and sidecar.get("managed") is not managed:
                time.sleep(0.2)
                continue
            if pid_not is not None and sidecar.get("pid") == pid_not:
                time.sleep(0.2)
                continue
            return payload
        time.sleep(0.2)
    raise AssertionError(f"sidecar did not become healthy: {last}")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def test_group_one_routes_and_proxy_fallback():
    sidecar = _Sidecar()
    process = None
    try:
        process, port = _start_daemon(sidecar_url=f"http://127.0.0.1:{sidecar.port}")

        status, headers, payload = _request(port, "/health")
        assert status == 200
        assert headers["x-hive-owner"] == "rust"
        assert payload["ok"] is True
        assert payload["conversations"] == 7
        assert payload["sidecar"]["managed"] is False
        assert payload["sidecar"]["url"] == f"http://127.0.0.1:{sidecar.port}"

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
        assert headers["x-hive-request-id"] == payload["request_id"]

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
        process, port = _start_daemon(
            sidecar_url=f"http://127.0.0.1:{sidecar.port}", token="required"
        )

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
        process, port = _start_daemon(sidecar_url=f"http://127.0.0.1:{_free_port()}")
        status, _, payload = _request(port, "/health")
        assert status == 200
        assert payload["ok"] is False
        assert "error" in payload
    finally:
        if process is not None:
            process.terminate()
            process.wait(timeout=5)


def test_managed_sidecar_starts_proxies_restarts_and_stops():
    """The daemon owns a real FastAPI sidecar: start, proxy, restart, stop."""
    process = None
    first_pid = second_pid = None
    try:
        process, port = _start_daemon(
            sidecar_port=0,
            env_extra={"HIVE_PYTHON": sys.executable},
        )
        health = _wait_health(port, managed=True)
        first_pid = health["sidecar"]["pid"]
        assert isinstance(first_pid, int)
        assert health["sidecar"]["url"].startswith("http://127.0.0.1:")
        with urllib.request.urlopen(
            f"{health['sidecar']['url']}/health", timeout=5
        ) as response:
            direct = json.loads(response.read())
        assert direct["service"] == "hive-sidecar"
        assert direct["version"]

        status, headers, payload = _request(port, "/v1/commands")
        assert status == 200
        assert headers["x-hive-owner"] == "python"
        assert payload["commands"]

        # Runtime/gateway groups still proxy, but through the real sidecar.
        for path in (
            "/v1/activity?limit=1",
            "/v1/tools/pending",
            "/v1/agent/status",
            "/v1/sandbox/status",
        ):
            status, headers, payload = _request(port, path)
            assert status == 200, (path, payload)
            assert headers["x-hive-owner"] == "python", path
            assert isinstance(payload, dict), path

        # Kill the sidecar; the supervisor must bring up a replacement.
        os.kill(first_pid, 9)
        health = _wait_health(port, managed=True, pid_not=first_pid)
        second_pid = health["sidecar"]["pid"]
        assert second_pid != first_pid
        status, headers, payload = _request(port, "/v1/commands")
        assert status == 200
        assert headers["x-hive-owner"] == "python"

        # Daemon shutdown stops the sidecar's process group.
        process.terminate()
        process.wait(timeout=20)
        process = None
        deadline = time.time() + 10
        while time.time() < deadline and _alive(second_pid):
            time.sleep(0.2)
        assert not _alive(second_pid)
    finally:
        if process is not None:
            process.terminate()
            process.wait(timeout=20)
        for pid in (first_pid, second_pid):
            if pid and _alive(pid):
                try:
                    os.kill(pid, 9)
                except ProcessLookupError:
                    pass
                raise AssertionError(f"leaked sidecar pid {pid}")


def test_managed_sidecar_rejects_an_unusable_HIVE_PYTHON():
    port = _free_port()
    env = dict(os.environ)
    env["HIVE_PYTHON"] = "/nonexistent/hive-python"
    process = subprocess.Popen(
        [
            str(_daemon_bin()),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--sidecar-port",
            "0",
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        code = process.wait(timeout=15)
        error = process.stderr.read() if process.stderr else ""
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
    assert code != 0
    assert "HIVE_PYTHON" in error
    assert "not an executable file" in error


def test_static_console_and_api_prefix(tmp_path):
    sidecar = _Sidecar()
    process = None
    try:
        (tmp_path / "index.html").write_text("<html><body>hive console</body></html>")
        (tmp_path / "app.js").write_text("console.log('hive');")
        (tmp_path / "styles.css").write_text("body { color: black; }")
        (tmp_path / "secret.txt").write_text("outside")

        process, port = _start_daemon(
            sidecar_url=f"http://127.0.0.1:{sidecar.port}", webui=str(tmp_path)
        )

        status, headers, body = _raw_request(port, "/")
        assert status == 200
        assert headers["content-type"].startswith("text/html")
        assert headers["cache-control"] == "no-store"
        assert b"hive console" in body
        assert headers["x-hive-owner"] == "rust"

        status, headers, body = _raw_request(port, "/app.js?v=94")
        assert status == 200
        assert "javascript" in headers["content-type"]
        assert b"console.log" in body
        assert headers["cache-control"] == "public, max-age=300"

        status, headers, body = _raw_request(port, "/styles.css")
        assert status == 200
        assert headers["content-type"].startswith("text/css")

        # The console calls /api/v1/*; the daemon strips the console prefix.
        status, headers, payload = _request(
            port, "/api/v1/echo?x=1", method="POST", body=b"{}"
        )
        assert status == 201
        assert headers["x-hive-owner"] == "python"
        assert payload["path"] == "/v1/echo?x=1"

        # The console's local role editor maps onto the sidecar's /v1/roles.
        status, _, payload = _request(port, "/api/roles")
        assert status == 200
        assert payload == {"roles": {"face": "fake"}, "from": "sidecar"}

        status, _, payload = _request(port, "/v1/version")
        assert status == 200
        assert payload["hive"] == "0.2.0"

        # Encoded traversal must not escape the asset root.
        status, _, body = _raw_request(port, "/%2e%2e/secret.txt")
        assert status in (404, 502)
        assert b"outside" not in body
    finally:
        if process is not None:
            process.terminate()
            process.wait(timeout=5)
        sidecar.close()


def _home(tmp_path) -> dict[str, str]:
    return {"HIVE_HOME": str(tmp_path / "home")}


def test_stack_crud_and_validate(tmp_path):
    process = None
    try:
        process, port = _start_daemon(env_extra=_home(tmp_path))
        stack = {
            "name": "endpoint-only",
            "version": 1,
            "tiers": [
                {
                    "role": "face",
                    "repo": "local",
                    "file": "remote.gguf",
                    "endpoint": "http://127.0.0.1:9/v1",
                    "ctx": 2048,
                    "ngl": 0,
                    "backend": "cpu",
                }
            ],
            "routing": {},
        }
        status, headers, payload = _request(
            port, "/v1/stacks/endpoint-only", method="PUT", body=json.dumps(stack).encode()
        )
        assert status == 200
        assert headers["x-hive-owner"] == "rust"
        assert payload["name"] == "endpoint-only"
        assert payload["tiers"][0]["role"] == "face"

        status, _, payload = _request(port, "/v1/stacks")
        assert status == 200
        assert payload == [
            {"name": "endpoint-only", "tiers": 1, "roles": ["face"]}
        ]

        status, _, payload = _request(port, "/v1/stacks/endpoint-only")
        assert status == 200
        assert payload["tiers"][0]["endpoint"] == "http://127.0.0.1:9/v1"

        status, _, payload = _request(
            port, "/v1/stacks/endpoint-only/validate", method="POST", body=b"{}"
        )
        assert status == 200
        assert payload["ok"] is True
        assert payload["per_card"] == []
        assert any("endpoint-backed" in warning for warning in payload["warnings"])

        status, _, payload = _request(port, "/v1/stacks/endpoint-only", method="DELETE")
        assert status == 200
        assert payload == {"ok": True, "deleted": "endpoint-only"}
        status, _, payload = _request(port, "/v1/stacks/endpoint-only")
        assert status == 404

        bad = dict(stack, name="other")
        status, _, payload = _request(
            port, "/v1/stacks/endpoint-only", method="PUT", body=json.dumps(bad).encode()
        )
        assert status == 400
        assert "does not match" in payload["detail"]
    finally:
        if process is not None:
            process.terminate()
            process.wait(timeout=5)


def test_apply_stack_and_server_lifecycle(tmp_path):
    llama = os.environ.get("HIVE_LLAMA_SERVER")
    model = os.environ.get("HIVE_LIVE_MODEL")
    if not llama or not model or not Path(llama).is_file() or not Path(model).is_file():
        pytest.skip("set HIVE_LLAMA_SERVER and HIVE_LIVE_MODEL for the live engine test")
    process = None
    try:
        process, port = _start_daemon(
            sidecar_port=0,
            env_extra={
                **_home(tmp_path),
                "HIVE_PYTHON": sys.executable,
                "HIVE_LLAMA_SERVER": llama,
                "HARNESS_LLAMA_SERVER": llama,
            },
        )
        stack = {
            "name": "live",
            "version": 1,
            "tiers": [
                {
                    "role": "face",
                    "repo": "local",
                    "file": model,
                    "ctx": 512,
                    "ngl": 0,
                    "backend": "cpu",
                }
            ],
            "routing": {},
        }
        status, _, _ = _request(
            port, "/v1/stacks/live", method="PUT", body=json.dumps(stack).encode()
        )
        assert status == 200

        status, _, payload = _request(
            port, "/v1/stacks/live/apply", method="POST", body=b"{}"
        )
        assert status == 200, payload
        assert payload["ok"] is True
        assert payload["tiers"][0]["key"] == "live-face"
        engine_port = payload["tiers"][0]["port"]
        assert engine_port > 0

        status, _, status_payload = _request(port, "/v1/stacks/status")
        assert status == 200
        assert status_payload["stack"] == "live"
        assert status_payload["tiers"][0]["port"] == engine_port

        status, _, server = _request(port, "/v1/server/status")
        assert status == 200
        assert server["running"] is True
        assert server["instances"][0]["port"] == engine_port
        assert server["instances"][0]["pid"]

        status, _, log = _request(port, "/v1/server/log?tail=20")
        assert status == 200
        assert isinstance(log["lines"], list)

        status, _, payload = _request(
            port, "/v1/stacks/live/unload", method="POST", body=b"{}"
        )
        assert status == 200
        assert payload["unloaded"] == ["face"]
        status, _, status_payload = _request(port, "/v1/stacks/status")
        assert status_payload["stack"] is None

        # The single-model server route shares the same runtime.
        launch = {
            "model": model,
            "backend": "cpu",
            "ngl": 0,
            "ctx_size": 512,
            "key": "smol",
        }
        status, _, instance = _request(
            port, "/v1/server/start", method="POST", body=json.dumps(launch).encode()
        )
        assert status == 200, instance
        assert instance["key"] == "smol"
        assert instance["pid"]
        status, _, providers = _request(port, "/v1/provider/config")
        assert status == 200
        mirrored = {
            provider["name"]: provider for provider in providers["providers"]
        }
        assert "local-smol" in mirrored
        assert mirrored["local-smol"]["base_url"] == f"http://127.0.0.1:{instance['port']}"
        assert providers["default"] == "local-smol"
        status, _, engines = _request(port, "/v1/engines")
        assert status == 200
        engine_rows = {engine["name"]: engine for engine in engines["engines"]}
        assert engine_rows["local-smol"]["kind"] == "llama_cpp"
        assert engines["default"] == "local-smol"
        status, _, server = _request(port, "/v1/server/status")
        assert server["loaded_model"] == instance["model"]

        # A second model can coexist; stopping one leaves the other loaded.
        second = dict(launch, key="smol2")
        status, _, second_instance = _request(
            port, "/v1/server/start", method="POST", body=json.dumps(second).encode()
        )
        assert status == 200, second_instance
        assert second_instance["key"] == "smol2"
        assert second_instance["port"] != instance["port"]
        status, _, server = _request(port, "/v1/server/status")
        loaded_keys = {row["key"] for row in server["loaded"]}
        assert loaded_keys == {"smol", "smol2"}, loaded_keys

        status, _, payload = _request(
            port, "/v1/server/stop?key=smol2", method="POST", body=b"{}"
        )
        assert status == 200
        status, _, server = _request(port, "/v1/server/status")
        assert {row["key"] for row in server["loaded"]} == {"smol"}

        # A crashed engine must stop looking loaded on the next status poll.
        os.kill(instance["pid"], 9)
        deadline = time.time() + 10
        server = {}
        while time.time() < deadline:
            status, _, server = _request(port, "/v1/server/status")
            if server.get("running") is False:
                break
            time.sleep(0.1)
        assert server.get("running") is False, server

        status, _, payload = _request(
            port, "/v1/server/stop?key=smol", method="POST", body=b"{}"
        )
        assert status == 200
        assert payload["unloaded"] == "smol"
        status, _, providers = _request(port, "/v1/provider/config")
        assert "local-smol" not in {
            provider["name"] for provider in providers["providers"]
        }
        status, _, engines = _request(port, "/v1/engines")
        assert "local-smol" not in {
            engine["name"] for engine in engines["engines"]
        }
        status, _, server = _request(port, "/v1/server/status")
        assert server["running"] is False
    finally:
        if process is not None:
            process.terminate()
            process.wait(timeout=10)


def test_local_library_and_hub_routes(tmp_path):
    hub = _Hub()
    process = None
    try:
        process, port = _start_daemon(
            env_extra={
                **_home(tmp_path),
                "HIVE_HF_API": hub.api,
                "HIVE_HF_BASE": hub.base,
            }
        )

        status, headers, payload = _request(port, "/v1/models/hub?q=tiny&limit=5")
        assert status == 200
        assert headers["x-hive-owner"] == "rust"
        assert payload["results"][0]["repo"] == "org/tiny"
        assert payload["results"][0]["license"] == "mit"
        assert payload["results"][0]["base_model"] == "org/base"
        assert payload["results"][0]["last_modified"] == "2026-01-02"

        status, _, payload = _request(port, "/v1/models/hub/files/org/tiny")
        assert status == 200
        assert payload == {
            "repo": "org/tiny",
            "files": [{"file": "tiny.gguf", "size_gb": 0.0}],
        }

        status, _, payload = _request(
            port,
            "/v1/models/hub/download",
            method="POST",
            body=json.dumps({"repo": "org/tiny", "file": "tiny.gguf"}).encode(),
        )
        assert status == 200
        assert payload["state"] in ("queued", "downloading")

        deadline = time.time() + 15
        job = {}
        while time.time() < deadline:
            status, _, payload = _request(port, "/v1/models/hub/downloads")
            assert status == 200
            job = next(
                (row for row in payload["downloads"] if row["filename"] == "tiny.gguf"),
                {},
            )
            if job.get("state") in ("done", "error"):
                break
            time.sleep(0.1)
        assert job["state"] == "done", job
        saved = Path(job["path"])
        assert saved.read_bytes() == _Hub.payload
        assert saved.parent == tmp_path / "home" / "models" / "gguf"

        status, _, library = _request(port, "/v1/models/local")
        assert status == 200
        assert library["models_dir"] == str(tmp_path / "home" / "models" / "gguf")
        assert any(row["file"] == "tiny.gguf" for row in library["models"])

        status, _, payload = _request(
            port, "/v1/models/local?file=tiny.gguf", method="DELETE"
        )
        assert status == 200
        assert payload["ok"] is True
        assert not saved.exists()
        status, _, payload = _request(
            port, "/v1/models/local?file=../outside.gguf", method="DELETE"
        )
        assert status == 400

        status, _, payload = _request(
            port,
            "/v1/models/hub/download",
            method="POST",
            body=json.dumps({"repo": "org/tiny", "file": "../escape.gguf"}).encode(),
        )
        assert status == 400

        status, _, payload = _request(
            port,
            "/v1/models/hub/download",
            method="POST",
            body=json.dumps({"repo": "org/tiny", "file": "broken.gguf"}).encode(),
        )
        assert status == 200
        deadline = time.time() + 15
        job = {}
        while time.time() < deadline:
            status, _, payload = _request(port, "/v1/models/hub/downloads")
            job = next(
                (row for row in payload["downloads"] if row["filename"] == "broken.gguf"),
                {},
            )
            if job.get("state") in ("done", "error"):
                break
            time.sleep(0.1)
        assert job["state"] == "error", job
        assert not (tmp_path / "home" / "models" / "gguf" / "broken.gguf").exists()
        assert (tmp_path / "home" / "models" / "gguf" / "broken.gguf.part").exists()
    finally:
        if process is not None:
            process.terminate()
            process.wait(timeout=5)
        hub.close()


def test_cors_defaults_are_not_wildcard():
    sidecar = _Sidecar()
    process = None
    try:
        process, port = _start_daemon(sidecar_url=f"http://127.0.0.1:{sidecar.port}")
        status, headers, _ = _raw_request(
            port,
            "/v1/version",
            method="OPTIONS",
            headers={
                "Origin": "http://evil.example",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert status in (200, 204)
        assert "access-control-allow-origin" not in headers

        status, headers, _ = _raw_request(
            port,
            "/v1/version",
            method="OPTIONS",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert status in (200, 204)
        assert headers["access-control-allow-origin"] == "http://localhost:5173"
    finally:
        if process is not None:
            process.terminate()
            process.wait(timeout=5)
        sidecar.close()


def test_streaming_client_disconnect_cancels_the_upstream():
    sidecar = _Sidecar()
    process = None
    try:
        process, port = _start_daemon(sidecar_url=f"http://127.0.0.1:{sidecar.port}")
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        connection.request("GET", "/v1/stream")
        response = connection.getresponse()
        assert response.status == 200
        assert response.read(12)
        connection.close()
        assert sidecar.stream_closed.wait(5), "upstream stream was not cancelled"
    finally:
        if process is not None:
            process.terminate()
            process.wait(timeout=5)
        sidecar.close()


def test_runs_index_and_report_json(tmp_path):
    runs = tmp_path / "home" / "runs"
    (runs / "old-run").mkdir(parents=True)
    (runs / "new-run").mkdir()
    (runs / "old-run" / "run_report.json").write_text(
        json.dumps({"kind": "benchmark", "score": 1})
    )
    (runs / "new-run" / "run_report.json").write_text(
        json.dumps({"kind": "cascade", "score": 2})
    )
    (runs / "plain").mkdir()
    os.utime(runs / "old-run", (1_700_000_000, 1_700_000_000))
    os.utime(runs / "new-run", (1_700_100_000, 1_700_100_000))
    os.utime(runs / "plain", (1_700_200_000, 1_700_200_000))

    process = None
    try:
        process, port = _start_daemon(env_extra=_home(tmp_path))
        status, headers, payload = _request(port, "/v1/runs")
        assert status == 200
        assert headers["x-hive-owner"] == "rust"
        names = [row["name"] for row in payload["runs"]]
        assert names == ["plain", "new-run", "old-run"]
        by_name = {row["name"]: row for row in payload["runs"]}
        assert by_name["new-run"]["kind"] == "cascade"
        assert by_name["plain"]["has_report"] is False

        status, _, payload = _request(port, "/v1/report/new-run")
        assert status == 200
        assert payload == {"kind": "cascade", "score": 2}

        status, _, payload = _request(port, "/v1/report/missing")
        assert status == 404

        status, _, payload = _request(port, "/v1/report/%2e%2e/%2e%2e/etc")
        assert status in (400, 404)
    finally:
        if process is not None:
            process.terminate()
            process.wait(timeout=5)


def test_hive_ui_detach_starts_the_daemon_and_console(tmp_path):
    cli = Path(os.environ.get("HIVE_RUST_BIN", REPO / "rust" / "target" / "debug" / "hive"))
    daemon = REPO / "rust" / "target" / "debug" / "hive-daemon"
    if not cli.is_file() or not daemon.is_file():
        pytest.skip("hive/hive-daemon binaries are not built")
    port = _free_port()
    env = {
        **os.environ,
        **_home(tmp_path),
        "HIVE_PYTHON": sys.executable,
        "HIVE_DAEMON_BIN": str(daemon),
        "TZ": "UTC",
    }
    result = subprocess.run(
        [
            str(cli),
            "ui",
            "web",
            "--detach",
            "--no-open",
            "--port",
            str(port),
            "--console-port",
            str(port),
        ],
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    deadline = time.time() + 20
    health = {}
    while time.time() < deadline:
        try:
            status, _, health = _request(port, "/health")
            if status == 200 and health.get("ok") is True:
                break
        except OSError:
            pass
        time.sleep(0.1)
    assert health.get("ok") is True, health
    sidecar_pid = health["sidecar"]["pid"]
    assert _alive(sidecar_pid)

    status, headers, body = _raw_request(port, "/")
    assert status == 200
    assert headers["content-type"].startswith("text/html")
    assert b"<html" in body.lower()

    pidfile = tmp_path / "home" / "logs" / "daemon.pid"
    daemon_pid = int(pidfile.read_text().strip())
    os.kill(daemon_pid, 15)
    deadline = time.time() + 15
    while time.time() < deadline and (_alive(daemon_pid) or _alive(sidecar_pid)):
        time.sleep(0.2)
    assert not _alive(daemon_pid)
    assert not _alive(sidecar_pid)


def test_group_one_payloads_match_the_frozen_golden(tmp_path):
    process = None
    try:
        process, port = _start_daemon(
            env_extra={
                **_home(tmp_path),
                "HIVE_STATE_DIR": str(tmp_path / "state"),
                "HIVE_PYTHON": sys.executable,
            }
        )
        for name, path in (
            ("daemon_version", "/v1/version"),
            ("daemon_config", "/v1/config"),
        ):
            status, _, payload = _request(port, path)
            assert status == 200
            actual = golden_support.normalize(payload, tmp_path, sys.executable)
            expected = json.loads((golden_support.GOLDEN / f"{name}.json").read_text())
            assert actual == expected, name
    finally:
        if process is not None:
            process.terminate()
            process.wait(timeout=5)


def test_local_library_path_and_import(tmp_path):
    external = tmp_path / "external"
    external.mkdir()
    (external / "ext.gguf").write_bytes(b"GGUF\x03\x00\x00\x00")
    target = tmp_path / "new-library"
    process = None
    try:
        process, port = _start_daemon(env_extra=_home(tmp_path))

        status, _, payload = _request(
            port,
            "/v1/models/local/path",
            method="POST",
            body=json.dumps({"folder": str(target)}).encode(),
        )
        assert status == 200
        assert payload["models_dir"] == str(target)
        assert payload["models"] == []
        assert (tmp_path / "home" / "harness_state" / "models_dir.txt").read_text() == str(target)

        status, _, payload = _request(
            port,
            "/v1/models/local/import-path",
            method="POST",
            body=json.dumps({"folder": str(external)}).encode(),
        )
        assert status == 200
        linked = Path(payload["linked"])
        assert linked.is_symlink()
        assert linked.resolve() == external.resolve()
        assert payload["imported"] == ["external/ext.gguf"]

        status, _, payload = _request(port, "/v1/models/local")
        assert payload["models_dir"] == str(target)
        assert any(row["file"] == "external/ext.gguf" for row in payload["models"])

        status, _, payload = _request(
            port,
            "/v1/models/local/path",
            method="POST",
            body=json.dumps({"folder": str(external / "ext.gguf")}).encode(),
        )
        assert status == 400
    finally:
        if process is not None:
            process.terminate()
            process.wait(timeout=5)


def test_live_studio_chat_through_the_daemon(tmp_path):
    llama = os.environ.get("HIVE_LLAMA_SERVER")
    model = os.environ.get("HIVE_LIVE_MODEL")
    if not llama or not model or not Path(llama).is_file() or not Path(model).is_file():
        pytest.skip("set HIVE_LLAMA_SERVER and HIVE_LIVE_MODEL for the live Studio test")
    process = None
    try:
        process, port = _start_daemon(
            sidecar_port=0,
            env_extra={
                **_home(tmp_path),
                "HIVE_PYTHON": sys.executable,
                "HIVE_LLAMA_SERVER": llama,
                "HARNESS_LLAMA_SERVER": llama,
            },
        )
        launch = {
            "model": model,
            "backend": "cpu",
            "ngl": 0,
            "ctx_size": 1024,
            "key": "studio",
        }
        status, _, instance = _request(
            port, "/v1/server/start", method="POST", body=json.dumps(launch).encode()
        )
        assert status == 200, instance

        status, headers, body = _raw_request(
            port,
            "/v1/splinter/stream",
            method="POST",
            body=json.dumps(
                {
                    "query": "Say hello in one short sentence.",
                    "conversation_id": "g2-chat",
                }
            ).encode(),
            headers={"content-type": "application/json"},
            timeout=120,
        )
        assert status == 200, body
        assert headers["content-type"].startswith("text/event-stream")
        events = [
            json.loads(line[6:])
            for line in body.decode().splitlines()
            if line.startswith("data: ")
        ]
        assert any(event.get("type") == "delta" for event in events), events
        assert not any(event.get("type") == "error" for event in events), events
        assert events[-1]["type"] == "done", events[-1]

        status, _, payload = _request(
            port, "/v1/server/stop?key=studio", method="POST", body=b"{}"
        )
        assert status == 200, payload
    finally:
        if process is not None:
            process.terminate()
            process.wait(timeout=15)


def test_openapi_merges_native_and_sidecar_routes():
    sidecar = _Sidecar()
    process = None
    try:
        process, port = _start_daemon(sidecar_url=f"http://127.0.0.1:{sidecar.port}")
        status, headers, spec = _request(port, "/openapi.json")
        assert status == 200
        assert headers["x-hive-owner"] == "rust"
        assert spec["openapi"] == "3.1.0"
        paths = spec["paths"]
        assert "/v1/stacks" in paths
        assert "/v1/server/start" in paths
        assert "/v1/echo" in paths
        assert (
            paths["/v1/version"]["get"]["summary"]
            == "Hive, Python, and platform versions"
        )
        assert "SidecarOnly" in spec["components"]["schemas"]
    finally:
        if process is not None:
            process.terminate()
            process.wait(timeout=5)
        sidecar.close()
