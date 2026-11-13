"""Phase 2 daemon: native group-1 routes and the sidecar proxy.

The fake sidecar records the exact method/path/body/headers it receives, so the
proxy contract is tested without starting the real Python Studio.
"""

from __future__ import annotations

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
                elif self.path == "/v1/roles":
                    self._respond(200, {"roles": {"face": "fake"}, "from": "sidecar"})
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


def _raw_request(
    port: int,
    path: str,
    *,
    method: str = "GET",
    body: bytes | None = None,
) -> tuple[int, dict, bytes]:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=body,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
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
            env_extra={
                **_home(tmp_path),
                "HIVE_LLAMA_SERVER": llama,
                "HARNESS_LLAMA_SERVER": llama,
            }
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
        status, _, server = _request(port, "/v1/server/status")
        assert server["loaded_model"] == instance["model"]
        status, _, payload = _request(
            port, "/v1/server/stop?key=smol", method="POST", body=b"{}"
        )
        assert status == 200
        assert payload["unloaded"] == "smol"
        status, _, server = _request(port, "/v1/server/status")
        assert server["running"] is False
    finally:
        if process is not None:
            process.terminate()
            process.wait(timeout=10)
