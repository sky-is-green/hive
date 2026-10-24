"""End-to-end: a detached serve survives its parent, then status/stop find it.

The tier is an endpoint (a tiny in-process OpenAI-compatible server), so this
exercises the CLI lifecycle without touching a GPU or a model file.
"""
import json
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

import harness.cli as cli
from harness import serve_state

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="detached-serve semantics are POSIX-first")


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 - http.server naming
        path = self.path.rstrip("/")
        if path in ("/v1/models", "/models"):
            body = json.dumps({"object": "list",
                               "data": [{"id": "fake-model"}]}).encode()
        elif path in ("/health", "/v1/health"):
            body = json.dumps({"status": "ok"}).encode()
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # noqa: D401 - silence the test server
        pass


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_detached_serve_status_logs_stop(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("HIVE_STATE_DIR", str(tmp_path / "state"))
    port = _free_port()
    server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    stacks = tmp_path / "stacks"
    stacks.mkdir()
    stack = stacks / "endpoint.json"
    stack.write_text(json.dumps({
        "name": "endpoint",
        "version": 1,
        "tiers": [{
            "role": "face", "repo": "local", "file": "fake.gguf",
            "endpoint": f"http://127.0.0.1:{port}/v1",
            "expect_model": "fake-model", "ctx": 8192, "backend": "endpoint",
        }],
    }), encoding="utf-8")

    name = "cli-detach-test"
    record = None
    try:
        rc = cli.main(["serve", str(stack), "--detach", "--name", name,
                       "--stacks-dir", str(stacks), "--timeout", "20"])
        assert rc == 0, capsys.readouterr().err
        capsys.readouterr()

        record = serve_state.read_record(name)
        assert record is not None
        assert serve_state.pid_alive(record["pid"])
        assert record["mode"] == "detached"

        assert cli.main(["status", "--json"]) == 0
        status = json.loads(capsys.readouterr().out)
        row = next(serve for serve in status["serves"] if serve["name"] == name)
        assert row["state"] == "running"
        assert row["tiers"][0]["probe"]["up"] is True
        assert row["tiers"][0]["probe"]["model"] == "fake-model"

        assert cli.main(["logs", name, "--json"]) == 0
        logs = json.loads(capsys.readouterr().out)
        assert any("applied" in line for line in logs["lines"])

        assert cli.main(["stop", name]) == 0
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and serve_state.pid_alive(record["pid"]):
            time.sleep(0.1)
        assert not serve_state.pid_alive(record["pid"])
        assert serve_state.read_record(name) is None
    finally:
        record = serve_state.read_record(name)
        if record and serve_state.pid_alive(record["pid"]):
            serve_state.terminate(record["pid"], force=True, timeout=5)
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
