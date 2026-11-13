"""Managed launch tiers: generated config, start, health, and stop.

These use a tiny fake engine repo (the real Strata/DeepInfer checkouts are not
needed) so the lifecycle is exercised in CI without a large model.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import time
from pathlib import Path

import pytest

from conftest import parse_json, run_rust

FAKE_SERVER = """#!{python}
import argparse, json, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

parser = argparse.ArgumentParser()
parser.add_argument("--config")
parser.add_argument("--port", type=int)
parser.add_argument("--engine", default="strata")
args = parser.parse_args()
config = json.load(open(args.config))
print("CONFIG", json.dumps(config), flush=True)

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/v1/models", "/models"):
            payload = {{"data": [{{"id": "fake-launch"}}]}}
        elif self.path == "/health":
            payload = {{"status": "ok", "loaded": True,
                        "model": "fake-launch", "max_context": 4096}}
        else:
            self.send_error(404)
            return
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass

ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
"""


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _fake_repo(tmp_path: Path, kind: str) -> Path:
    repo = tmp_path / f"{kind}-repo"
    (repo / "harness").mkdir(parents=True)
    (repo / "pyproject.toml").write_text("")
    (repo / "harness" / "paths.py").write_text("")
    (repo / ".venv" / "bin").mkdir(parents=True)
    (repo / ".venv" / "bin" / "python").symlink_to(sys.executable)
    (repo / "cfg.json").write_text(
        json.dumps(
            {
                "port": 1,
                "args": ["--max-context", "1", "--prefill", "512"],
                "ctx": 1,
                "engine_args": [],
            }
        )
    )
    if kind == "deepinfer":
        facade = repo / "tools" / "deepinfer" / "serve.py"
        facade.parent.mkdir(parents=True)
    else:
        facade = repo / "serve" / "server.py"
        facade.parent.mkdir(parents=True)
    facade.write_text(FAKE_SERVER.format(python=sys.executable))
    facade.chmod(0o755)
    return repo


def _stack(tmp_path: Path, kind: str, port: int) -> Path:
    repo = _fake_repo(tmp_path, kind)
    stack = {
        "name": f"{kind}-one",
        "version": 1,
        "tiers": [
            {
                "role": "face",
                "repo": "local",
                "file": "unused.gguf",
                "ctx": 2048,
                "endpoint": f"http://127.0.0.1:{port}/v1",
                "launch": {
                    "kind": kind,
                    "repo": str(repo),
                    "config": "cfg.json",
                    "python": ".venv/bin/python",
                    "engine_args": ["--flag", "7"],
                    "timeout_s": 30,
                },
            }
        ],
    }
    path = tmp_path / f"{kind}.json"
    path.write_text(json.dumps(stack) + "\n")
    return path


def _env(env: dict[str, str], tmp_path: Path, repo_root: Path | None = None) -> dict[str, str]:
    result = {
        **env,
        "HIVE_HOME": str(tmp_path / "home"),
        "HIVE_STATE_DIR": str(tmp_path / "state"),
    }
    if repo_root is not None:
        result["HIVE_SOURCE_ROOT"] = str(repo_root)
    return result


@pytest.mark.parametrize("kind", ["strata", "deepinfer"])
def test_launch_once_generates_config_and_stops(rust_bin, cli_env, tmp_path, kind):
    env, _ = cli_env
    port = _free_port()
    stack = _stack(tmp_path, kind, port)
    repo = tmp_path / f"{kind}-repo"
    result = run_rust(
        rust_bin,
        [
            "serve",
            str(stack),
            "--once",
            "--json",
            "--models-dir",
            str(tmp_path / "models"),
            "--log-dir",
            str(tmp_path / "logs"),
        ],
        _env(env, tmp_path, repo if kind == "deepinfer" else None),
    )
    assert result.returncode == 0, result.stderr
    payload = parse_json(result)
    tier = payload["tiers"][0]
    assert tier["managed"] is True
    assert tier["model"] == "fake-launch"
    assert tier["pid"]
    assert tier["port"] == port

    generated = tmp_path / "state" / kind / f"{kind}-{kind}-one-face.json"
    assert generated.is_file(), list((tmp_path / "state").rglob("*.json"))
    config = json.loads(generated.read_text())
    assert config["port"] == port
    if kind == "strata":
        assert config["args"][:4] == ["--max-context", "2048", "--prefill", "512"]
        assert config["log"].endswith(f"strata_{port}.engine.log")
    else:
        assert config["ctx"] == 2048
        assert config["engine_args"] == ["--flag", "7"]

    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", port), timeout=0.5)


def test_detached_launch_adopts_status_and_stops(rust_bin, cli_env, tmp_path):
    env, _ = cli_env
    port = _free_port()
    stack = _stack(tmp_path, "strata", port)
    env = _env(env, tmp_path)
    result = run_rust(
        rust_bin,
        [
            "serve",
            str(stack),
            "--detach",
            "--json",
            "--name",
            "launch",
            "--models-dir",
            str(tmp_path / "models"),
            "--log-dir",
            str(tmp_path / "logs"),
        ],
        env,
    )
    assert result.returncode == 0, result.stderr
    record = parse_json(result)
    supervisor = record["pid"]
    engine = record["tiers"][0]["pid"]
    try:
        status = run_rust(rust_bin, ["status", "--json"], env)
        assert status.returncode == 0
        view = parse_json(status)["serves"][0]
        assert view["state"] == "running"
        assert view["tiers"][0]["probe"]["up"] is True
        assert view["tiers"][0]["managed"] is True
    finally:
        stop = run_rust(rust_bin, ["stop", "launch", "--json"], env)
        assert stop.returncode == 0, stop.stderr
    deadline = time.time() + 5
    while time.time() < deadline:
        try:
            os.kill(engine, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        pytest.fail("launch engine survived stop")
    try:
        os.kill(supervisor, 9)
    except ProcessLookupError:
        pass
