"""Black-box parity: the same command must produce the same JSON.

Run after building the Rust binary::

    cargo build --manifest-path rust/Cargo.toml
    python -m pytest tests/conformance -q
"""

from __future__ import annotations

import json
import os
import re
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from conftest import (
    parse_json,
    proc_cmdline,
    run_both,
    run_python,
    run_rust,
    write_record,
)


def test_version_json_is_identical(rust_bin, cli_env):
    env, _ = cli_env
    py, rs = run_both(rust_bin, ["version", "--json"], env)
    assert py.returncode == rs.returncode == 0
    assert parse_json(py) == parse_json(rs)


def test_config_json_is_identical(rust_bin, cli_env, fixture_library, fixture_stacks):
    env, _ = cli_env
    args = [
        "config",
        "--models-dir",
        str(fixture_library),
        "--stacks-dir",
        str(fixture_stacks),
        "--json",
    ]
    py, rs = run_both(rust_bin, args, env)
    assert py.returncode == rs.returncode == 0
    assert parse_json(py) == parse_json(rs)


def test_models_list_matches_the_python_gguf_parser(rust_bin, cli_env, fixture_library):
    env, _ = cli_env
    args = ["models", "list", "--models-dir", str(fixture_library), "--json"]
    py, rs = run_both(rust_bin, args, env)
    assert py.returncode == rs.returncode == 0
    assert parse_json(py) == parse_json(rs)


def test_serve_model_local_generates_the_same_stack(
    rust_bin, cli_env, serve_fixture, tmp_path
):
    if not any(Path("/sys/class/drm").glob("card*/device/mem_info_vram_total")):
        pytest.skip("Python's residency preflight needs a visible GPU")
    env, _ = cli_env
    python_stacks = tmp_path / "python-stacks"
    rust_stacks = tmp_path / "rust-stacks"
    python_stacks.mkdir()
    rust_stacks.mkdir()
    model = serve_fixture["models"] / "tiny.gguf"
    common = [
        "serve",
        "--model",
        str(model),
        "--once",
        "--json",
        "--save",
        "generated",
        "--models-dir",
        str(serve_fixture["models"]),
        "--llama-server",
        str(serve_fixture["engine"]),
    ]
    py = run_python([*common, "--stacks-dir", str(python_stacks)], env)
    rs = run_rust(rust_bin, [*common, "--stacks-dir", str(rust_stacks)], env)
    assert py.returncode == rs.returncode == 0, (py.stderr, rs.stderr)
    python_saved = json.loads((python_stacks / "generated.json").read_text())
    rust_saved = json.loads((rust_stacks / "generated.json").read_text())
    assert python_saved == rust_saved
    assert python_saved["tiers"][0]["file"] == str(model.resolve())
    assert "routing" not in python_saved

    python_status = parse_json(py)
    rust_status = parse_json(rs)
    assert python_status["stack"] == rust_status["stack"] == "generated"
    for field in (
        "role",
        "key",
        "ctx",
        "model",
        "backend",
        "resident",
        "base_url",
        "caps",
        "managed",
        "pid",
        "tok_s",
        "vram_gb",
    ):
        assert python_status["tiers"][0][field] == rust_status["tiers"][0][field], field
    assert len(python_status["tiers"][0]["per_card"]) == len(
        rust_status["tiers"][0]["per_card"]
    )


def test_models_dir_read_matches(rust_bin, cli_env, fixture_library):
    env, _ = cli_env
    args = ["models", "dir", "--models-dir", str(fixture_library), "--json"]
    py, rs = run_both(rust_bin, args, env)
    assert py.returncode == rs.returncode == 0
    assert parse_json(py) == parse_json(rs)


def test_models_dir_set_matches(rust_bin, cli_env, tmp_path):
    env, _ = cli_env
    python_state = tmp_path / "python-state"
    rust_state = tmp_path / "rust-state"
    target = tmp_path / "new-library"
    py = run_python(
        ["models", "dir", str(target), "--json"],
        {**env, "HIVE_STATE_DIR": str(python_state)},
    )
    rs = run_rust(
        rust_bin,
        ["models", "dir", str(target), "--json"],
        {**env, "HIVE_STATE_DIR": str(rust_state)},
    )
    assert py.returncode == rs.returncode == 0
    assert parse_json(py) == parse_json(rs)
    assert (python_state / "models_dir.txt").read_text() == str(target)
    assert (rust_state / "models_dir.txt").read_text() == str(target)


def test_models_import_and_remove_match(rust_bin, cli_env, tmp_path):
    env, _ = cli_env
    source = tmp_path / "model.gguf"
    source.write_bytes(b"GGUF")
    python_models = tmp_path / "python-models"
    rust_models = tmp_path / "rust-models"
    python_state = tmp_path / "python-state"
    rust_state = tmp_path / "rust-state"
    python_env = {**env, "HIVE_STATE_DIR": str(python_state)}
    rust_env = {**env, "HIVE_STATE_DIR": str(rust_state)}
    import_args = ["models", "import", str(source), "--json"]
    py = run_python([*import_args, "--models-dir", str(python_models)], python_env)
    rs = run_rust(rust_bin, [*import_args, "--models-dir", str(rust_models)], rust_env)
    assert py.returncode == rs.returncode == 0
    assert _normalize_paths(parse_json(py), python_models) == _normalize_paths(
        parse_json(rs), rust_models
    )
    assert (python_models / "model.gguf").is_file()
    assert (rust_models / "model.gguf").is_file()

    py = run_python(
        ["models", "remove", "model.gguf", "--yes", "--json", "--models-dir", str(python_models)],
        python_env,
    )
    rs = run_rust(
        rust_bin,
        ["models", "remove", "model.gguf", "--yes", "--json", "--models-dir", str(rust_models)],
        rust_env,
    )
    assert py.returncode == rs.returncode == 0
    assert _normalize_paths(parse_json(py), python_models) == _normalize_paths(
        parse_json(rs), rust_models
    )
    assert not (python_models / "model.gguf").exists()
    assert not (rust_models / "model.gguf").exists()


def test_models_import_folder_matches(rust_bin, cli_env, tmp_path):
    env, _ = cli_env
    source = tmp_path / "folder" / "nested"
    source.mkdir(parents=True)
    (source / "a.gguf").write_bytes(b"GGUF")
    python_models = tmp_path / "python-models"
    rust_models = tmp_path / "rust-models"
    py = run_python(
        ["models", "import", str(source.parent), "--json", "--models-dir", str(python_models)],
        env,
    )
    rs = run_rust(
        rust_bin,
        ["models", "import", str(source.parent), "--json", "--models-dir", str(rust_models)],
        env,
    )
    assert py.returncode == rs.returncode == 0
    assert _normalize_paths(parse_json(py), python_models) == _normalize_paths(
        parse_json(rs), rust_models
    )


def test_stacks_list_matches(rust_bin, cli_env, fixture_stacks):
    env, _ = cli_env
    args = ["stacks", "list", "--stacks-dir", str(fixture_stacks), "--json"]
    py, rs = run_both(rust_bin, args, env)
    assert py.returncode == rs.returncode == 0
    assert parse_json(py) == parse_json(rs)


def _normalize_uptime(value):
    if isinstance(value, dict):
        return {
            key: (0 if key == "uptime_s" else _normalize_uptime(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_normalize_uptime(item) for item in value]
    return value


def test_status_no_probe_matches(rust_bin, cli_env, live_proc):
    env, state = cli_env
    write_record(
        state,
        "live",
        {
            "version": 1,
            "name": "live",
            "pid": live_proc.pid,
            "cmdline": proc_cmdline(live_proc.pid),
            "mode": "foreground",
            "stack": "fake-one",
            "host": "127.0.0.1",
            "started_at": 1_700_000_000.0,
            "log_path": None,
            "log_dir": str(state / "logs"),
            "tiers": [
                {"role": "face", "port": 1234, "ctx": 2048, "backend": "vulkan"}
            ],
        },
    )
    write_record(
        state,
        "stale",
        {
            "version": 1,
            "name": "stale",
            "pid": 999_999,
            "cmdline": "gone",
            "started_at": 1_700_000_001.0,
            "tiers": [],
        },
    )
    (state / "serve" / "broken.json").write_text("{not json")

    args = ["status", "--json", "--no-probe"]
    py, rs = run_both(rust_bin, args, env)
    assert py.returncode == rs.returncode == 0
    assert _normalize_uptime(parse_json(py)) == _normalize_uptime(parse_json(rs))


def test_status_probe_matches(rust_bin, cli_env, live_proc):
    env, state = cli_env
    server = _ModelServer()
    try:
        write_record(
            state,
            "live",
            {
                "version": 1,
                "name": "live",
                "pid": live_proc.pid,
                "cmdline": proc_cmdline(live_proc.pid),
                "host": "127.0.0.1",
                "started_at": 1_700_000_000.0,
                "tiers": [
                    {
                        "role": "face",
                        "port": server.port,
                        "ctx": 2048,
                        "backend": "vulkan",
                    }
                ],
            },
        )
        py, rs = run_both(rust_bin, ["status", "--json"], env)
        assert py.returncode == rs.returncode == 0
        assert _normalize_uptime(parse_json(py)) == _normalize_uptime(parse_json(rs))
    finally:
        server.close()


def test_stop_removes_a_stale_record_without_signalling(rust_bin, cli_env, tmp_path):
    env, _ = cli_env
    python_state = tmp_path / "python-state"
    rust_state = tmp_path / "rust-state"
    _write_stale(python_state)
    _write_stale(rust_state)
    python_env = {**env, "HIVE_STATE_DIR": str(python_state)}
    rust_env = {**env, "HIVE_STATE_DIR": str(rust_state)}
    py = run_python(["stop", "--json"], python_env)
    rs = run_rust(rust_bin, ["stop", "--json"], rust_env)
    assert py.returncode == rs.returncode == 0
    assert parse_json(py) == parse_json(rs)
    assert not (python_state / "serve" / "stale.json").exists()
    assert not (rust_state / "serve" / "stale.json").exists()


def test_logs_json_matches(rust_bin, cli_env, live_proc, tmp_path):
    env, state = cli_env
    log_dir = tmp_path / "engine-logs"
    log_dir.mkdir()
    (state / "serve").mkdir()
    supervisor_log = state / "serve" / "live.log"
    supervisor_log.write_text("line one\nline two\nline three\n")
    (log_dir / "llama_server_1234.log").write_text("engine a\nengine b\n")
    write_record(
        state,
        "live",
        {
            "version": 1,
            "name": "live",
            "pid": live_proc.pid,
            "cmdline": proc_cmdline(live_proc.pid),
            "started_at": 1_700_000_000.0,
            "log_path": str(supervisor_log),
            "log_dir": str(log_dir),
            "tiers": [
                {"role": "face", "port": 1234, "ctx": 2048, "backend": "vulkan"},
                {"role": "worker", "base_url": "http://127.0.0.1:9/v1"},
            ],
        },
    )
    for args in (
        ["logs", "live", "--json", "-n", "2"],
        ["logs", "live", "--tier", "face", "--json", "-n", "1"],
    ):
        py, rs = run_both(rust_bin, args, env)
        assert py.returncode == rs.returncode == 0
        assert parse_json(py) == parse_json(rs)


def test_stacks_show_and_validate_match(
    rust_bin, cli_env, fixture_stacks, fixture_library
):
    env, _ = cli_env
    for name in ("a-first", "b-second"):
        args = ["stacks", "show", name, "--stacks-dir", str(fixture_stacks)]
        py, rs = run_both(rust_bin, args, env)
        assert py.returncode == rs.returncode == 0, name
        assert py.stdout == rs.stdout, name

    args = [
        "stacks",
        "validate",
        "a-first",
        "--stacks-dir",
        str(fixture_stacks),
        "--models-dir",
        str(fixture_library),
        "--json",
    ]
    py, rs = run_both(rust_bin, args, env)
    assert py.returncode == rs.returncode
    assert parse_json(py) == parse_json(rs)

    for name in ("e-bad-role", "f-bad-engine-arg", "g-bad-launch"):
        args = ["stacks", "show", name, "--stacks-dir", str(fixture_stacks)]
        py, rs = run_both(rust_bin, args, env)
        assert py.returncode == rs.returncode == 1, name
        assert py.stderr == rs.stderr, name


def test_tui_requires_a_terminal(rust_bin, cli_env):
    env, _ = cli_env
    py, rs = run_both(rust_bin, ["tui"], env)
    assert py.returncode == rs.returncode == 1
    assert py.stderr == rs.stderr


def test_setup_no_fetch_error_matches(rust_bin, cli_env, tmp_path):
    env, _ = cli_env
    empty_home = tmp_path / "empty-home"
    empty_home.mkdir()
    fresh = {
        **env,
        "HIVE_HOME": str(empty_home),
        "HIVE_STATE_DIR": str(empty_home / "state"),
    }
    args = [
        "setup",
        "--no-fetch",
        "--json",
        "--models-dir",
        str(tmp_path / "models"),
        "--stacks-dir",
        str(tmp_path / "stacks"),
    ]
    py, rs = run_both(rust_bin, args, fresh)
    assert py.returncode == rs.returncode == 1
    assert py.stderr == rs.stderr


def test_backends_json_matches(rust_bin, cli_env):
    env, _ = cli_env
    for args in (["backends", "--json"], ["backends", "list", "--json"]):
        py, rs = run_both(rust_bin, args, env)
        assert py.returncode == rs.returncode == 0
        assert parse_json(py) == parse_json(rs)


def _normalize_paths(value, root: Path):
    if isinstance(value, dict):
        return {key: _normalize_paths(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_paths(item, root) for item in value]
    if isinstance(value, str):
        return value.replace(str(root), "<root>")
    return value


def test_help_catalog_is_identical(rust_bin, cli_env):
    env, _ = cli_env
    for topic in (
        "serve",
        "tui",
        "status",
        "stop",
        "logs",
        "models list",
        "models search",
        "models files",
        "models dir",
        "models download",
        "models import",
        "models remove",
        "stacks list",
        "stacks show",
        "stacks validate",
        "backends",
        "backends list",
        "backends fetch",
        "setup",
        "doctor",
        "config",
        "version",
        "help",
    ):
        args = ["help", *topic.split(), "--json"]
        py, rs = run_both(rust_bin, args, env)
        assert py.returncode == rs.returncode == 0, topic
        assert parse_json(py) == parse_json(rs), topic

    py, rs = run_both(rust_bin, ["help", "--json"], env)
    python_root = parse_json(py)
    rust_root = parse_json(rs)
    assert rust_root["options"] == python_root["options"]
    assert rust_root["description"] == python_root["description"]
    _assert_catalog_subset(rust_root, python_root)


def _assert_catalog_subset(rust: dict, python: dict) -> None:
    """Every command the Rust binary advertises must match Python exactly.

    Commands that Phase 1 has not ported yet are allowed to be absent from the
    Rust catalog; a ported command may not drift.
    """
    for key in ("name", "path", "prog", "description", "options", "arguments"):
        assert rust[key] == python[key], key
    if "aliases" in rust:
        assert rust["aliases"] == python.get("aliases"), "aliases"
    python_commands = {row["name"]: row for row in python["commands"]}
    for row in rust["commands"]:
        assert row["name"] in python_commands, row["name"]
        _assert_catalog_subset(row, python_commands[row["name"]])


def test_doctor_checks_and_stable_details_match(
    rust_bin, cli_env, fixture_library, fixture_stacks
):
    env, _ = cli_env
    args = [
        "doctor",
        "--models-dir",
        str(fixture_library),
        "--stacks-dir",
        str(fixture_stacks),
        "--json",
    ]
    py, rs = run_both(rust_bin, args, env)
    assert py.returncode == rs.returncode
    python_payload = parse_json(py)
    rust_payload = parse_json(rs)
    assert rust_payload["ok"] == python_payload["ok"]
    assert rust_payload["summary"] == python_payload["summary"]
    python_checks = {row["name"]: row for row in python_payload["checks"]}
    rust_checks = {row["name"]: row for row in rust_payload["checks"]}
    assert list(rust_checks) == list(python_checks)
    for name, row in rust_checks.items():
        assert row["status"] == python_checks[name]["status"], name
        if name in {"python", "hive", "platform", "stacks", "state", "backends", "rocm", "hf-token"}:
            assert row["detail"] == python_checks[name]["detail"], name
        elif name == "memory":
            assert re.sub(r"\d+(?:\.\d+)?", "<n>", row["detail"]) == re.sub(
                r"\d+(?:\.\d+)?", "<n>", python_checks[name]["detail"]
            )
        elif name == "models":
            assert row["detail"].startswith(str(fixture_library))
            assert "GGUF" in row["detail"] and "GiB free" in row["detail"]
        elif name == "cards":
            assert re.sub(r"\d+(?:\.\d+)?", "<n>", row["detail"]) == re.sub(
                r"\d+(?:\.\d+)?", "<n>", python_checks[name]["detail"]
            )


def test_serve_list_matches(rust_bin, cli_env, fixture_stacks):
    env, _ = cli_env
    args = ["serve", "--list", "--stacks-dir", str(fixture_stacks), "--json"]
    py, rs = run_both(rust_bin, args, env)
    assert py.returncode == rs.returncode == 0
    assert parse_json(py) == parse_json(rs)


def test_serve_once_starts_and_stops_a_real_process(
    rust_bin, cli_env, serve_fixture
):
    env, state = cli_env
    args = [
        "serve",
        str(serve_fixture["stack"]),
        "--once",
        "--json",
        "--models-dir",
        str(serve_fixture["models"]),
        "--stacks-dir",
        str(serve_fixture["stacks"]),
        "--llama-server",
        str(serve_fixture["engine"]),
    ]
    result = run_rust(rust_bin, args, env)
    assert result.returncode == 0, result.stderr
    payload = parse_json(result)
    assert payload["ok"] is True
    assert payload["stack"] == "fake-one"
    tier = payload["tiers"][0]
    assert tier["role"] == "face"
    assert tier["model"] == "fake-model"
    assert tier["ctx"] == 2048
    assert tier["port"] > 0
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", tier["port"]), timeout=0.5)
    assert not list((state / "serve").glob("*.json"))


def test_detached_serve_survives_status_and_kill_9_cleans_the_engine(
    rust_bin, cli_env, serve_fixture
):
    env, state = cli_env
    args = [
        "serve",
        str(serve_fixture["stack"]),
        "--detach",
        "--json",
        "--name",
        "detached",
        "--models-dir",
        str(serve_fixture["models"]),
        "--stacks-dir",
        str(serve_fixture["stacks"]),
        "--llama-server",
        str(serve_fixture["engine"]),
    ]
    result = run_rust(rust_bin, args, env)
    assert result.returncode == 0, result.stderr
    record = parse_json(result)
    supervisor = record["pid"]
    deadline = time.time() + 2
    engine = None
    while time.time() < deadline:
        children = Path(f"/proc/{supervisor}/task/{supervisor}/children").read_text().split()
        if children:
            engine = int(children[0])
            break
        time.sleep(0.05)
    assert engine is not None, "supervisor has no engine child"
    try:
        status = run_rust(rust_bin, ["status", "--json", "--no-probe"], env)
        assert status.returncode == 0
        view = parse_json(status)["serves"][0]
        assert view["name"] == "detached"
        assert view["state"] == "running"
        assert view["tiers"][0]["pid"] is None

        logs = run_rust(rust_bin, ["logs", "detached", "--json", "-n", "20"], env)
        assert logs.returncode == 0
        assert "serving" in "\n".join(parse_json(logs)["lines"])

        os.kill(supervisor, 9)
        deadline = time.time() + 5
        while time.time() < deadline:
            try:
                os.kill(engine, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        else:
            pytest.fail("engine survived a kill -9 of its supervisor")

        pruned = run_rust(rust_bin, ["status", "--json", "--prune"], env)
        assert pruned.returncode == 0
        assert parse_json(pruned)["pruned"] == ["detached"]
    finally:
        for pid in (supervisor, engine):
            try:
                os.kill(pid, 9)
            except ProcessLookupError:
                pass


class _ModelServer:
    """A local OpenAI-compatible endpoint for the status probe test."""

    def __init__(self) -> None:
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @staticmethod
    def _handler():
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path != "/v1/models":
                    self.send_error(404)
                    return
                body = json.dumps({"data": [{"id": "fake-model"}]}).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args):
                pass

        return Handler

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=2)


def _write_stale(state: Path) -> None:
    write_record(
        state,
        "stale",
        {
            "version": 1,
            "name": "stale",
            "pid": 999_999,
            "cmdline": "gone",
            "started_at": 1_700_000_000.0,
            "tiers": [],
        },
    )
