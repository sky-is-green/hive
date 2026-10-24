"""The ``hive`` command surface: parsing, JSON, and the safe paths."""
import json
import os
import sys
import time
from pathlib import Path

import pytest

import harness.cli as cli
import harness.serve as serve
from harness import serve_state


def _endpoint_stack(path: Path, name: str = "demo",
                    endpoint: str = "http://127.0.0.1:9/v1") -> Path:
    path.write_text(json.dumps({
        "name": name,
        "version": 1,
        "tiers": [{
            "role": "face", "repo": "local", "file": "demo.gguf",
            "endpoint": endpoint, "ctx": 8192, "backend": "endpoint",
        }],
    }), encoding="utf-8")
    return path


def _record(name: str = "demo", **overrides) -> dict:
    record = {
        "name": name,
        "pid": os.getpid(),
        "mode": "foreground",
        "stack": "demo",
        "host": "127.0.0.1",
        "started_at": time.time(),
        "cmdline": serve_state.pid_cmdline(os.getpid()),
        "log_path": None,
        "log_dir": "/tmp",
        "tiers": [{"role": "face", "port": 1234, "ctx": 8192,
                   "backend": "rocm", "base_url": None}],
    }
    record.update(overrides)
    return record


def test_root_help_version_and_unknown(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert "hive" in capsys.readouterr().out

    assert cli.main([]) == 2
    assert "usage: hive" in capsys.readouterr().err

    with pytest.raises(SystemExit) as exc:
        cli.main(["not-a-command"])
    assert exc.value.code == 2


def test_version_and_config(capsys):
    assert cli.main(["version"]) == 0
    assert capsys.readouterr().out.startswith("hive ")

    assert cli.main(["config", "--json", "--models-dir", "/tmp/hive-models",
                     "--stacks-dir", "/tmp/hive-stacks"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["models_dir"] == "/tmp/hive-models"
    assert payload["stacks_dir"] == "/tmp/hive-stacks"
    assert "state_dir" in payload


def test_stacks_list_show_validate(tmp_path, capsys):
    stacks = tmp_path / "stacks"
    stacks.mkdir()
    _endpoint_stack(stacks / "demo.json")

    assert cli.main(["stacks", "list", "--stacks-dir", str(stacks), "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["name"] == "demo"

    assert cli.main(["stacks", "show", "demo", "--stacks-dir", str(stacks)]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["tiers"][0]["role"] == "face"

    assert cli.main(["stacks", "validate", "demo",
                     "--stacks-dir", str(stacks), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["ok"] is True
    assert result["tiers"][0]["detail"].startswith("endpoint")


def test_stacks_validate_names_a_missing_backend(tmp_path, capsys):
    stacks = tmp_path / "stacks"
    stacks.mkdir()
    stack = stacks / "bad.json"
    stack.write_text(json.dumps({
        "name": "bad", "version": 1,
        "tiers": [{"role": "face", "repo": "local", "file": "m.gguf",
                   "backend": "no-such-backend"}],
    }), encoding="utf-8")
    assert cli.main(["stacks", "validate", "bad", "--stacks-dir", str(stacks),
                     "--models-dir", str(tmp_path / "models")]) == 1
    out = capsys.readouterr()
    assert "no-such-backend" in out.out + out.err


def test_models_list_remove_and_import(tmp_path, capsys):
    models = tmp_path / "models"
    models.mkdir()
    (models / "a.gguf").write_bytes(b"")
    assert cli.main(["models", "list", "--models-dir", str(models), "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [row["name"] for row in rows] == ["a"]
    assert "list_local:" not in capsys.readouterr().out

    assert cli.main(["models", "remove", "a.gguf",
                     "--models-dir", str(models)]) == 2
    assert (models / "a.gguf").exists()

    assert cli.main(["models", "remove", "a.gguf", "--yes",
                     "--models-dir", str(models)]) == 0
    assert not (models / "a.gguf").exists()
    capsys.readouterr()  # drop the "removed" line before the JSON check

    source = tmp_path / "source.gguf"
    source.write_bytes(b"GGUF")
    assert cli.main(["models", "import", str(source), "--copy",
                     "--models-dir", str(models), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["imported"] == "source.gguf"
    assert (models / "source.gguf").is_file()


def test_models_search_and_files(capsys, monkeypatch):
    class _Hub:
        def hub_search(self, query, limit=12):
            assert query == "qwen"
            return [{"repo": "org/qwen", "downloads": 10, "likes": 2,
                     "last_modified": "2026-01-01", "license": "apache-2.0"}]

        def hub_files(self, repo):
            assert repo == "org/qwen"
            return [{"file": "q.gguf", "size_gb": 4.2}]

    monkeypatch.setattr(cli, "_models_manager", lambda args: _Hub())
    assert cli.main(["models", "search", "qwen", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["repo"] == "org/qwen"

    assert cli.main(["models", "files", "org/qwen", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["file"] == "q.gguf"


def test_models_download_mocked(tmp_path, capsys, monkeypatch):
    hub = pytest.importorskip("huggingface_hub")

    def fake_download(*, repo_id, filename, revision, local_dir):
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"GGUF")
        return str(path)

    monkeypatch.setattr(hub, "hf_hub_download", fake_download)
    models = tmp_path / "models"
    assert cli.main(["models", "download", "org/repo:file.gguf",
                     "--models-dir", str(models), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["repo"] == "org/repo"
    assert payload["file"] == "file.gguf"
    assert (models / "file.gguf").is_file()


def test_status_json_and_stop(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("HIVE_STATE_DIR", str(tmp_path))
    serve_state.write_record(_record())

    assert cli.main(["status", "--no-probe", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["serves"][0]["name"] == "demo"
    assert payload["serves"][0]["state"] == "running"

    calls = []
    monkeypatch.setattr(serve_state, "terminate",
                        lambda pid, force=False, timeout=15.0: (calls.append(pid) or (True, "stopped")))
    assert cli.main(["stop", "demo", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert calls == [os.getpid()]
    assert serve_state.read_record("demo") is None


@pytest.mark.skipif(sys.platform == "win32",
                    reason="no /proc command-line fingerprint on Windows")
def test_stop_never_kills_a_reused_pid(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("HIVE_STATE_DIR", str(tmp_path))
    serve_state.write_record(_record(cmdline="something else entirely"))
    calls = []
    monkeypatch.setattr(serve_state, "terminate",
                        lambda *a, **k: (calls.append("called") or (True, "stopped")))
    assert cli.main(["stop", "demo"]) == 0
    assert calls == []
    assert serve_state.read_record("demo") is None
    assert "stale" in capsys.readouterr().out


@pytest.mark.skipif(sys.platform == "win32",
                    reason="no /proc command-line fingerprint on Windows")
def test_status_prune(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("HIVE_STATE_DIR", str(tmp_path))
    serve_state.write_record(_record(cmdline="gone"))
    assert cli.main(["status", "--prune", "--no-probe"]) == 0
    assert serve_state.read_record("demo") is None
    assert "pruned" in capsys.readouterr().out


def test_logs_supervisor_and_tier(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("HIVE_STATE_DIR", str(tmp_path))
    log = tmp_path / "serve.log"
    log.write_text("one\ntwo\nthree\n", encoding="utf-8")
    tier_log = tmp_path / "llama_server_1234.log"
    tier_log.write_text("engine-a\nengine-b\n", encoding="utf-8")
    serve_state.write_record(_record(log_path=str(log), log_dir=str(tmp_path)))

    assert cli.main(["logs", "demo", "-n", "2", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["lines"] == ["two", "three"]

    assert cli.main(["logs", "demo", "--tier", "face", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["source"] == "tier face"
    assert payload["lines"] == ["engine-a", "engine-b"]


def test_doctor_json_with_a_declared_backend(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(cli, "linux_amd_devices", lambda: [])
    monkeypatch.setattr(cli.backend_manifest, "rows", lambda: [{
        "name": "rocm", "installed": True, "binary": "/bin/true",
        "accelerators": ["amd"], "label": "ROCm",
    }])
    assert cli.main(["doctor", "--json", "--models-dir", str(tmp_path / "models"),
                     "--stacks-dir", str(tmp_path / "stacks")]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert any(check["name"] == "backends" for check in payload["checks"])


def test_serve_missing_stack_is_a_usage_error(capsys):
    assert serve.main(["/tmp/does-not-exist.json", "--once"]) == 2
    assert "no stack file" in capsys.readouterr().err


class _FakeModels:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.log_dir = Path("/tmp")
        _FakeModels.instances.append(self)


class _FakeManager:
    instances = []

    def __init__(self, models, **kwargs):
        self.models = models
        self.applied = None
        self.unloads = 0
        _FakeManager.instances.append(self)

    def apply(self, stack):
        self.applied = stack
        return {"ok": True, "tiers": [{"role": "face", "port": 8080}]}

    def status(self):
        return {"ok": True, "stack": self.applied.name if self.applied else None,
                "tiers": [{"role": "face", "port": 8080, "ctx": 8192,
                           "backend": "vulkan", "base_url": None}],
                "warnings": []}

    def unload(self):
        self.unloads += 1
        return {"ok": True, "unloaded": ["face"]}


def test_serve_downloads_a_remote_model(tmp_path, monkeypatch, capsys):
    class _RemoteModels:
        def __init__(self, models_dir=None):
            self.models_dir = Path(models_dir).expanduser() if models_dir else tmp_path

        def resolve_model(self, name):
            return None

    monkeypatch.setattr(serve, "LlamaServerManager", _RemoteModels)
    hub = pytest.importorskip("huggingface_hub")

    def fake_download(*, repo_id, filename, local_dir):
        path = Path(local_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"GGUF")
        return str(path)

    monkeypatch.setattr(hub, "hf_hub_download", fake_download)
    serve._ensure_hf_model("org/repo", "m.gguf", str(tmp_path))
    assert (tmp_path / "m.gguf").is_file()
    assert "downloading org/repo:m.gguf" in capsys.readouterr().err


def test_serve_accepts_a_local_gguf(tmp_path, monkeypatch):
    _FakeManager.instances.clear()
    monkeypatch.setattr(serve, "LlamaServerManager", _FakeModels)
    monkeypatch.setattr(serve, "StackManager", _FakeManager)
    model = tmp_path / "little.gguf"
    model.write_bytes(b"GGUF")
    assert serve.main(["--model", str(model), "--cards", "0", "--once"]) == 0
    stack = _FakeManager.instances[-1].applied
    assert stack is not None
    assert stack.tiers[0].repo == "local"
    assert Path(stack.tiers[0].file) == model.resolve()


def test_child_argv_keeps_every_flag():
    parser = serve._standalone_parser()
    args = parser.parse_args([
        "demo", "--detach", "--name", "unit", "--cards", "2",
        "--ctx", "4096", "--save", "saved", "--timeout", "42",
    ])
    argv = serve._child_argv(args, "unit")
    assert argv[:3] == [sys.executable, "-m", "harness.cli"]
    assert "serve" in argv and "demo" in argv
    assert "--name" in argv and argv[argv.index("--name") + 1] == "unit"
    assert "--_child" in argv
    assert "--detach" not in argv
    assert "--cards" in argv and argv[argv.index("--cards") + 1] == "2"
