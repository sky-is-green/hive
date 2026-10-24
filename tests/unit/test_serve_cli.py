"""``hive serve`` — the CLI owns the lifecycle and stays engine-blind."""

import json

import harness.serve as serve


class _Models:
    last_kwargs = None

    def __init__(self, **kwargs):
        _Models.last_kwargs = kwargs


class _Manager:
    instances = []

    def __init__(self, models, *, stacks_root=None):
        self.models = models
        self.stacks_root = stacks_root
        self.applied = None
        self.unloads = 0
        _Manager.instances.append(self)

    def apply(self, stack):
        self.applied = stack
        return {"ok": True, "tiers": [{"role": "face", "port": 8080, "key": "k"}]}

    def status(self):
        return {
            "ok": True,
            "stack": self.applied.name if self.applied else None,
            "tiers": [
                {"role": "face", "port": 8080, "base_url": None, "ctx": 8192, "backend": "rocm"},
                {"role": "worker", "port": 8081, "base_url": "http://127.0.0.1:9000/v1",
                 "ctx": 32768, "backend": "strata"},
            ],
            "warnings": [],
        }

    def unload(self):
        self.unloads += 1
        return {"ok": True, "unloaded": []}


class _RefusingManager(_Manager):
    def apply(self, stack):
        raise RuntimeError("stack 't' refused: over budget")


def _stack_file(tmp_path, name="t"):
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps({
        "name": name,
        "version": 1,
        "tiers": [{"role": "face", "repo": "org/model", "file": "m.gguf"}],
    }), encoding="utf-8")
    return path


def test_serve_once_applies_and_unloads(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(serve, "LlamaServerManager", _Models)
    monkeypatch.setattr(serve, "StackManager", _Manager)
    _Manager.instances.clear()

    rc = serve.main([str(_stack_file(tmp_path)), "--once"])

    out = capsys.readouterr().out
    assert rc == 0
    manager = _Manager.instances[-1]
    assert manager.applied.name == "t"
    assert manager.unloads == 1
    # a spawned tier is addressed by its port, an endpoint tier by base_url
    assert "http://127.0.0.1:8080/v1" in out
    assert "http://127.0.0.1:9000/v1" in out


def test_serve_list(tmp_path, capsys):
    _stack_file(tmp_path)
    rc = serve.main(["--list", "--stacks-dir", str(tmp_path)])
    assert rc == 0
    assert "t" in capsys.readouterr().out


def test_serve_missing_stack_is_usage_error(tmp_path, capsys):
    rc = serve.main([str(tmp_path / "nope.json"), "--once"])
    assert rc == 2
    assert "no stack file" in capsys.readouterr().err


def test_serve_apply_failure_returns_one(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(serve, "LlamaServerManager", _Models)
    monkeypatch.setattr(serve, "StackManager", _RefusingManager)
    rc = serve.main([str(_stack_file(tmp_path)), "--once"])
    assert rc == 1
    assert "refused" in capsys.readouterr().err
