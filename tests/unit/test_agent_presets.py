"""Agent preset write safety: user presets only, validated ids, atomic write."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import harness.app as app_mod
from backend.openai_compat import OpenAICompatBackend
from cortex.e2e import FakeUltraSmall, MockTransport
from harness.app import create_app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    user_root = tmp_path / "user-presets"
    user_root.mkdir()
    monkeypatch.setattr(app_mod, "_agent_presets_user_root", lambda: user_root)

    def backend_factory(model=None):
        return OpenAICompatBackend(
            base_url="http://mock",
            model=model or "mock-model",
            transport=MockTransport(latency_ms=0),
        )

    app = create_app(
        ultra_factory=FakeUltraSmall,
        backend_factory=backend_factory,
        runs_root=tmp_path / "runs",
        providers_file=tmp_path / "providers.local.json",
        engines_file=tmp_path / "engines.local.json",
        log_dir=str(tmp_path / "logs"),
        state_dir=str(tmp_path / "state"),
    )
    with TestClient(app) as test_client:
        yield test_client, user_root


def _user_preset(root: Path, preset_id: str, content: str = "name: test\n") -> Path:
    folder = root / preset_id
    folder.mkdir(parents=True)
    (folder / "preset.yml").write_text("name: Test preset\n", encoding="utf-8")
    (folder / "agent.cordis.yml").write_text(content, encoding="utf-8")
    return folder


def test_write_updates_an_existing_user_preset(client):
    c, root = client
    folder = _user_preset(root, "mine", "old: true\n")

    r = c.post("/v1/agent-presets/write", json={
        "agentPreset": "mine",
        "content": "new: true\n",
    })

    assert r.status_code == 200, r.text
    assert (folder / "agent.cordis.yml").read_text(encoding="utf-8") == "new: true\n"
    assert not list(folder.glob("*.tmp")), "the atomic temp file must be gone"
    assert r.json()["trust"] == "user"


def test_write_never_touches_a_system_preset(client):
    c, _root = client

    # There is no user copy named "standard"; the route must not create one.
    r = c.post("/v1/agent-presets/write", json={
        "agentPreset": "standard",
        "content": "tampered: true\n",
    })

    assert r.status_code == 404, r.text


@pytest.mark.parametrize("preset_id", ["../escape", "Bad Id", "a/b", ""])
def test_write_rejects_invalid_ids(client, preset_id):
    c, _root = client

    r = c.post("/v1/agent-presets/write", json={
        "agentPreset": preset_id,
        "content": "x: y\n",
    })

    assert r.status_code in (404, 422), r.text


def test_write_rejects_empty_content(client):
    c, root = client
    _user_preset(root, "mine")

    r = c.post("/v1/agent-presets/write", json={
        "agentPreset": "mine",
        "content": "",
    })

    assert r.status_code == 422, r.text


def test_read_and_list_mark_the_user_tree(client):
    c, root = client
    _user_preset(root, "mine", "hello: world\n")

    read = c.post("/v1/agent-presets/read", json={"agentPreset": "mine"})
    assert read.status_code == 200, read.text
    assert read.json()["content"] == "hello: world\n"
    assert read.json()["trust"] == "user"

    listed = c.get("/v1/agent-presets/list").json()["presets"]
    mine = next(p for p in listed if p["id"] == "mine")
    assert mine["trust"] == "user"
