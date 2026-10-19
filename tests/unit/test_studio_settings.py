"""Compaction, engine surgery, and the Studio toggles."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from harness.app import create_app


@pytest.fixture()
def client(tmp_path):
    app = create_app(
        providers_file=tmp_path / "providers.local.json",
        engines_file=tmp_path / "engines.local.json",
        runs_root=tmp_path / "runs",
        log_dir=str(tmp_path / "logs"),
        state_dir=str(tmp_path / "state"),
    )
    return TestClient(app)


MESSAGES = [
    {"role": "user", "text": "Plan the deployment of the 176B serve."},
    {"role": "assistant", "text": "We decided to run strata on :8081 with a 3072 MiB reserve."},
    {"role": "user", "text": "Now the cascade run."},
    {"role": "assistant", "text": "The cascade will use the organic-66 set with the deterministic judge."},
]


def test_settings_round_trip(client):
    body = client.get("/v1/studio/settings").json()
    assert body["sandbox_enabled"] is True
    assert body["tools_enabled"] is True
    updated = client.put("/v1/studio/settings",
                         json={"sandbox_enabled": False, "tools_enabled": False}).json()
    assert updated["sandbox_enabled"] is False
    assert updated["tools_enabled"] is False
    again = client.get("/v1/studio/settings").json()
    assert again["sandbox_enabled"] is False


def test_sandbox_off_blocks_execution(client):
    client.put("/v1/studio/settings", json={"sandbox_enabled": False})
    r = client.post("/v1/sandbox/exec", json={"command": "echo hi"})
    assert r.status_code == 403


def test_tools_off_blocks_the_agent(client):
    client.put("/v1/studio/settings", json={"tools_enabled": False})
    r = client.post("/v1/agent/stream", json={"message": "hello", "conversation_id": "c"})
    assert r.status_code == 403


def test_compaction_produces_a_checkpoint(client):
    # keep=10 tokens: the older half is summarized, the newest stays verbatim
    r = client.post("/v1/compaction/run",
                    json={"messages": MESSAGES, "trigger": "manual", "keep": 10})
    body = r.json()
    assert body["status"] == "completed", body
    cp = body["checkpoint"]
    assert "## Goal" in cp["summary"]       # the template headings
    assert "## State" in cp["summary"]
    assert cp["rendered"].startswith("<conversation-checkpoint>")
    kinds = [e["event"] for e in body["events"]]
    assert "compaction.started" in kinds
    assert "compaction.completed" in kinds


def test_compaction_needs_messages(client):
    r = client.post("/v1/compaction/run", json={"messages": []})
    assert r.status_code == 422


def test_compaction_status_endpoint(client):
    body = client.get("/v1/compaction/status").json()
    assert body["auto"] is True
    assert "## Goal" in body["headings"]


def test_surgery_without_candidates_is_unavailable(client):
    r = client.post("/v1/compaction/surgery",
                    json={"checkpoint": {"summary": "s", "recent": "user: hi"},
                          "current": "a", "candidates": []})
    body = r.json()
    assert body["status"] == "unavailable"
    assert "reason" in body


def test_strata_probe_never_raises(client):
    body = client.get("/v1/studio/strata").json()
    assert "alive" in body
    assert "base_url" in body


def test_dev_mode_defaults_off_and_toggles(client):
    body = client.get("/v1/studio/settings").json()
    assert body["dev_mode"] is False
    r = client.post("/v1/commands/run", json={"line": "/dev on", "conversation_id": "c"})
    assert r.json()["kind"] == "success"
    assert client.get("/v1/studio/settings").json()["dev_mode"] is True
    client.post("/v1/commands/run", json={"line": "/dev off", "conversation_id": "c"})
    assert client.get("/v1/studio/settings").json()["dev_mode"] is False


def test_dev_command_toggles_without_an_argument(client):
    r = client.post("/v1/commands/run", json={"line": "/dev", "conversation_id": "c"})
    assert r.json()["kind"] == "success"
    assert client.get("/v1/studio/settings").json()["dev_mode"] is True


def test_dev_command_is_discoverable(client):
    names = {c["name"] for c in client.get("/v1/commands").json()["commands"]}
    assert "dev" in names
