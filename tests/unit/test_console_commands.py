"""The console command palette: discovery, and the cascade/sandbox commands."""

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


def test_commands_are_discoverable(client):
    body = client.get("/v1/commands").json()
    names = {c["name"] for c in body["commands"]}
    assert {"help", "status", "model", "provider", "engine", "cascade", "sandbox"} <= names
    for c in body["commands"]:
        assert c["description"], f"command {c['name']} has no description"


def test_run_command_rejects_an_unknown_command(client):
    r = client.post("/v1/commands/run", json={"line": "/definitely-not-a-command"})
    assert r.status_code == 200
    assert r.json()["kind"] == "error"


def test_sandbox_command_runs_a_shell_command(client):
    r = client.post("/v1/commands/run", json={"line": "/sandbox echo hello-sandbox"})
    body = r.json()
    assert body["kind"] == "success", body
    assert "hello-sandbox" in body["text"]


def test_sandbox_command_needs_an_argument(client):
    r = client.post("/v1/commands/run", json={"line": "/sandbox"})
    assert r.json()["kind"] == "error"


def test_cascade_command_lists_runs(client):
    r = client.post("/v1/commands/run", json={"line": "/cascade"})
    body = r.json()
    assert body["kind"] == "success", body
    assert "cascade runs" in body["text"]


def test_cascade_command_rejects_a_bad_task_file(client):
    r = client.post("/v1/commands/run", json={"line": "/cascade not-a-file.txt"})
    assert r.json()["kind"] == "error"
