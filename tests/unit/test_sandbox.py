"""The sandbox refuses mutable images and never falls back to the host silently."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from harness import sandbox
from harness.app import create_app
from harness.state_store import StatePaths


@pytest.fixture()
def state(tmp_path):
    return StatePaths(tmp_path / "state").ensure()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    app = create_app(
        providers_file=tmp_path / "providers.local.json",
        engines_file=tmp_path / "engines.local.json",
        runs_root=tmp_path / "runs",
        log_dir=str(tmp_path / "logs"),
        state_dir=str(tmp_path / "state"),
    )
    return TestClient(app)


def test_default_policy_is_docker_and_not_pinned(state):
    policy = sandbox.SandboxPolicy(state)
    data = policy.read()
    assert data["mode"] == "docker"
    assert data["local_fallback"] == "approval"
    with pytest.raises(sandbox.SandboxError, match="not pinned"):
        policy.image_ref()


def test_a_digest_makes_the_image_reference_exact(state):
    policy = sandbox.SandboxPolicy(state)
    digest = "a" * 64
    policy.update(docker_image="hive-sandbox:1", image_digest=digest)
    assert policy.image_ref() == f"hive-sandbox:1@sha256:{digest}"


def test_image_digest_is_validated(state):
    policy = sandbox.SandboxPolicy(state)
    with pytest.raises(sandbox.SandboxError, match="64-character"):
        policy.update(image_digest="not-a-digest")


def test_host_network_mode_is_rejected(state):
    policy = sandbox.SandboxPolicy(state)
    with pytest.raises(sandbox.SandboxError, match="network must be"):
        policy.update(network="host")


def test_local_fallback_modes(state):
    policy = sandbox.SandboxPolicy(state)
    assert policy.local_fallback(None, approved=False)[0] is False
    assert policy.local_fallback("s1", approved=True)[0] is True
    assert policy.local_fallback(None, approved=True)[0] is False
    policy.update(local_fallback="deny")
    assert policy.local_fallback("s1", approved=True)[0] is False
    policy.update(local_fallback="allow")
    assert policy.local_fallback(None, approved=False)[0] is True


def test_pin_accepts_an_offline_digest(state):
    policy = sandbox.SandboxPolicy(state)
    data = sandbox.pin_image(policy, "hive-sandbox:1", "b" * 64)
    assert data["docker_image"] == "hive-sandbox:1"
    assert data["image_digest"] == "b" * 64


def test_local_fallback_requires_approval_with_a_session(client, monkeypatch):
    monkeypatch.setattr(sandbox, "docker_status", lambda: {
        "installed": True, "daemon": False, "binary": "/usr/bin/docker",
        "detail": "docker daemon is not running"})

    denied = client.post("/v1/sandbox/exec", json={
        "command": "echo hi", "session_id": "s1"})
    assert denied.status_code == 409
    detail = denied.json()["detail"]
    assert detail["requires_local_fallback_approval"] is True

    allowed = client.post("/v1/sandbox/exec", json={
        "command": "echo hi", "session_id": "s1",
        "allow_local_fallback": True})
    assert allowed.status_code == 200
    body = allowed.json()
    assert body["backend"] == "local"
    assert body["stdout"].strip() == "hi"

    audit = StatePaths(client.app.state.harness.state_dir).tool_audit_file()
    events = [json.loads(line) for line in audit.read_text().splitlines()]
    assert any(row["event"] == "sandbox.local_fallback" for row in events)


def test_an_unpinned_docker_image_is_a_failure_not_a_fallback(
        client, monkeypatch):
    monkeypatch.setattr(sandbox, "docker_status", lambda: {
        "installed": True, "daemon": True, "binary": "/usr/bin/docker",
        "detail": "docker daemon is running"})
    response = client.post("/v1/sandbox/exec", json={
        "command": "echo hi", "session_id": "s1",
        "allow_local_fallback": True})
    assert response.status_code == 409
    assert "not pinned" in response.json()["detail"]


def test_a_request_cannot_override_the_policy_image(client, monkeypatch):
    monkeypatch.setattr(sandbox, "docker_status", lambda: {
        "installed": True, "daemon": False, "binary": "/usr/bin/docker",
        "detail": "docker daemon is not running"})
    response = client.post("/v1/sandbox/exec", json={
        "command": "echo hi", "image": "evil:latest",
        "session_id": "s1", "allow_local_fallback": True})
    assert response.status_code == 403


def test_docker_run_uses_the_pinned_image_and_hardening(
        client, monkeypatch):
    state = StatePaths(client.app.state.harness.state_dir)
    sandbox.SandboxPolicy(state).update(
        docker_image="hive-sandbox:1", image_digest="c" * 64)
    monkeypatch.setattr(sandbox, "docker_status", lambda: {
        "installed": True, "daemon": True, "binary": "/usr/bin/docker",
        "detail": "docker daemon is running"})
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "42\n", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    response = client.post("/v1/sandbox/exec", json={
        "command": "python -c 'print(42)'", "session_id": "s1"})
    assert response.status_code == 200
    body = response.json()
    assert body["backend"] == "docker"
    command = calls[-1]
    assert command[:2] == ["/usr/bin/docker", "run"]
    for flag, value in (("--network", "none"), ("--memory", "512m"),
                        ("--cpus", "1"), ("--pids-limit", "128"),
                        ("--cap-drop", "ALL")):
        assert flag in command and command[command.index(flag) + 1] == value
    assert "--security-opt" in command
    assert f"hive-sandbox:1@sha256:{'c' * 64}" in command


def test_status_reports_the_pin_and_fallback(client, monkeypatch):
    monkeypatch.setattr(sandbox, "docker_status", lambda: {
        "installed": True, "daemon": False, "binary": "/usr/bin/docker",
        "detail": "docker daemon is not running"})
    body = client.get("/v1/sandbox/status").json()
    assert body["docker_installed"] is True
    assert body["docker_daemon"] is False
    assert body["pinned"] is False
    assert body["fallback_requires_approval"] is True
    assert "approval" in body["fallback"]
