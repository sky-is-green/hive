"""The Studio can inspect and change the gateway policy."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from harness.app import create_app
from harness.security import GrantStore
from harness.state_store import StatePaths


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


def test_policy_get_and_put(client):
    body = client.get("/v1/tools/policy").json()
    assert body["policy"]["tier"] == "guarded"
    assert body["grants"] == []
    changed = client.put("/v1/tools/policy", json={
        "tier": "strict",
        "master": {"approval_prompts": False, "sandbox": True},
    })
    assert changed.status_code == 200
    assert changed.json()["policy"]["tier"] == "strict"
    assert changed.json()["policy"]["master"]["approval_prompts"] is False


def test_invalid_tier_is_rejected(client):
    r = client.put("/v1/tools/policy", json={"tier": "everything"})
    assert r.status_code == 422


def test_global_grant_can_be_revoked(client):
    state = StatePaths(client.app.state.harness.state_dir)
    GrantStore(state).grant("hive:bash@1.0.0", scope="global",
                            session_id=None, reason="approved for this box")
    assert client.get("/v1/tools/policy").json()["grants"]
    removed = client.post("/v1/tools/revoke", json={"all": True})
    assert removed.status_code == 200
    assert removed.json()["removed"]["global"] == 1
    assert client.get("/v1/tools/policy").json()["grants"] == []


def test_revoke_needs_a_target(client):
    r = client.post("/v1/tools/revoke", json={})
    assert r.status_code == 422


def test_audit_endpoint_is_empty_by_default(client):
    body = client.get("/v1/tools/audit").json()
    assert body["records"] == []
    assert body["path"].endswith(".jsonl")


def test_agent_status_does_not_claim_gateway_enforcement(client):
    body = client.get("/v1/agent/status").json()
    assert body["gateway_enforced"] is False
    assert body["security_tier"] == "guarded"
    assert "full-access" in body["permission_policy"]
