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
    assert body["gateway_configured"] is False
    assert body["security_tier"] == "guarded"
    assert "blocked" in body["permission_policy"]


def test_registered_mcp_metadata_overrides_conservative_defaults(client):
    body = {
        "session_id": "conv-mcp",
        "tool": {
            "name": "mcp__github__search", "source": "mcp",
            "version": "1.0.0", "read_only": False, "side_effect": True,
            "reason_required": True, "network": "private",
            "filesystem": "host", "risk": "high", "sandbox": "required",
        },
        "action": "mcp__github__search: repos",
        "reason": "search the issue tracker", "args": {"query": "repos"},
    }
    pending = client.post("/v1/tools/call", json=body)
    assert pending.status_code == 200
    assert pending.json()["status"] == "pending"

    registered = client.post("/v1/tools/register", json={
        "tool": {
            "name": "mcp__github__search", "source": "mcp",
            "version": "1.0.0", "description": "Search GitHub",
            "read_only": True, "side_effect": False,
            "reason_required": False, "network": "public",
            "filesystem": "none", "risk": "low", "sandbox": "optional",
        },
        "verified": True, "server": "github", "note": "test",
    })
    assert registered.status_code == 200
    assert registered.json()["verified"] is True
    assert client.get("/v1/tools").json()["tools"]

    approved = client.post("/v1/tools/call", json=body)
    assert approved.status_code == 200
    assert approved.json()["status"] == "approved"
    assert approved.json()["reason"] == "read-only"

    removed = client.post("/v1/tools/unregister", json={
        "source": "mcp", "name": "mcp__github__search"})
    assert removed.json()["removed"] is True
    assert client.post("/v1/tools/call", json=body).json()["status"] == "pending"


def test_agent_status_claims_enforcement_only_after_preflight(client):
    service = client.app.state.agent
    service._hook_installed = True
    service._hook_preflight_key = ("/usr/bin/dsh", "/tmp/dsh-home")
    body = client.get("/v1/agent/status").json()
    assert body["gateway_enforced"] is True
    assert body["gateway_configured"] is True
    assert body["gateway_hook"]["verified"] is True


def _call(client, **overrides):
    body = {
        "session_id": "conv-1",
        "tool": {
            "name": "bash", "source": "dsh", "version": "1.0.0",
            "read_only": False, "side_effect": True, "reason_required": True,
            "network": "none", "filesystem": "workspace",
            "risk": "high", "sandbox": "required",
        },
        "action": "bash: rg -n TODO .",
        "reason": "find the TODO markers",
        "args": {"command": "rg -n TODO ."},
    }
    body.update(overrides)
    return client.post("/v1/tools/call", json=body)


def test_call_pending_decision_and_grant(client):
    pending = _call(client)
    assert pending.status_code == 200
    body = pending.json()
    assert body["status"] == "pending"
    request_id = body["request_id"]

    assert client.get("/v1/tools/pending", params={
        "session_id": "conv-1"}).json()["requests"][0]["request_id"] == request_id
    decided = client.post("/v1/tools/decide", json={
        "session_id": "conv-1", "request_id": request_id,
        "choice": "session"})
    assert decided.status_code == 200
    settled = client.get("/v1/tools/decision", params={
        "session_id": "conv-1", "request_id": request_id})
    assert settled.json()["status"] == "approved"
    assert _call(client).json()["status"] == "approved"


def test_call_missing_reason_and_headless_are_denied(client):
    denied = _call(client, reason="").json()
    assert denied["status"] == "denied"
    headless = _call(client, headless=True).json()
    assert headless["status"] == "denied"
    assert headless["reason"] == "headless side-effect"
