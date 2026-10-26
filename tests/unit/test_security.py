"""The tool gateway core: deny-by-default, explicit grants, audit."""

from __future__ import annotations

import json

import pytest

from harness.security import (
    ApprovalError,
    ApprovalGateway,
    GrantStore,
    SecurityError,
    SecurityStore,
    SessionGrants,
    ToolRequest,
    ToolSpec,
    TIERS,
)
from harness.state_store import StatePaths


def _spec(**overrides):
    data = {
        "name": "bash", "source": "hive", "version": "1.0.0",
        "description": "run a command", "read_only": False,
        "side_effect": True, "reason_required": True,
        "network": "none", "filesystem": "workspace",
        "risk": "medium", "sandbox": "required",
    }
    data.update(overrides)
    return ToolSpec.from_dict(data)


def _req(session="conv-1", reason="find the TODO markers", **overrides):
    data = {"session_id": session, "action": "rg -n TODO .",
            "reason": reason, "args": {"argv": ["rg", "-n", "TODO", "."]}}
    data.update(overrides)
    return ToolRequest(**data)


def _state(tmp_path):
    return StatePaths(tmp_path / "state").ensure()


def _gateway(tmp_path, *, tier="guarded", session="conv-1", headless=False,
             now=None):
    state = _state(tmp_path)
    SecurityStore(state).set_tier(tier)
    return ApprovalGateway(state, session_id=session, headless=headless,
                           now=now or (lambda: 1000.0))


def test_default_tier_is_guarded_and_invalid_tier_is_refused(tmp_path):
    store = SecurityStore(_state(tmp_path))
    assert store.read()["tier"] == "guarded"
    with pytest.raises(SecurityError):
        store.set_tier("everything")
    assert store.set_tier("strict")["tier"] == "strict"


def test_corrupt_policy_falls_back_to_guarded(tmp_path):
    state = _state(tmp_path)
    state.security_file().write_text("{broken", encoding="utf-8")
    assert SecurityStore(state).read()["tier"] == "guarded"
    assert list(state.settings_dir.glob("corrupt-*-security.json"))


def test_missing_reason_is_denied_before_prompt(tmp_path):
    gateway = _gateway(tmp_path)
    result = gateway.authorize(_spec(), _req(reason=""))
    assert result["status"] == "denied"
    assert result["reason"] == "missing reason"
    assert gateway.pending() == []


def test_generic_reason_is_denied(tmp_path):
    gateway = _gateway(tmp_path)
    result = gateway.authorize(_spec(), _req(reason="test"))
    assert result["status"] == "denied"


def test_read_only_headless_is_allowed(tmp_path):
    gateway = _gateway(tmp_path, headless=True)
    spec = _spec(read_only=True, side_effect=False, reason_required=False)
    result = gateway.authorize(spec, _req(reason=""))
    assert result["status"] == "approved"
    assert result["reason"] == "read-only"


def test_private_read_only_needs_approval_even_in_open(tmp_path):
    gateway = _gateway(tmp_path, tier="open")
    spec = _spec(name="web_fetch", read_only=True, side_effect=False,
                 reason_required=False, network="private")
    pending = gateway.authorize(spec, _req(reason="read the private page"))
    assert pending["status"] == "pending"
    assert gateway.decide(pending["request_id"], "once")["status"] == "approved"


def test_private_grants_are_scoped_to_one_origin(tmp_path):
    gateway = _gateway(tmp_path)
    spec = _spec(name="web_fetch", read_only=True, side_effect=False,
                 reason_required=False, network="private")
    local = _req(reason="read the local console",
                 resource="http://localhost:8765")
    pending = gateway.authorize(spec, local)
    assert pending["resource"] == "http://localhost:8765"
    gateway.decide(pending["request_id"], "always", scope="global")
    assert gateway.authorize(spec, local)["status"] == "approved"

    metadata = _req(reason="read instance metadata",
                    resource="http://169.254.169.254")
    assert gateway.authorize(spec, metadata)["status"] == "pending"


def test_side_effect_headless_is_denied(tmp_path):
    gateway = _gateway(tmp_path, headless=True)
    result = gateway.authorize(_spec(), _req())
    assert result["status"] == "denied"
    assert result["reason"] == "headless side-effect"


def test_guarded_side_effect_is_pending_until_decision(tmp_path):
    gateway = _gateway(tmp_path)
    pending = gateway.authorize(_spec(), _req())
    assert pending["status"] == "pending"
    assert pending["request_id"].startswith("req-")
    assert len(gateway.pending()) == 1
    approved = gateway.decide(pending["request_id"], "once")
    assert approved["status"] == "approved"
    assert gateway.pending() == []


def test_session_grant_is_scoped_to_one_session(tmp_path):
    gateway = _gateway(tmp_path, session="conv-1")
    pending = gateway.authorize(_spec(), _req("conv-1"))
    gateway.decide(pending["request_id"], "session")
    assert gateway.authorize(_spec(), _req("conv-1"))["status"] == "approved"

    other = ApprovalGateway(gateway.state, session_id="conv-2",
                            grants=gateway.grants)
    assert other.authorize(_spec(), _req("conv-2"))["status"] == "pending"


def test_global_grant_persists_across_gateway_restart_and_version_bump(tmp_path):
    gateway = _gateway(tmp_path)
    pending = gateway.authorize(_spec(), _req())
    gateway.decide(pending["request_id"], "always", scope="global")

    restarted = ApprovalGateway(gateway.state, session_id="conv-1")
    assert restarted.authorize(_spec(), _req())["status"] == "approved"

    version_two = _spec(version="1.1.0")
    assert restarted.authorize(version_two, _req())["status"] == "pending"


def test_strict_tier_requires_a_fresh_decision_even_with_global_grant(tmp_path):
    gateway = _gateway(tmp_path, tier="guarded")
    pending = gateway.authorize(_spec(), _req())
    gateway.decide(pending["request_id"], "always", scope="global")
    gateway.security.set_tier("strict")
    assert gateway.authorize(_spec(), _req())["status"] == "pending"


def test_open_tier_allows_public_but_not_private_without_grant(tmp_path):
    gateway = _gateway(tmp_path, tier="open")
    public = _spec(network="public")
    private = _spec(network="private")
    assert gateway.authorize(public, _req())["status"] == "approved"
    pending = gateway.authorize(private, _req())
    assert pending["status"] == "pending"
    gateway.decide(pending["request_id"], "session")
    assert gateway.authorize(private, _req())["status"] == "approved"


def test_deny_and_timeout_are_structured(tmp_path):
    clock = {"now": 1000.0}
    gateway = _gateway(tmp_path, now=lambda: clock["now"])
    one = gateway.authorize(_spec(), _req(timeout_s=10))
    denied = gateway.decide(one["request_id"], "deny")
    assert denied["status"] == "denied"

    two = gateway.authorize(_spec(), _req(timeout_s=10))
    clock["now"] += 11
    expired = gateway.decide(two["request_id"], "once")
    assert expired["status"] == "denied"
    assert expired["reason"] == "timeout"


def test_execute_is_the_only_execution_path_and_pending_does_not_run(tmp_path):
    gateway = _gateway(tmp_path)
    calls = []
    pending = gateway.execute(_spec(), _req(), lambda request: calls.append(request))
    assert pending["status"] == "pending"
    assert calls == []

    gateway.decide(pending["request_id"], "session")
    result = gateway.execute(_spec(), _req(), lambda request: {"ok": True})
    assert result["status"] == "approved"
    assert result["result"] == {"ok": True}


def test_global_grant_revoke_and_listing(tmp_path):
    state = _state(tmp_path)
    grants = GrantStore(state, sessions=SessionGrants())
    grants.grant("hive:bash@1.0.0", scope="global", session_id=None,
                 reason="approved for this box")
    assert grants.has("hive:bash@1.0.0")
    assert grants.list()[0]["scope"] == "global"
    removed = grants.revoke("hive:bash@1.0.0")
    assert removed["global"] == 1
    assert not grants.has("hive:bash@1.0.0")


def test_missing_approval_is_a_clear_error(tmp_path):
    gateway = _gateway(tmp_path)
    with pytest.raises(ApprovalError, match="no pending"):
        gateway.decide("req-nope", "once")


def test_audit_is_append_only_and_hashes_args(tmp_path):
    gateway = _gateway(tmp_path)
    pending = gateway.authorize(_spec(), _req())
    gateway.decide(pending["request_id"], "once")
    text = gateway.state.tool_audit_file().read_text(encoding="utf-8")
    assert "tool.pending" in text and "tool.approved" in text
    assert "args_sha256" in text
    assert '"argv"' not in text  # arguments are hashed, not written raw
    assert len(text.splitlines()) >= 2


def test_tool_spec_unknown_metadata_defaults_to_side_effect():
    spec = ToolSpec.from_dict({"name": "mystery"})
    assert spec.is_side_effect is True
    assert spec.needs_reason is True
    assert spec.network == "private"
    assert spec.filesystem == "host"
    assert spec.risk == "high"
