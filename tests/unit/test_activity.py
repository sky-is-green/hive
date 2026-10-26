"""Unit tests for the activity ledger, tripwires, and emergency cease."""

from __future__ import annotations

import json

import pytest

from harness.activity import (
    ActivityError,
    ActivityHub,
    ActivityLedger,
    ActivityRules,
    CeaseStore,
)
from harness.security import GrantStore, GatewayHub, ToolRequest, ToolSpec
from harness.state_store import StatePaths


@pytest.fixture()
def state(tmp_path):
    return StatePaths(tmp_path / "state").ensure()


@pytest.fixture()
def client(tmp_path):
    from fastapi.testclient import TestClient
    from harness.app import create_app

    app = create_app(
        providers_file=tmp_path / "providers.local.json",
        engines_file=tmp_path / "engines.local.json",
        runs_root=tmp_path / "runs",
        log_dir=str(tmp_path / "logs"),
        state_dir=str(tmp_path / "state"),
    )
    return TestClient(app)


def test_ledger_records_redacts_publishes_and_replays(state):
    ledger = ActivityLedger(state)
    subscriber = ledger.subscribe()
    row = ledger.record({
        "type": "tool", "tool": "bash", "phase": "call",
        "args": {"command": "ls", "api_key": "secret"},
    }, conversation_id="c1", session_id="s1")
    assert row["conversation_id"] == "c1"
    assert row["session_id"] == "s1"
    assert row["args"]["api_key"] == "[redacted]"
    assert subscriber.get_nowait()["args"]["command"] == "ls"
    ledger.unsubscribe(subscriber)

    replayed = ledger.read_file("c1", limit=10)
    assert len(replayed) == 1
    assert replayed[0]["args"]["command"] == "ls"
    on_disk = [json.loads(line) for line in ledger.path().read_text().splitlines()]
    assert on_disk[0]["args"]["api_key"] == "[redacted]"


def test_rules_are_disabled_by_default_and_match_when_enabled(state):
    rules = ActivityRules(state)
    assert rules.read()["enabled"] is False
    assert rules.matches({"type": "tool", "tool": "bash"}) == []

    rules.write([{
        "id": "no-recursive-root-delete",
        "action": "cease",
        "match": {"type": "tool", "phase": "call", "tool": "bash",
                  "field": "args", "regex": r"rm\s+-rf\s+/(\s|$)"},
        "reason": "recursive delete of the filesystem root",
    }], enabled=True)
    hit = rules.matches({
        "type": "tool", "phase": "call", "tool": "bash",
        "args": {"command": "rm -rf / --no-preserve-root"},
    })
    assert len(hit) == 1
    assert hit[0]["rule"]["id"] == "no-recursive-root-delete"

    assert rules.matches({
        "type": "tool", "phase": "call", "tool": "bash",
        "args": {"command": "ls -la"},
    }) == []

    rules.set_enabled(False)
    assert rules.matches({
        "type": "tool", "phase": "call", "tool": "bash",
        "args": {"command": "rm -rf /"},
    }) == []


def test_rules_reject_bad_regex_and_unknown_match_keys(state):
    rules = ActivityRules(state)
    with pytest.raises(ActivityError, match="invalid regex"):
        rules.write([{
            "id": "bad", "action": "warn",
            "match": {"regex": "("},
        }], enabled=True)
    with pytest.raises(ActivityError, match="unknown match"):
        rules.write([{
            "id": "bad", "action": "warn",
            "match": {"nope": "x"},
        }], enabled=True)


def test_cease_store_blocks_and_resumes(state):
    store = CeaseStore(state)
    assert not store.is_blocked(session_id="agent-c1-g1")
    store.block(session_id="agent-c1-g1", conversation_id="c1", reason="test")
    assert store.is_blocked(session_id="agent-c1-g1")
    assert store.is_blocked(session_id="agent-c1-g2")  # same conversation
    assert store.is_blocked(conversation_id="c1")
    assert store.clear(session_id="agent-c1-g2") == 1
    assert not store.is_blocked(session_id="agent-c1-g1")
    assert not store.is_blocked(conversation_id="c1")


def test_hub_cease_calls_stop_revokes_session_grants_and_records(state):
    grants = GrantStore(state)
    grants.grant("dsh:bash@1.0.0", scope="session", session_id="s1",
                 reason="test grant")
    assert grants.list(session_id="s1")

    hub = ActivityHub(state, grants=grants)
    calls = []

    def stop(**fields):
        calls.append(fields)
        return {"ok": True}

    hub.register_cease_callback(stop)
    result = hub.cease_now(conversation_id="c1", session_id="s1",
                           reason="test cease")
    assert result["ok"] is True
    assert calls and calls[0]["session_id"] == "s1"
    assert hub.cease.is_blocked(session_id="s1")
    assert grants.list(session_id="s1") == []
    assert result["event"]["type"] == "cease"
    assert hub.ledger.read_file("c1")[-1]["type"] == "cease"


def test_hub_tripwire_ceases_and_is_not_recursive(state):
    rules = ActivityRules(state)
    rules.write([{
        "id": "private-key-output",
        "action": "cease",
        "match": {"type": "tool", "phase": "result", "field": "result",
                  "regex": "BEGIN (RSA |OPENSSH )?PRIVATE KEY"},
        "reason": "private key material in tool output",
    }], enabled=True)
    hub = ActivityHub(state, rules=rules)
    calls = []
    hub.register_cease_callback(lambda **fields: calls.append(fields) or {"ok": True})

    hub.record({
        "type": "tool", "tool": "bash", "phase": "result",
        "result": "-----BEGIN OPENSSH PRIVATE KEY-----\nabc",
    }, conversation_id="c1", session_id="s1")

    assert calls
    assert hub.cease.is_blocked(session_id="s1")
    kinds = [row["type"] for row in hub.ledger.read_file("c1")]
    assert "tripwire" in kinds
    assert "cease" in kinds
    # The cease record must not re-enter the rule engine.
    assert sum(1 for kind in kinds if kind == "cease") == 1


def test_gateway_denies_tool_calls_after_cease(state):
    CeaseStore(state).block(session_id="s1", reason="test")
    hub = GatewayHub(state)
    gateway = hub.gateway("s1")
    spec = ToolSpec(name="read", source="dsh", read_only=True,
                    network="public", filesystem="workspace")
    request = ToolRequest(session_id="s1", action="read: a.txt",
                          reason="read a file")
    result = gateway.authorize(spec, request)
    assert result["status"] == "denied"
    assert "cease" in result["reason"]


def test_activity_hub_status_reports_rules_and_blocks(state):
    hub = ActivityHub(state)
    status = hub.status()
    assert status["rules"]["enabled"] is False
    assert status["blocks"] == []


def test_activity_rules_api_round_trip(client):
    current = client.get("/v1/activity/rules").json()
    assert current["enabled"] is False
    changed = client.put("/v1/activity/rules", json={
        "enabled": True,
        "rules": [{
            "id": "no-private-keys", "action": "cease",
            "match": {"type": "tool", "field": "result",
                      "regex": "BEGIN .*PRIVATE KEY"},
        }],
    })
    assert changed.status_code == 200, changed.text
    body = client.get("/v1/activity/rules").json()
    assert body["enabled"] is True
    assert body["rules"][0]["id"] == "no-private-keys"


def test_activity_control_needs_a_target(client):
    r = client.post("/v1/activity/control", json={"action": "cease"})
    assert r.status_code == 422


def test_activity_cease_blocks_and_resume_clears(client, monkeypatch):
    import harness.sandbox as sandbox

    monkeypatch.setattr(sandbox, "docker_status", lambda: {
        "installed": False, "daemon": False, "binary": "",
        "detail": "docker is not installed", "host": "unix:///none",
        "rootless": False,
    })
    ceased = client.post("/v1/activity/control", json={
        "action": "cease", "conversation_id": "c1",
        "reason": "test cease",
    })
    assert ceased.status_code == 200, ceased.text
    body = ceased.json()
    assert body["ok"] is True
    assert client.get("/v1/activity").json()["blocks"]
    resumed = client.post("/v1/activity/control", json={
        "action": "resume", "conversation_id": "c1",
    })
    assert resumed.status_code == 200, resumed.text
    assert resumed.json()["removed"] == 1
    assert client.get("/v1/activity").json()["blocks"] == []
