"""The CLI exposes the security tier and revocable grants."""

from __future__ import annotations

import json

from harness import cli
from harness.security import GrantStore
from harness.state_store import StatePaths


def test_tools_policy_status_and_grants(tmp_path, capsys):
    state = str(tmp_path / "state")
    assert cli.main(["tools", "policy", "--state-dir", state, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["tier"] == "guarded"

    assert cli.main(["tools", "policy", "strict", "--state-dir", state,
                     "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["tier"] == "strict"

    assert cli.main(["tools", "status", "--state-dir", state, "--json"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["tier"] == "strict"
    assert status["master"]["approval_prompts"] is True


def test_tools_grants_and_revoke(tmp_path, capsys):
    state = StatePaths(tmp_path / "state").ensure()
    GrantStore(state).grant("hive:bash@1.0.0", scope="global",
                            session_id=None, reason="approved for this box")
    args = ["tools", "grants", "--state-dir", str(state.root), "--json"]
    assert cli.main(args) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["tool_identity"] == "hive:bash@1.0.0"

    assert cli.main(["tools", "revoke", "hive:bash@1.0.0", "--yes",
                     "--state-dir", str(state.root), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["global"] == 1
    assert cli.main(["tools", "grants", "--state-dir", str(state.root),
                     "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == []


def test_tools_revoke_needs_confirmation_and_target(tmp_path, capsys):
    state = str(tmp_path / "state")
    assert cli.main(["tools", "revoke", "--state-dir", state]) == 2
    assert "name a tool" in capsys.readouterr().err
    assert cli.main(["tools", "revoke", "hive:bash@1.0.0",
                     "--state-dir", state]) == 2
    assert "pass --yes" in capsys.readouterr().err


def test_tools_audit_is_empty_then_readable(tmp_path, capsys):
    state = str(tmp_path / "state")
    assert cli.main(["tools", "audit", "--state-dir", state, "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == []


def test_tools_pending_and_decide_use_the_live_studio(monkeypatch, capsys):
    calls = []

    def fake_api(args, path, *, method="GET", body=None):
        calls.append((path, method, body))
        if path.startswith("/v1/tools/pending"):
            return {"requests": [{"request_id": "req-1", "session_id": "conv-1",
                                  "tool_identity": "dsh:bash@1.0.0",
                                  "action": "bash: rg TODO", "reason": "find"}]}
        return {"status": "approved", "reason": "once"}

    monkeypatch.setattr(cli, "_tool_api", fake_api)
    assert cli.main(["tools", "pending", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["request_id"] == "req-1"
    assert calls[0][0] == "/v1/tools/pending"

    assert cli.main(["tools", "decide", "req-1", "--session", "conv-1",
                     "--choice", "once", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "approved"
    assert calls[1] == ("/v1/tools/decide", "POST", {
        "session_id": "conv-1", "request_id": "req-1",
        "choice": "once", "scope": "session"})


def test_tools_pending_is_readable_when_gateway_is_down(monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise RuntimeError("gateway unreachable")

    monkeypatch.setattr(cli, "_tool_api", fail)
    assert cli.main(["tools", "pending"]) == 1
    assert "unreachable" in capsys.readouterr().err
