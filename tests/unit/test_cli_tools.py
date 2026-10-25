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
