"""The CLI exposes the activity ledger, tripwires, and emergency cease."""

from __future__ import annotations

import json

from harness import cli
from harness.activity import ActivityLedger, ActivityRules
from harness.state_store import StatePaths


def test_activity_rules_set_enable_disable(tmp_path, capsys):
    state = str(tmp_path / "state")
    rules = tmp_path / "rules.json"
    rules.write_text(json.dumps({
        "enabled": True,
        "rules": [{"id": "keys", "action": "cease",
                   "match": {"field": "result", "regex": "PRIVATE KEY"}}],
    }), encoding="utf-8")
    assert cli.main(["activity", "rules", "--set", str(rules),
                     "--state-dir", state, "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["enabled"] is True
    assert out["rules"][0]["id"] == "keys"

    assert cli.main(["activity", "rules", "--disable",
                     "--state-dir", state, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["enabled"] is False
    assert cli.main(["activity", "rules", "--enable",
                     "--state-dir", state, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["enabled"] is True


def test_activity_tail_reads_recorded_events(tmp_path, capsys):
    state = StatePaths(tmp_path / "state").ensure()
    ActivityLedger(state).record({
        "type": "tool", "tool": "bash", "phase": "call",
        "args": {"command": "ls"},
    }, conversation_id="c1", session_id="s1")
    assert cli.main(["activity", "tail", "--state-dir", str(state.root),
                     "--conversation", "c1", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["tool"] == "bash"
    assert rows[0]["args"]["command"] == "ls"


def test_activity_cease_needs_confirmation_and_target(tmp_path, capsys):
    assert cli.main(["activity", "cease", "--conversation", "c1"]) == 2
    assert "pass --yes" in capsys.readouterr().err
    assert cli.main(["activity", "cease", "--yes"]) == 2
    assert "name --conversation" in capsys.readouterr().err


def test_activity_cease_and_resume_use_the_studio(monkeypatch, capsys):
    calls = []

    def fake_api(args, path, *, method="GET", body=None):
        calls.append((path, method, body))
        if body and body.get("action") == "cease":
            return {"ok": True, "block": {"conversation_id": "c1"}}
        return {"ok": True, "removed": 1}

    monkeypatch.setattr(cli, "_tool_api", fake_api)
    assert cli.main(["activity", "cease", "--conversation", "c1",
                     "--reason", "saw it", "--yes", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True
    assert calls[0][0] == "/v1/activity/control"
    assert calls[0][2]["action"] == "cease"
    assert calls[0][2]["reason"] == "saw it"

    assert cli.main(["activity", "resume", "--conversation", "c1",
                     "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["removed"] == 1
    assert calls[1][2]["action"] == "resume"
