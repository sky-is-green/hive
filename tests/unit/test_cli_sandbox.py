"""The sandbox policy is inspectable and pinnable from the terminal."""

from __future__ import annotations

import json

from harness import cli


def test_sandbox_status_and_policy_round_trip(tmp_path, capsys):
    state = str(tmp_path / "state")
    assert cli.main(["sandbox", "status", "--state-dir", state, "--json"]) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["policy"]["local_fallback"] == "approval"
    assert body["pinned"] is False

    assert cli.main(["sandbox", "policy", "--local-fallback", "deny",
                     "--state-dir", state, "--json"]) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["local_fallback"] == "deny"

    assert cli.main(["sandbox", "policy", "--state-dir", state,
                     "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["local_fallback"] == "deny"


def test_sandbox_pin_accepts_an_offline_digest(tmp_path, capsys):
    state = str(tmp_path / "state")
    digest = "d" * 64
    assert cli.main(["sandbox", "pin", "--image", "hive-sandbox:1",
                     "--digest", digest, "--state-dir", state, "--json"]) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["image_digest"] == digest
    assert cli.main(["sandbox", "status", "--state-dir", state,
                     "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["pinned"] is True


def test_sandbox_policy_rejects_an_unpinned_digest(tmp_path, capsys):
    state = str(tmp_path / "state")
    assert cli.main(["sandbox", "policy", "--digest", "nope",
                     "--state-dir", state]) == 2
    assert "64-character" in capsys.readouterr().err
