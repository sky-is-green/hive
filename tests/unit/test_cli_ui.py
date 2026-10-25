"""Switching surfaces must not guess or start a second server."""

from __future__ import annotations

import json

from harness import cli


def test_ui_status_reports_both_surfaces_without_starting_one(capsys):
    assert cli.main(["ui", "status", "--port", "9", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["web"]["running"] is False
    assert payload["switch"]["web"] == "hive ui web"
    assert payload["switch"]["terminal"] == "hive ui terminal"


def test_ui_terminal_requires_a_tty(capsys, monkeypatch):
    monkeypatch.setenv("HIVE_NO_TUI", "1")
    assert cli.main(["ui", "terminal"]) == 1
    assert "interactive terminal" in capsys.readouterr().err


def test_ui_help_names_the_modes(capsys):
    assert cli.main(["help", "ui"]) == 0
    out = capsys.readouterr().out
    assert "web" in out and "terminal" in out and "status" in out


def test_ui_web_passes_the_shared_state_root(monkeypatch):
    calls = []

    def fake_call(argv, cwd=None, env=None):
        calls.append((argv, cwd, env))
        return 0

    monkeypatch.setattr(cli.subprocess, "call", fake_call)
    assert cli.main(["ui", "web", "--mock", "--no-open", "--port", "9999"]) == 0
    argv, cwd, env = calls[0]
    assert argv[:3] == [cli.sys.executable, "-m", "harness"]
    assert "--state-dir" in argv
    assert argv[argv.index("--state-dir") + 1] == str(cli.serve_state.state_dir())
    assert cwd == str(cli.REPO_ROOT)
    assert env["HIVE_TOOL_URL"] == "http://127.0.0.1:9999"
